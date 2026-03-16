import asyncio
import os
import logging
import uuid
import fitz  # PyMuPDF
from typing import Callable, Coroutine, Any, List

# In a real app, these imports would be absolute from clirag.storage and clirag.ingestion
from clirag.ingestion.chunker import RecursiveTextChunker

logger = logging.getLogger("IngestionWorkers")

class IngestionWorker:
    """
    Handles Asynchronous, Zero-LLM Data Ingestion.
    Supports Typer/Rich Async hooks & Deduplication logic.
    """
    def __init__(self, db_storage_instance: Any, nlp_engine_instance: Any, kuzu_graph_instance: Any = None):
        # We pass instances rather than constructing to share DB connections in the CLI state
        self.db = db_storage_instance
        self.nlp = nlp_engine_instance
        self.kuzu_graph = kuzu_graph_instance
        
        # EMERGENCY PATCH 2: IMPLEMENT MUTEX RAM LOCKS
        # Prevents OOM crashes by forcing strict linear chunk processing through the heavy NLP / C++ modules.
        self.ram_lock = asyncio.Lock()
        
        # Initialize the chunker for safe token limits
        self.chunker = RecursiveTextChunker(chunk_size=500, overlap=50)

    async def ingest_file_async(self, filepath: str, progress_callback: Callable[[str, float], Coroutine[Any, Any, None]] = None) -> str:
        """
        Asynchronously ingests a file.
        Yields status updates cleanly to a 'progress_callback' to keep Rich CLI UI active.
        """
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find {filepath}")

        # --- Phase 1: File Verification & Deduplication ---
        if progress_callback:
            await progress_callback(f"Computing SHA-256 for '{os.path.basename(filepath)}'...", 5.0)
            
        # This is a CPU bound task, offload to another thread to avoid blocking Typer CLI loop
        file_hash = await asyncio.to_thread(self.db.compute_file_hash, filepath)
        
        # Deduplication Check
        if self.db.is_file_ingested(file_hash):
            msg = f"Skipping '{os.path.basename(filepath)}'. File hash already exists in DuckDB."
            if progress_callback:
                await progress_callback(msg, 100.0)
            logger.info(msg)
            return "SKIPPED_DEDUPLICATED"

        # Initialize new Doc ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        self.db.insert_document(doc_id, os.path.basename(filepath), file_hash)

        # --- Phase 2: Extraction (PDF/OCR) ---
        if progress_callback:
            await progress_callback(f"Extracting Raw Text from '{os.path.basename(filepath)}'...", 20.0)
            
        # Wait for file IO bounds
        # Use our new real recursive chunker instead of the mock logic
        # Phase 2.1: Extract Text and Tables
        extracted_results = await asyncio.to_thread(self._extract_content_with_tables, filepath)
        total_chunks = len(extracted_results)
        
        # --- Phase 3: NLP Entity Extraction & Vectorization (Parallel) ---
        if progress_callback:
            await progress_callback("Initializing AI Warriors NLP Engine...", 40.0)

        for i, item in enumerate(extracted_results):
            text_chunk = item["text"]
            is_table = item.get("is_table", False)
            chunk_id = f"chunk_{doc_id}_{i}"
            
            # AI Warriors Optimization: Skip heavy GLiNER on tables and tiny chunks
            should_skip_nlp = is_table or len(text_chunk.strip()) < 100
            
            async with self.ram_lock:
                vector_data = await asyncio.to_thread(self.nlp.extract_and_vectorize, text_chunk, skip_nlp=should_skip_nlp)
            
            # ... rest of the loop remains similar ...
            await asyncio.to_thread(self.db.insert_chunk, doc_id, chunk_id, text_chunk, vector_data['dense_embedding'])
            await asyncio.to_thread(self.db.insert_colbert_embeddings, chunk_id, vector_data['colbert_embeddings'])
            if self.kuzu_graph and vector_data['graph_entities']:
                for ent in vector_data['graph_entities']:
                    ent['doc_id'] = doc_id
                await asyncio.to_thread(self.kuzu_graph.upsert_entities, vector_data['graph_entities'])

            if progress_callback:
                progress = 40.0 + ((i + 1) / total_chunks) * 60.0
                await progress_callback(f"Processed {'Table' if is_table else 'Text'} Chunk {i+1}/{total_chunks}", progress)

        if progress_callback:
            await progress_callback("Refreshing AI Warriors Search Index...", 95.0)
        self.db.refresh_fts_index()

        if progress_callback:
            await progress_callback(f"Ingestion Complete for '{os.path.basename(filepath)}'.", 100.0)
            
        return doc_id

    def _extract_content_with_tables(self, filepath: str) -> List[dict]:
        """
        AI Warriors High-Speed Extractor.
        Detects tables using PyMuPDF and handles them as distinct, tagged chunks.
        """
        results = []
        
        try:
            if filepath.lower().endswith(".pdf"):
                doc = fitz.open(filepath)
                for page in doc:
                    # 1. Extract Tables first
                    tabs = page.find_tables()
                    table_texts = []
                    for tab in tabs:
                        # Convert table to markdown-like format
                        df = tab.to_pandas()
                        md = f"\n[TABLE_DATA]\n{df.to_markdown(index=False)}\n"
                        table_texts.append(md)
                        results.append({"text": md, "is_table": True})
                    
                    # 2. Extract remaining text (blocking out table areas to avoid duplication if possible)
                    # Simplified: extract all text and deduplicate or just append
                    page_text = page.get_text("text") or ""
                    if page_text.strip():
                        # We chunk the non-table text
                        text_chunks = self.chunker.split_text(page_text)
                        for c in text_chunks:
                            results.append({"text": c, "is_table": False})
                doc.close()
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()
                    text_chunks = self.chunker.split_text(full_text)
                    for c in text_chunks:
                        results.append({"text": c, "is_table": False})
                    
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return []
            
        return results

# Quick test scaffolding
if __name__ == "__main__":
    class MockStorage:
        def compute_file_hash(self, fp): return "hash123"
        def is_file_ingested(self, h): return False
        def insert_document(self, d, f, h): print(f"Registered doc {d}")
        def insert_chunk(self, d, c, t, v): pass
        def insert_colbert_embeddings(self, c, v): pass
    
    class MockNLP:
        def extract_and_vectorize(self, text):
            # Returns Dense and List[List[float]] format
            return {
                "dense_embedding": [0.1, 0.2, 0.3], 
                "colbert_embeddings": [[0.1, 0.1], [0.2, 0.2]],
                "graph_entities": [{"head": "A", "tail": "B", "rel": "is"}]
            }

    async def mock_cli_progress(msg: str, percent: float):
        print(f"[RICH UI] {percent:0.1f}% | {msg}")

    async def run_test():
        with open("test_dummy.txt", "w") as f: f.write("dummy")
        
        worker = IngestionWorker(MockStorage(), MockNLP())
        print("\n--- Typer CLI Ingestion Simulation ---")
        await worker.ingest_file_async("test_dummy.txt", progress_callback=mock_cli_progress)
        print("--- Success ---\n")
        
        os.remove("test_dummy.txt")

    asyncio.run(run_test())
