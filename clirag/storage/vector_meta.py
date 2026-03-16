import duckdb
import os
import hashlib
from typing import List, Dict, Any, Optional

class VectorMetadataStorage:
    """
    Manages DuckDB based embedded analytical storage for CLIRAG.
    Satisfies Pillar 4: Disk-Bound Storage (SSD constrained)
    """
    
    def __init__(self, db_path: str = "data/duckdb/clirag_meta.duckdb", read_only: bool = False):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            self._initialize_schema()

    def _initialize_schema(self):
        """Creates the necessary tables if they do not exist."""
        
        # Table 1: Document Metadata (Crucial for File Deduplication)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS Document_Metadata (
                doc_id VARCHAR PRIMARY KEY,
                filename VARCHAR NOT NULL,
                file_hash VARCHAR UNIQUE NOT NULL, -- SHA256 for Deduplication
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: Dense Vectors (BM25 / Standard Dense embeddings like MiniLM)
        # Note: Depending on DuckDB version and extensions, vector storage might vary.
        # Float[] representation for dense vectors.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS Dense_Vectors (
                chunk_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR REFERENCES Document_Metadata(doc_id),
                text_content TEXT,
                dense_embedding FLOAT[] -- e.g., 384d for all-MiniLM
            )
        """)

        # Table 3: ColBERT Multi-Vectors (Late Interaction Math)
        # Specific for Route 3: Needle-in-a-Haystack deep searches. 
        # Token-level embeddings stored as a list of lists (Flattened or nested depending on setup).
        # We store FLOAT[][] which represents [num_tokens][embedding_dim]
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ColBERT_Embeddings (
                chunk_id VARCHAR REFERENCES Dense_Vectors(chunk_id),
                token_vectors FLOAT[][], -- e.g., [128][128] for 128 tokens, 128d
                PRIMARY KEY (chunk_id)
            )
        """)

        # EMERGENCY PATCH 1: BM25 FTS IN DUCKDB
        # Satisfies: Route 1 (BM25) FTS Index requirement
        self.conn.execute("INSTALL fts;")
        self.conn.execute("LOAD fts;")
        # Only create FTS index if it doesn't already exist.
        # The index is properly refreshed after each ingestion via refresh_fts_index().
        try:
            self.conn.execute(
                "SELECT * FROM fts_main_Dense_Vectors.match_bm25('__ping__', fields := 'text_content') LIMIT 0"
            )
        except Exception:
            # Index does not exist yet — create it
            try:
                self.conn.execute("""
                    PRAGMA create_fts_index(
                        'Dense_Vectors', 'chunk_id', 'text_content', overwrite=1
                    );
                """)
            except Exception:
                pass  # Table may be empty on first run


    def compute_file_hash(self, filepath: str) -> str:
        """Computes the SHA-256 hash of a file for exact match deduplication."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_file_ingested(self, file_hash: str) -> bool:
        """
        Checks if a file has already been ingested to prevent Graph/Vector duplication.
        Satisfies: FILE & CHUNK DEDUPLICATION Addendum
        """
        result = self.conn.execute(
            "SELECT count(*) FROM Document_Metadata WHERE file_hash = ?", 
            [file_hash]
        ).fetchone()
        
        return result[0] > 0 if result else False

    def insert_document(self, doc_id: str, filename: str, file_hash: str):
        """Registers a new document."""
        self.conn.execute(
            "INSERT INTO Document_Metadata (doc_id, filename, file_hash) VALUES (?, ?, ?)",
            [doc_id, filename, file_hash]
        )

    def insert_chunk(self, doc_id: str, chunk_id: str, text: str, dense_embedding: List[float]):
        """Inserts a textual chunk and its standard dense embedding."""
        # DuckDB handles Python lists natively when casting to ARRAY type if aligned.
        self.conn.execute(
            "INSERT INTO Dense_Vectors (chunk_id, doc_id, text_content, dense_embedding) VALUES (?, ?, ?, ?)",
            [chunk_id, doc_id, text, dense_embedding]
        )

    def insert_colbert_embeddings(self, chunk_id: str, token_vectors: List[List[float]]):
        """
        Inserts Multi-Vector ColBERT embeddings.
        Supports Route 3 Late-Interaction Math.
        """
        self.conn.execute(
             "INSERT INTO ColBERT_Embeddings (chunk_id, token_vectors) VALUES (?, ?)",
             [chunk_id, token_vectors]
        )

    def refresh_fts_index(self):
        """
        DuckDB FTS indexes do not auto-update on inserts.
        This re-creates the FTS index, called after a document is fully ingested.
        Satisfies: DUCKDB FTS INDEXING BUG.
        """
        try:
            self.conn.execute("PRAGMA drop_fts_index('Dense_Vectors');")
        except Exception:
            pass # Ignore if doesn't exist yet
            
        self.conn.execute("""
            PRAGMA create_fts_index(
                'Dense_Vectors', 'chunk_id', 'text_content', overwrite=1
            );
        """)

    def delete_document(self, doc_id: str):
        """
        Cascaded deletion of all document data.
        """
        # 1. Delete from sub-tables first
        self.conn.execute("DELETE FROM ColBERT_Embeddings WHERE chunk_id IN (SELECT chunk_id FROM Dense_Vectors WHERE doc_id = ?)", [doc_id])
        self.conn.execute("DELETE FROM Dense_Vectors WHERE doc_id = ?", [doc_id])
        # 2. Delete meta
        self.conn.execute("DELETE FROM Document_Metadata WHERE doc_id = ?", [doc_id])
        # 3. Refresh FTS
        self.refresh_fts_index()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns a list of all ingested documents."""
        res = self.conn.execute("SELECT doc_id, filename, ingested_at FROM Document_Metadata").fetchall()
        return [{"doc_id": r[0], "filename": r[1], "ingested_at": r[2]} for r in res]

    def get_doc_id_by_filename(self, filename: str) -> Optional[str]:
        """Resolves a filename to its doc_id if it exists."""
        res = self.conn.execute(
            "SELECT doc_id FROM Document_Metadata WHERE filename = ? OR filename LIKE ?", 
            [filename, f"%{filename}%"]
        ).fetchone()
        return res[0] if res else None

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    import tempfile
    
    print("\n--- Test DuckDB Storage Schema ---")
    storage = VectorMetadataStorage(db_path=":memory:") # Test in memory
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"Mock File Content to Hash.")
        tmp_path = tmp.name
        
    f_hash = storage.compute_file_hash(tmp_path)
    print(f"Generated Hash: {f_hash}")
    
    print(f"Is Ingested prior? {storage.is_file_ingested(f_hash)}")
    
    storage.insert_document("doc_001", "mock.txt", f_hash)
    print(f"Ingested file.")
    print(f"Is Ingested check 2? {storage.is_file_ingested(f_hash)}")
    
    storage.insert_chunk("chunk_001", "doc_001", "This is some mock text.", [0.1, 0.2, 0.3])
    storage.insert_colbert_embeddings("chunk_001", [[0.1, 0.2], [0.3, 0.4]])
    
    print("Successfully inserted Dense and ColBERT multi-vectors.")
    storage.close()
    os.remove(tmp_path)
    print("--- Success ---\n")
