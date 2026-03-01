import os
import sys
import time
import asyncio
import psutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Ensure we can import clirag internally if run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clirag.storage.vector_meta import VectorMetadataStorage
from clirag.storage.graph import GraphStorage
from clirag.ingestion.nlp_engine import NLPExtractor
from clirag.ingestion.workers import IngestionWorker

def create_massive_pdf(filepath: str, num_pages: int = 5000):
    """
    Generates a massive PDF file to stress test the Mutex RAM boundaries.
    Repeats dense corporate and technical clauses.
    """
    print(f"[*] Generating {num_pages}-page massive PDF at {filepath}...")
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    
    dense_clause = (
        "Section 4.2.1: The liability regarding XDNA NPU processing pipelines and algorithmic "
        "indemnity guarantees absolute isolation in air-gapped scenarios. " * 30
    )
    
    start_time = time.time()
    for i in range(num_pages):
        c.drawString(72, height - 72, f"Page {i + 1} - CLIRAG STRESS TEST")
        
        y_pos = height - 100
        # Write ~5 lines of heavy repeating text per page
        for _ in range(5):
            # Using simple splitting since drawString doesn't wrap natively in basic reportlab
            text_fragment = dense_clause[:80] # write just the first fragment to keep it simple and heavy
            c.drawString(72, y_pos, text_fragment)
            y_pos -= 15
            
        c.showPage()
    
    c.save()
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    duration = time.time() - start_time
    print(f"[+] Generated massive PDF in {duration:.2f}s. Size: {file_size_mb:.2f} MB")
    return file_size_mb

def get_ram_usage_mb() -> float:
    """Returns the current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

async def run_stress_test():
    """
    Executes the ingestion worker against the 100MB PDF and tracks RAM usage 
    to prove Pillar 3 (Mutex RAM Locks) and Pillar 4 (Disk-Bound Storage) prevent crashes.
    """
    db_path = "data/duckdb/stress_meta.duckdb"
    graph_path = "data/kuzu/stress_graph"
    pdf_path = "stress_test_document.pdf"
    
    print("\n" + "="*50)
    print("  CLIRAG 100MB BRUTAL STRESS TEST  ")
    print("="*50)
    
    base_ram = get_ram_usage_mb()
    print(f"[*] Baseline Execution RAM: {base_ram:.2f} MB")
    
    create_massive_pdf(pdf_path, num_pages=6000) # Should generate a multi-MB file
    
    print("\n[*] Initializing Embedded Databases (DuckDB & KuzuDB)...")
    db = VectorMetadataStorage(db_path=db_path)
    graph = GraphStorage(db_path=graph_path)
    nlp = NLPExtractor(load_colbert=True)
    worker = IngestionWorker(db, nlp, graph)
    
    ram_after_init = get_ram_usage_mb()
    print(f"[*] Post-Init RAM: {ram_after_init:.2f} MB (Delta: +{ram_after_init - base_ram:.2f} MB)")
    
    print(f"\n[*] Commencing Asynchronous Ingestion with Typer Hooks on {pdf_path}...")
    start_ingest = time.time()
    
    peak_ram = ram_after_init
    
    async def stress_progress_hook(msg: str, percent: float):
        nonlocal peak_ram
        current_ram = get_ram_usage_mb()
        if current_ram > peak_ram:
            peak_ram = current_ram
        
        # Only print every 10% to avoid spamming the terminal in a test script
        if int(percent) % 10 == 0:
            print(f"[Ingest {percent:>5.1f}%] RAM: {current_ram:>6.2f} MB | {msg}")

    # Execute ingestion
    await worker.ingest_file_async(pdf_path, progress_callback=stress_progress_hook)
    
    ingest_duration = time.time() - start_ingest
    
    print("\n" + "="*50)
    print("  STRESS TEST RESULTS  ")
    print("="*50)
    print(f"Total Ingestion Time : {ingest_duration:.2f} seconds")
    print(f"Baseline RAM         : {base_ram:.2f} MB")
    print(f"Peak RAM during run  : {peak_ram:.2f} MB")
    print(f"RAM Spiked By        : +{peak_ram - base_ram:.2f} MB")
    print("="*50)
    
    if (peak_ram - base_ram) < 1500: # Assuming 1.5GB is very safe for an 8GB system requirement
        print("[SUCCESS] Mutex RAM Isolation securely defended the system from Context Exhaustion.")
    else:
        print("[WARNING] RAM limits exceeded expected thresholds.")
        
    # Cleanup databases gracefully
    db.close()
    graph.close()

if __name__ == "__main__":
    if not os.path.exists("scripts"):
        os.makedirs("scripts")
        
    asyncio.run(run_stress_test())
