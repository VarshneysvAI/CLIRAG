import asyncio
import time
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text

# Import local CLIRAG modules
from clirag.storage.vector_meta import VectorMetadataStorage
from clirag.storage.graph import GraphStorage
from clirag.ingestion.nlp_engine import NLPExtractor
from clirag.ingestion.workers import IngestionWorker
from clirag.router.classifier import QueryClassifier
from clirag.reasoning.rlm_sandbox import RLMSandbox
from clirag.hardware.probe import HardwareProbe

app = typer.Typer(
    name="CLIRAG",
    help="100% Offline, Hardware-Agnostic Edge AI Analysis Engine.",
    add_completion=False,
)
console = Console()

# --- Global State Initialization ---
# In a real CLI, we might lazy-load these to speed up `--help` commands.
class CLIRAGState:
    def __init__(self):
        self.console = console
        self._db = None
        self._graph = None
        self._nlp = None
        self._worker = None
        
    def get_worker(self):
        if not self._worker:
            # Boot Sequence
            probe = HardwareProbe()
            backend = probe.detect_optimal_backend()
            console.print(f"[dim]Hardware Probe Detected: {backend.value}[/dim]")
            
            self._db = VectorMetadataStorage()
            self._graph = GraphStorage()
            self._nlp = NLPExtractor(load_colbert=True)
            self._worker = IngestionWorker(self._db, self._nlp, self._graph)
        return self._worker

state = CLIRAGState()

@app.command()
def ingest(filepath: str):
    """
    Ingest a document (PDF, TXT, MD) into the system.
    Demonstrates Zero-LLM Asynchronous processing with memory locks.
    """
    console.print(Panel.fit(f"[bold cyan]CLIRAG Ingestion Engine[/bold cyan]\nTarget: {filepath}", border_style="cyan"))
    
    worker = state.get_worker()

    async def run_ingestion():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            
            task_id = progress.add_task(description=f"Parsing {filepath}...", total=100.0)

            async def cli_progress_hook(msg: str, percent: float):
                progress.update(task_id, description=msg, completed=percent)

            # Await the core logic
            doc_id = await worker.ingest_file_async(filepath, progress_callback=cli_progress_hook)
            
            if doc_id == "SKIPPED_DEDUPLICATED":
                progress.update(task_id, description="[yellow]Skipped (Deduplication Check Passed)[/yellow]", completed=100.0)
            else:
                progress.update(task_id, description="[green]Ingestion Complete![/green]", completed=100.0)

    # Run the asyncio event loop
    asyncio.run(run_ingestion())
    console.print()

@app.command()
def ask(query: str):
    """
    Query the ingestion database using the Adaptive RAG router and RLM Sandbox.
    """
    console.print(Panel.fit(f"[bold magenta]CLIRAG Analysis[/bold magenta]\nQuery: '{query}'", border_style="magenta"))
    
    # Pillar 5: Adaptive RAG
    with console.status("[bold yellow]Routing Query via Classifier Agent...[/bold yellow]") as status:
        classifier = QueryClassifier()
        route = classifier.classify(query)
        time.sleep(0.4) # Simulate fast inference latency
        
    console.print(f"🚦 [bold blue]Selected Route:[/bold blue] {route}")

    # Pillar 6: The Reasoning Engine
    with console.status("[bold green]Engaging RLM Sandbox (Max Retries: 3)...[/bold green]") as status:
        sandbox = RLMSandbox(max_retries=3)
        final_answer = sandbox.execute_query(query, route)
    
    console.print("\n[bold]Final AI Response:[/bold]")
    # Pillar 1 UX: TTFT Streaming simulation
    for char in final_answer:
        console.print(char, end="", style="bold white")
        time.sleep(0.01) # Simulate token streaming
    console.print("\n")

if __name__ == "__main__":
    app()
