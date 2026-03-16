import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

# Import the shared state
from .state import state

# --- FastAPI App ---
engine_app = FastAPI(
    title="CLIRAG Edge Engine",
    description="Persistent RAM residency for offline AI models."
)

class QueryRequest(BaseModel):
    query: str
    route: str = "AUTO"
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None

@engine_app.get("/health")
def health():
    return {"status": "ok", "state": "ready" if state._sandbox else "cold"}

@engine_app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    """
    Main endpoint for < 2s queries.
    Utilizes the RAM-resident model for instant inference.
    """
    classifier = state.get_classifier()
    route = req.route if req.route != "AUTO" else classifier.classify(req.query)
    
    # Resolve doc_name to doc_id server-side if provided
    doc_id = req.doc_id
    if req.doc_name and not doc_id:
        # Use existing server-side DB connection
        db = state._ensure_db()
        doc_id = db.get_doc_id_by_filename(req.doc_name)

    # Run heavy LLM work in an executor to avoid blocking event loop
    sandbox = state.get_sandbox()
    loop = asyncio.get_event_loop()
    try:
        answer = await loop.run_in_executor(None, sandbox.execute_query, req.query, route, doc_id)
        return {"answer": answer, "route": route}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@engine_app.post("/ingest")
async def ingest_endpoint(filepath: str):
    """
    Server-side ingestion.
    """
    worker = state.get_worker()
    doc_id = await worker.ingest_file_async(filepath)
    return {"status": "ok", "doc_id": doc_id}
