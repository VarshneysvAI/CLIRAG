import os
import logging

logger = logging.getLogger("CLIRAGState")

class CLIRAGState:
    """
    Lazy-loading Global State.
    Heavy libraries are imported ONLY when needed inside the getters.
    This module is used both by the CLI (main.py) and the Server (engine.py).
    """
    def __init__(self):
        self._db = None
        self._graph = None
        self._nlp = None
        self._worker = None
        self._sandbox = None
        self._hw_kwargs = None
        self._classifier = None

    def _ensure_hw_kwargs(self):
        if self._hw_kwargs is None:
            from clirag.hardware.probe import HardwareProbe
            probe = HardwareProbe()
            self._hw_kwargs = probe.get_llama_cpp_kwargs()
        return self._hw_kwargs

    def _ensure_db(self, read_only: bool = False):
        if self._db is None:
            from clirag.storage.vector_meta import VectorMetadataStorage
            self._db = VectorMetadataStorage(read_only=read_only)
        return self._db

    def _ensure_graph(self):
        if self._graph is None:
            from clirag.storage.graph import GraphStorage
            self._graph = GraphStorage()
        return self._graph

    def get_worker(self):
        if not self._worker:
            from clirag.ingestion.workers import IngestionWorker
            from clirag.ingestion.nlp_engine import NLPExtractor
            db = self._ensure_db()
            graph = self._ensure_graph()
            self._ensure_hw_kwargs()
            self._nlp = NLPExtractor(load_colbert=False)
            self._worker = IngestionWorker(db, self._nlp, graph)
        return self._worker

    def get_sandbox(self):
        if not self._sandbox:
            from clirag.reasoning.rlm_sandbox import RLMSandbox
            db = self._ensure_db()
            graph = self._ensure_graph()
            hw_kwargs = self._ensure_hw_kwargs()
            self._sandbox = RLMSandbox(db_storage=db, graph_storage=graph, llama_kwargs=hw_kwargs)
        return self._sandbox

    def get_classifier(self):
        if not self._classifier:
            from clirag.router.classifier import QueryClassifier
            self._classifier = QueryClassifier()
        return self._classifier

# Global instance
state = CLIRAGState()
