import logging
from typing import Dict, Any, List, Optional
import os

# Suppress library noise and progress bars globally
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("gliner").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore" # Final silence for deprecated warnings

# Global Models Directory
_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
logger = logging.getLogger("NLPEngine")

class NLPExtractor:
    """
    Sub-second Entity Extraction & Vector Generation.
    Implements Lazy Loading (Pillar 7) to save RAM and Startup Time.
    """
    def __init__(self, load_colbert: bool = False):
        self.dense_model = None
        self.entity_model = None
        self.colbert_model = None
        self.load_colbert = load_colbert
        logger.debug("NLPExtractor initialized (Lazy Mode Active).")

    def _resolve_model_path(self, repo_id: str, local_name: str) -> str:
        """
        AI Warriors Robust Resolver.
        Checks for flat folders first, then HF cache folders, then falls back to repo_id.
        """
        # 1. Check flat folder
        flat_path = os.path.join(_MODELS_DIR, local_name)
        if os.path.exists(flat_path):
            return flat_path
            
        # 2. Check HF cache style (models--repo--id)
        # Convert repo/id to models--repo--id
        cache_folder_name = f"models--{repo_id.replace('/', '--')}"
        cache_path = os.path.join(_MODELS_DIR, cache_folder_name)
        
        if os.path.exists(cache_path):
            snapshots_path = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_path):
                snapshots = os.listdir(snapshots_path)
                if snapshots:
                    # Sort snapshots to get the latest (usually only one)
                    latest_snapshot = sorted(snapshots)[-1]
                    return os.path.join(snapshots_path, latest_snapshot)
        
        # 3. Fallback to repo_id (lib will try to find it in default cache)
        logger.warning(f"AI Warriors: Local model {local_name} not found in {_MODELS_DIR}. Falling back to HF Repo: {repo_id}")
        return repo_id

    def _ensure_dense_model(self):
        """Lazy Load: sentence-transformers (all-MiniLM-L6-v2)"""
        if self.dense_model is None:
            from sentence_transformers import SentenceTransformer
            model_path = self._resolve_model_path(
                "sentence-transformers/all-MiniLM-L6-v2", 
                "all-MiniLM-L6-v2"
            )
            logger.debug(f"Lazy Loading Dense Embedding Model from {model_path}...")
            try:
                # Force local_files_only if it's a path
                is_path = os.path.sep in model_path or os.path.exists(model_path)
                self.dense_model = SentenceTransformer(
                    model_path, 
                    device="cpu", # Force CPU for predictable slowness
                    local_files_only=is_path
                )
            except Exception as e:
                logger.error(f"AI Warriors: Dense model load failed: {e}. Run 'clirag bootstrap' while online.")
                self.dense_model = None
        return self.dense_model

    def _ensure_entity_model(self):
        """Lazy Load: GLiNER"""
        if self.entity_model is None:
            from gliner import GLiNER
            model_path = self._resolve_model_path(
                "urchade/gliner_small-v2.1", 
                "gliner_small-v2.1"
            )
            logger.debug(f"Lazy Loading GLiNER Entity Extractor from {model_path}...")
            try:
                is_path = os.path.sep in model_path or os.path.exists(model_path)
                self.entity_model = GLiNER.from_pretrained(
                    model_path, 
                    local_files_only=is_path
                )
            except Exception as e:
                logger.error(f"AI Warriors: GLiNER load failed: {e}. Run 'clirag bootstrap' to fix models.")
                self.entity_model = "FAILED"
        return self.entity_model if self.entity_model != "FAILED" else None

    def extract_and_vectorize(self, text: str, extract_entities: bool = True, skip_nlp: bool = False) -> Dict[str, Any]:
        """
        Processes a text chunk. Lazily loads models on first call.
        If skip_nlp is True, returns empty entities (saves significant time for tables).
        """
        # 1. Dense Embedding (Required)
        model = self._ensure_dense_model()
        if not model:
            return {"dense_embedding": [0.0]*384, "colbert_embeddings": [], "graph_entities": []}
        
        # Disable progress bar for individual encodes
        dense_vec = model.encode(text, show_progress_bar=False).tolist()
        
        # 2. Entity Extraction (Lazy & Conditional)
        entities = []
        if extract_entities and not skip_nlp:
            entity_model = self._ensure_entity_model()
            if entity_model:
                labels = ["Person", "Organization", "Location", "Date", "Event", "Concept", "Technology", "Company"]
                try:
                    raw_entities = entity_model.predict_entities(text, labels=labels)
                    # Relational Heuristic
                    for i in range(len(raw_entities) - 1):
                        entities.append({
                            "head": raw_entities[i]["text"],
                            "type": raw_entities[i]["label"],
                            "tail": raw_entities[i+1]["text"],
                            "relation": "RELATED_TO_IN_CONTEXT"
                        })
                except Exception as e:
                    logger.error(f"GLiNER prediction failed: {e}")

        return {
            "dense_embedding": dense_vec,
            "colbert_embeddings": [], # ColBERT bypassed in MVP
            "graph_entities": entities
        }

if __name__ == "__main__":
    extractor = NLPExtractor()
    print("\n[DEBUG] Testing Lazy Ingestion (First call triggers load)...")
    data = extractor.extract_and_vectorize("CLIRAG Modernization Demo")
    print(f"Entities: {data['graph_entities']}")
    print("--- Success ---\n")
