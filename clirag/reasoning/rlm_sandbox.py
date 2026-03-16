import time
import json
import logging
import os
from typing import Dict, Any, Tuple, Optional

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

logger = logging.getLogger("RLMSandbox")

# Resolve the project root once at module level (two levels up from this file)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Phi-3 chat template tokens
_PHI3_SYS = "\x3c|system|\x3e"
_PHI3_USR = "\x3c|user|\x3e"
_PHI3_AST = "\x3c|assistant|\x3e"
_PHI3_END = "\x3c|end|\x3e"


class RLMSandbox:
    """
    Constrained Recursive Language Model (RLM) Sandbox.
    Prevents infinite LLM hallucination loops during tool execution.
    Satisfies Pillar 6.
    """
    
    def __init__(self, db_storage: Any = None, graph_storage: Any = None,
                 max_retries: int = 3, llama_kwargs: Optional[Dict[str, Any]] = None):
        self.MAX_RETRIES = max_retries
        self.db_storage = db_storage
        self.graph_storage = graph_storage
        
        # Real DB tools
        self.available_tools = {
            "search_graph": self.search_graph,
            "search_exact_text": self.search_exact_text,
            "read_document_chunk": self.read_document_chunk
        }

        # Route-specific tool restrictions: each route only exposes relevant tools
        self.route_tools = {
            "BM25": ["search_exact_text", "read_document_chunk"],
            "LightRAG": ["search_graph", "search_exact_text", "read_document_chunk"],
            "ColBERT": ["search_exact_text", "read_document_chunk"],
        }
        
        logger.info("Initializing llama.cpp with Hardware optimizations...")
        
        # Use __file__-based resolution so model is found regardless of CWD
        local_model_path = os.path.join(_PROJECT_ROOT, "models", "Phi-3-mini-4k-instruct-q4.gguf")
        
        if os.path.exists(local_model_path):
            logger.info(f"Using local model: {local_model_path}")
            model_path = local_model_path
        else:
            logger.info("Local model not found. Attempting HuggingFace Hub fallback...")
            cache_dir = os.path.join(_PROJECT_ROOT, "models")
            try:
                model_path = hf_hub_download(
                    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf", 
                    filename="Phi-3-mini-4k-instruct-q4.gguf",
                    cache_dir=cache_dir
                )
            except Exception:
                try:
                    model_path = hf_hub_download(
                        repo_id="microsoft/Phi-3-mini-4k-instruct-gguf", 
                        filename="Phi-3-mini-4k-instruct-q4.gguf",
                        local_files_only=True,
                        cache_dir=cache_dir
                    )
                except Exception as e:
                    logger.error(f"Failed to locate model fallback: {e}")
                    model_path = None

        if model_path:
            # Use pre-computed kwargs if provided (avoids redundant HardwareProbe call)
            kwargs = llama_kwargs if llama_kwargs else {}
            self.llm = Llama(model_path=model_path, verbose=False, **kwargs)
        else:
            self.llm = None

    def execute_query(self, query: str, route: str, doc_id: Optional[str] = None) -> str:
        """
        Main REPL Loop mimicking the LLM interacting with the retrieval tools securely.
        The route parameter restricts tools, and doc_id restricts retrieval scope.
        """
        # doc_id is now passed explicitly to tool methods for thread-safety.
        if not self.llm:
            logger.error("LLM model (Phi-3) is not loaded. Cannot execute reasoning.")
            return "Error: The AI reasoning model (Phi-3) is missing or could not be loaded. Please ensure the .gguf model file is in the 'models' folder."

        retries = 0
        current_context = ""
        current_query = query
        
        # Get the tools allowed for this route
        allowed_tool_names = self.route_tools.get(route, list(self.available_tools.keys()))
        allowed_tools = {name: self.available_tools[name] for name in allowed_tool_names if name in self.available_tools}
        
        logger.info(f"Entering RLM Sandbox. Route: {route} | Allowed Tools: {list(allowed_tools.keys())} | Max Retries: {self.MAX_RETRIES}")
        
        while retries < self.MAX_RETRIES:
            logger.info(f"[Retry {retries}/{self.MAX_RETRIES}] LLM generation with context size: {len(current_context)}")
            
            action, term = self._prompt_llm(current_query, current_context, route, retries, list(allowed_tools.keys()))
            
            if action == "FINAL_ANSWER":
                logger.info(f"[SUCCESS] LLM derived final answer on attempt {retries}.")
                if not term or term.strip() == "":
                    return "I found relevant sections in the document but could not synthesize a specific answer. Please try rephrasing."
                return term
                
            elif action in allowed_tools:
                logger.info(f"[TOOL] LLM requested tool: {action}('{term}')")
                # Handle different tool signatures
                if action in ["search_graph", "search_exact_text"]:
                    tool_result = allowed_tools[action](term, doc_id=doc_id)
                else:
                    tool_result = allowed_tools[action](term)
                
                if tool_result:
                    current_context += f"\n[Doc Info]: {tool_result}"
                else:
                    logger.warning(f"[FAIL] Tool {action} returned no results for '{term}'.")
                    current_query = f"{query} (Note: previous search for '{term}' failed)"
            else:
                logger.error(f"[FORBIDDEN] LLM attempted forbidden function: {action}")
                current_context += f"\n[System Error]: Function '{action}' is forbidden for route '{route}'."
            
            retries += 1
            
        logger.warning(f"MAX RETRY LIMIT ({self.MAX_RETRIES}) reached. Forcing fallback output.")
        return "I could not find a definitive answer in the documents based on the given constraints."

    def _prompt_llm(self, query: str, context: str, route: str, attempt: int, tool_names: list) -> Tuple[str, str]:
        """
        Uses llama.cpp to prompt the model strictly for JSON tool actions.
        """
        tools_str = "', '".join(tool_names)
        system_prompt = (
            "You are an analytical AI reasoning agent for Team AI Warriors. "
            "Your task is to retrieve information using tools or provide a FINAL_ANSWER. "
            f"Available Tools: '{tools_str}'. "
            "Instruction: Output exactly ONE valid JSON object. "
            'Correct Format: {"action": "FINAL_ANSWER", "term": "your answer text"}\n'
            "Decision:"
        )
        
        user_prompt = f"Context available so far: {context}\n\nQuery: {query}\n\nAttempt: {attempt}. Output your JSON action decision now."
        
        # Phi-3 chat template
        prompt = f"{_PHI3_SYS}\n{system_prompt}{_PHI3_END}\n{_PHI3_USR}\n{user_prompt}{_PHI3_END}\n{_PHI3_AST}\n"
        
        response = self.llm(prompt, max_tokens=256, temperature=0.0, stop=[_PHI3_END])
        text = response["choices"][0]["text"].strip()
        logger.info(f"LLM Raw Output: '{text}'")
        
        try:
            # EMERGENCY PARSING: Handle potential missing brackets or extra text
            text_cleaned = text.strip()
            if not text_cleaned.startswith("{"):
                # Try to find { in the text
                start = text_cleaned.find("{")
                end = text_cleaned.rfind("}")
                if start != -1 and end != -1:
                    text_cleaned = text_cleaned[start:end+1]
                else:
                    # Very primitive attempt
                    if "FINAL_ANSWER" in text_cleaned:
                         return "FINAL_ANSWER", text_cleaned.split("FINAL_ANSWER")[-1].strip(": \"'{}")
            
            data = json.loads(text_cleaned)
            action = data.get("action", "FINAL_ANSWER")
            term = data.get("term", "No action specified.")
            return action, term
        except Exception as e:
            logger.error(f"LLM produced invalid JSON: {text} | Error: {e}")
            if attempt < self.MAX_RETRIES - 1:
                return "search_exact_text", query  # Fallback action
            else:
                return "FINAL_ANSWER", "Failed to parse final answer from the LLM based on constraints."

    def search_graph(self, term: str, doc_id: Optional[str] = None) -> str:
        if not self.graph_storage:
             return "Error: Graph storage not initialized."
             
        query = """
        MATCH (a:Entity)-[r:Relates]->(b:Entity) 
        WHERE (a.name CONTAINS $term OR b.name CONTAINS $term) 
        """
        if doc_id:
            query += " AND a.doc_id = $doc_id "
        
        query += " RETURN a.name, r.relation, b.name LIMIT 5;"
        
        try:
            params = {"term": term}
            if doc_id:
                params["doc_id"] = doc_id
                
            res = self.graph_storage.conn.execute(query, params).get_as_df()
            if res.empty:
                return ""
            
            results = []
            for _, row in res.iterrows():
                 results.append(f"Node({row['a.name']}) -[{row['r.relation']}]-> Node({row['b.name']})")
                 
            return " | ".join(results)
        except Exception as e:
            logger.error(f"Graph Search Error for '{term}': {e}")
            return ""

    def search_exact_text(self, term: str, doc_id: Optional[str] = None) -> str:
        if not self.db_storage:
             return "Error: Vector storage not initialized."
             
        try:
            sql = "SELECT chunk_id, text_content, fts_main_Dense_Vectors.match_bm25(chunk_id, ?) as score FROM Dense_Vectors WHERE score IS NOT NULL "
            params = [term]
            
            if doc_id:
                sql += " AND doc_id = ? "
                params.append(doc_id)
                # Increase recall for document-specific searches
                limit = 3
            else:
                limit = 1
                
            sql += f" ORDER BY score DESC LIMIT {limit}"
            
            rows = self.db_storage.conn.execute(sql, params).fetchall()
            
            # Fallback for document-specific search if FTS returns nothing
            if not rows and doc_id:
                logger.info(f"FTS returned nothing for '{term}' in {doc_id}. Falling back to LIKE search.")
                rows = self.db_storage.conn.execute(
                    "SELECT chunk_id, text_content FROM Dense_Vectors WHERE doc_id = ? AND text_content ILIKE ? LIMIT 2",
                    [doc_id, f"%{term}%"]
                ).fetchall()

            if rows:
                results = []
                for row in rows:
                    chunk_id, text = row[:2]
                    snippet = text[:1000] + "..." if len(text) > 1000 else text
                    results.append(f"[{chunk_id}]: {snippet}")
                return "\n".join(results)
            return ""
        except Exception as e:
             logger.error(f"Text Search Error for '{term}': {e}")
             return ""

    def read_document_chunk(self, chunk_id: str) -> str:
        if not self.db_storage:
             return "Error: Vector storage not initialized."
             
        try:
            res = self.db_storage.conn.execute(
                "SELECT text_content FROM Dense_Vectors WHERE chunk_id = ?",
                [chunk_id]
            ).fetchone()
            
            if res:
                return f"Full text for {chunk_id}: {res[0]}"
            return f"Error: {chunk_id} not found."
        except Exception as e:
            logger.error(f"Chunk Read Error for '{chunk_id}': {e}")
            return ""

if __name__ == "__main__":
    sandbox = RLMSandbox()
    print("\n--- RLM Sandbox Test ---")
    ans = sandbox.execute_query("What is the indemnity clause?", "ColBERT")
    print(f"\nFinal Terminal Output:\n{ans}")
    print("--- Success ---\n")
