import logging
import re
from typing import Literal

logger = logging.getLogger("QueryClassifier")

RouteType = Literal["BM25", "LightRAG", "ColBERT"]

class QueryClassifier:
    """
    Sub-second heuristic classifier Agent that routes the user's prompt
    to the most efficient retrieval method natively without heavy LLM latency.
    """
    
    def __init__(self):
        # Pre-compile heuristic regex patterns for sub-ms matching
        self.factual_patterns = re.compile(
            r'^(what|who|when|where|which|how many) |(date|time|number|name|location)', 
            re.IGNORECASE
        )
        self.relational_patterns = re.compile(
            r'(relate|relation|connection|link|depend|impact|affect|compare|difference|similar|ecosystem)',
            re.IGNORECASE
        )
        self.needle_patterns = re.compile(
            r'(clause|specifically|mention|exact string|quote|deep within|hidden|fine print|footnote)',
            re.IGNORECASE
        )

    def classify(self, query: str) -> RouteType:
        """
        Routes the prompt dynamically.
        - Factual/Shallow -> BM25
        - Conceptual/Relational -> LightRAG
        - Needle/Deep -> ColBERT
        """
        logger.debug(f"Classifying Query: '{query}'")
        
        # 1. Check for deep/needle indicators first (Overrides others)
        if self.needle_patterns.search(query):
            logger.info("Route Selected: ColBERT (Needle/Deep matching detected)")
            return "ColBERT"
            
        # 2. Check for relationship/graph indicators
        if self.relational_patterns.search(query):
            logger.info("Route Selected: LightRAG (Relational traversal detected)")
            return "LightRAG"
            
        # 3. Check for specific factual/lexical keyword lookups
        if self.factual_patterns.search(query):
            logger.info("Route Selected: BM25 (Factual/Shallow lookup detected)")
            return "BM25"
            
        # 4. Fallback: If ambiguity exists, default to semantic BM25 density 
        # In a full deployment, this might trigger a fast <10ms encoder match
        if len(query.split()) > 10:
             logger.info("Route Selected: ColBERT (Long complex query fallback)")
             return "ColBERT"
             
        logger.info("Route Selected: BM25 (Default Route)")
        return "BM25"

if __name__ == "__main__":
    classifier = QueryClassifier()
    print("--- Classifier Tests ---")
    print("Q: What is the launch date of the product? ->", classifier.classify("What is the launch date of the product?"))
    print("Q: How does the new policy affect remote work? ->", classifier.classify("How does the new policy affect remote work?"))
    print("Q: Find the clause specifically mentioning indemnity. ->", classifier.classify("Find the clause specifically mentioning indemnity."))
