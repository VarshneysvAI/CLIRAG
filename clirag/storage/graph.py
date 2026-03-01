import kuzu
import os
import logging
from typing import List, Dict

logger = logging.getLogger("GraphStorage")

class GraphStorage:
    """
    Manages KùzuDB embedded graph storage for LightRAG logic.
    Satisfies Pillar 4: Disk-Bound Storage (SSD constrained)
    """
    
    def __init__(self, db_path: str = "data/kuzu/clirag_graph"):
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        
        # Initialize standard KuzuDB embedded file
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        self._initialize_schema()

    def _initialize_schema(self):
        """Builds the Entity and Relation Tables if they do not exist."""
        # Check existing tables to prevent re-creation errors
        tables = self.conn.execute("CALL show_tables() RETURN name;").get_as_df()
        
        if tables.empty or "Entity" not in tables['name'].values:
            logger.info("Initializing KùzuDB Graph Schema...")
            # Nodes
            self.conn.execute("CREATE NODE TABLE Entity(name STRING, type STRING, PRIMARY KEY (name))")
            # Edges
            self.conn.execute("CREATE REL TABLE Relates(FROM Entity TO Entity, relation STRING)")

    def upsert_entities(self, entities: List[Dict[str, str]]):
        """
        Securely merges nodes and edges using Cypher queries to append to the 
        Knowledge Graph without rebuilding it.
        Satisfies: MISSING KÙZUDB IMPLEMENTATION.
        
        Example entity format:
        [{"head": "Company", "type": "ORGANIZATION", "tail": "AMD", "relation": "PARTNERSHIP"}]
        """
        for item in entities:
            head_name = item.get('head')
            head_type = item.get('type', 'UNKNOWN')
            tail_name = item.get('tail')
            relation = item.get('relation', 'RELATED_TO')

            if not head_name or not tail_name:
                continue
            
            # Use Cypher MERGE to insert or update existing nodes to prevent duplication
            try:
                # Merge Head Node
                self.conn.execute(
                    "MERGE (h:Entity {name: $name}) ON CREATE SET h.type = $type",
                    {"name": head_name, "type": head_type}
                )
                
                # Merge Tail Node (assuming generic type if not specified for tail)
                self.conn.execute(
                    "MERGE (t:Entity {name: $name}) ON CREATE SET t.type = 'UNKNOWN'",
                    {"name": tail_name}
                )
                
                # Merge Relationship
                self.conn.execute(
                    """
                    MATCH (h:Entity {name: $head_name}), (t:Entity {name: $tail_name})
                    MERGE (h)-[r:Relates {relation: $relation}]->(t)
                    """,
                    {"head_name": head_name, "tail_name": tail_name, "relation": relation}
                )
            except Exception as e:
                logger.error(f"Failed to upsert graph entity {head_name} -> {tail_name}: {e}")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    import shutil
    print("\n--- Test KùzuDB Graph Storage ---")
    test_path = "data/kuzu_test"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
        
    g = GraphStorage(db_path=test_path)
    mock_entities = [
        {"head": "CLIRAG", "type": "SOFTWARE", "tail": "Typer", "relation": "BUILT_WITH"},
        {"head": "CLIRAG", "type": "SOFTWARE", "tail": "AMD XDNA", "relation": "RUNS_ON"}
    ]
    g.upsert_entities(mock_entities)
    
    # Query test
    res = g.conn.execute("MATCH (a:Entity)-[r:Relates]->(b:Entity) RETURN a.name, r.relation, b.name;").get_as_df()
    print("Graph DB Contents:\n", res)
    g.close()
    
    shutil.rmtree(test_path)
    print("--- Success ---\n")
