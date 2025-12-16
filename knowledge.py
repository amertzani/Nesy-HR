"""
Knowledge Graph Module
======================

Provides access to the knowledge graph and related utilities.
"""

import os
import pickle
from rdflib import Graph
from typing import List, Tuple, Optional, Dict

# Global graph instance
graph: Optional[Graph] = None


def load_knowledge_graph() -> Optional[str]:
    """
    Load knowledge graph from pickle file.
    
    Returns:
        Success message if loaded, None otherwise
    """
    global graph
    
    kg_file = "knowledge_graph.pkl"
    if not os.path.exists(kg_file):
        return None
    
    try:
        with open(kg_file, 'rb') as f:
            graph = pickle.load(f)
        
        if graph and len(graph) > 0:
            return f"Knowledge graph loaded: {len(graph)} triples"
        else:
            return None
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        return None


def get_fact_source_document(subject: str, predicate: str, obj: str) -> List[Tuple[str, Optional[str]]]:
    """
    Get source document information for a fact.
    
    Args:
        subject: Fact subject
        predicate: Fact predicate
        obj: Fact object
    
    Returns:
        List of (source, timestamp) tuples
    """
    if graph is None:
        return []
    
    sources = []
    try:
        from rdflib import URIRef, Literal
        from urllib.parse import quote
        
        # Try to find the subject URI
        subject_uri = URIRef(f"urn:entity:{quote(subject.replace(' ', '_'), safe='')}")
        
        # Look for source_document triples
        for s, p, o in graph.triples((subject_uri, None, None)):
            predicate_str = str(p)
            if 'source_document' in predicate_str:
                source = str(o)
                sources.append((source, None))
        
        # If no sources found, return default
        if not sources:
            sources.append(("unknown", None))
    except Exception:
        sources.append(("unknown", None))
    
    return sources


def get_fact_details(subject: str, predicate: str, obj: str) -> Dict:
    """
    Get detailed information about a fact.
    
    Args:
        subject: Fact subject
        predicate: Fact predicate
        obj: Fact object
    
    Returns:
        Dictionary with fact details
    """
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "sources": get_fact_source_document(subject, predicate, obj)
    }


# Try to load graph on import
if graph is None:
    load_knowledge_graph()

