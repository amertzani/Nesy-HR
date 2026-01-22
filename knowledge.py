"""
Knowledge Graph Module
======================

Provides access to the knowledge graph and related utilities.
"""

import os
import pickle
from rdflib import Graph
from typing import List, Tuple, Optional, Dict, Any

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


def get_fact_provenance(subject: str, predicate: str, obj: str) -> Dict[str, Any]:
    """
    Get complete provenance metadata for a fact (source, timestamp, pathway).
    
    Args:
        subject: Fact subject
        predicate: Fact predicate
        obj: Fact object
    
    Returns:
        Dictionary with source_document, timestamp, and agent_id/pathway
    """
    if graph is None:
        return {
            "source_document": "unknown",
            "timestamp": None,
            "agent_id": None,
            "source_type": "unknown"
        }
    
    try:
        from rdflib import URIRef, Literal
        from urllib.parse import quote, unquote
        
        source_document = "unknown"
        timestamp = None
        agent_id = None
        
        # Try multiple URI patterns
        uri_patterns = [
            f"urn:entity:{quote(subject.replace(' ', '_'), safe='')}",
            f"urn:fact:{quote(f'{subject}|{predicate}|{obj}'.replace(' ', '_'), safe='')}",
        ]
        
        # Also try to find by matching subject in existing triples
        for s, p, o in graph:
            # Skip metadata triples
            predicate_str = str(p)
            if any(x in predicate_str for x in ['source_document', 'uploaded_at', 'agent_id', 'metadata']):
                continue
            
            # Check if this triple matches our fact
            s_str = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
            p_str = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
            o_str = str(o)
            
            # If this triple matches our fact, look for its metadata
            if (subject.lower() in s_str.lower() or s_str.lower() in subject.lower()) and \
               (predicate.lower() in p_str.lower() or p_str.lower() in predicate.lower()) and \
               (obj.lower() in o_str.lower() or o_str.lower() in obj.lower()):
                # Found matching triple, now look for its metadata
                for ms, mp, mo in graph.triples((s, None, None)):
                    mp_str = str(mp)
                    if 'source_document' in mp_str:
                        source_document = str(mo)
                    elif 'uploaded_at' in mp_str:
                        timestamp = str(mo)
                    elif 'agent_id' in mp_str:
                        agent_id = str(mo)
                
                # If we found metadata, break
                if source_document != "unknown" or timestamp or agent_id:
                    break
        
        # If still not found, try direct URI lookup
        if source_document == "unknown" and not timestamp and not agent_id:
            for uri_pattern in uri_patterns:
                subject_uri = URIRef(uri_pattern)
                for s, p, o in graph.triples((subject_uri, None, None)):
                    predicate_str = str(p)
                    
                    if 'source_document' in predicate_str:
                        source_document = str(o)
                    elif 'uploaded_at' in predicate_str:
                        timestamp = str(o)
                    elif 'agent_id' in predicate_str:
                        agent_id = str(o)
        
        # Classify source type based on source document
        source_type = "unknown"
        if source_document and source_document != "unknown":
            source_lower = source_document.lower()
            if 'operational_insights' in source_lower:
                source_type = "operational"
            elif any(ext in source_lower for ext in ['.csv', '.pdf', '.docx', '.txt']):
                source_type = "document_agent"
            elif 'statistics' in source_lower:
                source_type = "statistics"
            else:
                source_type = "document_agent"  # Default assumption
        
        return {
            "source_document": source_document,
            "timestamp": timestamp,
            "agent_id": agent_id,
            "source_type": source_type
        }
    except Exception as e:
        return {
            "source_document": "unknown",
            "timestamp": None,
            "agent_id": None,
            "source_type": "unknown"
        }


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

