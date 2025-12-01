"""
Operational Query Agent - Processes operational queries
Computes pre-aggregated insights by grouping data and calculating statistics.
Stores insights as facts in knowledge base for LLM access.
"""

from typing import List, Dict, Any, Optional, Tuple
from agent_system import agents_store
from operational_queries import process_operational_query


OPERATIONAL_QUERY_AGENT_ID = "operational_query_agent"


def process_operational_query_with_agent(query_info: Dict[str, Any], question: str) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main function for the Operational Query Agent.
    Processes operational queries by computing and returning pre-computed insights.
    """
    operational_agent = agents_store.get(OPERATIONAL_QUERY_AGENT_ID)
    if operational_agent:
        operational_agent.status = "processing"
    
    try:
        # Process operational query using the new logic
        answer, evidence_facts, routing_info = process_operational_query(query_info, question)
        
        if operational_agent:
            operational_agent.status = "active"
            
        return answer, evidence_facts, routing_info
    
    except Exception as e:
        print(f"⚠️  Error processing operational query: {e}")
        import traceback
        traceback.print_exc()
        
        if operational_agent:
            operational_agent.status = "active"
        
        return (
            f"I encountered an error processing this operational query: {str(e)}",
            [],
            {"strategy": "operational_query_agent", "reason": f"Error: {str(e)}"}
        )
