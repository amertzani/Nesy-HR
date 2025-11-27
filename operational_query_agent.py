"""
Operational Query Agent - Processes operational-level analytical queries (O1, O2, O3)
Uses statistics from Statistics Agent and facts from Knowledge Graph
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from agent_system import document_agents, agents_store, STATISTICS_AGENT_ID
from knowledge import graph as kb_graph
from strategic_query_agent import (
    reconstruct_dataframe_from_facts,
    normalize_column_name
)
from strategic_queries import (
    process_o1_1, process_o1_2, process_o2_1, process_o3_1, process_o4_1
)


OPERATIONAL_QUERY_AGENT_ID = "operational_query_agent"


def process_operational_query_with_agent(query_info: Dict[str, Any], question: str) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main function for the Operational Query Agent.
    Processes operational queries (O1, O2, O3) using DataFrame reconstruction from knowledge graph.
    """
    strategic_type = query_info.get("strategic_type")  # Will be "O1", "O2", or "O3"
    subtype = query_info.get("subtype")  # "O1.1", "O1.2", "O2.1", "O3.1"
    variables = query_info.get("variables", [])
    
    # Reconstruct DataFrame from knowledge graph
    df = reconstruct_dataframe_from_facts()
    
    if df is None or len(df) == 0:
        # Check if we have statistics but no DataFrame
        from agent_system import agents_store
        stats_agent = agents_store.get(STATISTICS_AGENT_ID)
        all_stats = []
        for agent_id, agent in document_agents.items():
            if hasattr(agent, 'metadata') and agent.metadata:
                stats = agent.metadata.get("statistics")
                if stats:
                    all_stats.append(stats)
        
        if all_stats:
            stats = all_stats[0]
            return (
                f"I found statistics for {stats.get('total_rows', 0)} rows, but need to reconstruct the data for operational analysis. "
                f"Please ensure the knowledge graph contains the required employee data.",
                [],
                {"strategy": "operational_query_agent", "reason": "DataFrame reconstruction needed"}
            )
        
        return (
            "I couldn't find the required data in the knowledge graph. Please ensure a CSV file has been uploaded and processed.",
            [],
            {"strategy": "operational_query_agent", "reason": "No data available"}
        )
    
    # Normalize column names - handle variations
    actual_columns = {}
    column_mapping = {
        "PerformanceScore": ["PerformanceScore", "PerfScoreID"],
        "EngagementSurvey": ["EngagementSurvey"],
        "EmploymentStatus": ["EmploymentStatus"],
        "Department": ["Department"],
        "Position": ["Position"],
        "RecruitmentSource": ["RecruitmentSource"],
        "Absences": ["Absences"],
        "ManagerName": ["ManagerName"],
        "DateofHire": ["DateofHire"],
        "SpecialProjectsCount": ["SpecialProjectsCount"]
    }
    
    # For O4 queries, use metric and group_by from query_info
    if strategic_type == "O4":
        metric = query_info.get("metric")
        group_by = query_info.get("group_by")
        if metric:
            actual_col = normalize_column_name(df, metric)
            if actual_col:
                actual_columns["metric"] = actual_col
            else:
                available_cols = ', '.join(df.columns[:15].tolist())
                return (
                    f"I couldn't find the '{metric}' column in the dataset. "
                    f"Available columns: {available_cols}...",
                    [],
                    {"strategy": "operational_query_agent", "reason": f"Column '{metric}' not found"}
                )
        if group_by:
            actual_col = normalize_column_name(df, group_by)
            if actual_col:
                actual_columns["group_by"] = actual_col
            else:
                available_cols = ', '.join(df.columns[:15].tolist())
                return (
                    f"I couldn't find the '{group_by}' column in the dataset. "
                    f"Available columns: {available_cols}...",
                    [],
                    {"strategy": "operational_query_agent", "reason": f"Column '{group_by}' not found"}
                )
    else:
        # For O1, O2, O3 queries, use variables list
        for var in variables:
            # Try the primary name first
            actual_col = normalize_column_name(df, var)
            
            # If not found and we have a mapping, try alternatives
            if not actual_col and var in column_mapping:
                for alt_name in column_mapping[var]:
                    actual_col = normalize_column_name(df, alt_name)
                    if actual_col:
                        break
            
            if actual_col:
                actual_columns[var] = actual_col
            else:
                # Provide helpful error with available columns
                available_cols = ', '.join(df.columns[:15].tolist())
                return (
                    f"I couldn't find the '{var}' column (or its variations) in the dataset. "
                    f"Available columns: {available_cols}... "
                    f"Please ensure the dataset contains columns matching: {', '.join(column_mapping.get(var, [var]))}",
                    [],
                    {"strategy": "operational_query_agent", "reason": f"Column '{var}' not found"}
                )
    
    answer_parts = []
    evidence_facts = []
    
    try:
        # Route to appropriate operational query processor
        if strategic_type == "O1":
            if subtype == "O1.1":
                answer_parts, evidence_facts = process_o1_1(df, actual_columns)
            elif subtype == "O1.2":
                answer_parts, evidence_facts = process_o1_2(df, actual_columns)
        
        elif strategic_type == "O2":
            if subtype == "O2.1":
                answer_parts, evidence_facts = process_o2_1(df, actual_columns)
        
        elif strategic_type == "O3":
            if subtype == "O3.1":
                answer_parts, evidence_facts = process_o3_1(df, actual_columns)
        
        elif strategic_type == "O4":
            if subtype == "O4.1":
                answer_parts, evidence_facts = process_o4_1(df, actual_columns, query_info)
        
        if answer_parts:
            answer = "\n\n".join(answer_parts)
            
            # Store insights in knowledge graph so LLM can access them later
            try:
                from knowledge import add_to_graph
                from datetime import datetime
                insight_text = f"Operational Analysis ({strategic_type}.{subtype}): {answer}"
                add_to_graph(
                    insight_text,
                    source_document="operational_insights",
                    uploaded_at=datetime.now().isoformat(),
                    agent_id=OPERATIONAL_QUERY_AGENT_ID
                )
                print(f"✅ Stored operational insight in knowledge graph for LLM access")
            except Exception as e:
                print(f"⚠️  Failed to store operational insight: {e}")
            
            return answer, evidence_facts, {
                "strategy": "operational_query_agent",
                "reason": f"Processed {strategic_type} query using operational analysis",
                "query_type": strategic_type,
                "subtype": subtype,
                "data_source": "knowledge_graph"
            }
        else:
            return (
                "I processed the query but couldn't generate a complete answer. Please check if the required columns exist in the dataset.",
                evidence_facts,
                {"strategy": "operational_query_agent", "reason": "Incomplete analysis"}
            )
    
    except Exception as e:
        print(f"⚠️  Error processing operational query: {e}")
        import traceback
        traceback.print_exc()
        return (
            f"I encountered an error processing this operational query: {str(e)}",
            [],
            {"strategy": "operational_query_agent", "reason": f"Error: {str(e)}"}
        )

