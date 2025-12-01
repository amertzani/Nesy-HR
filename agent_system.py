"""
Multi-Agent System for HR Decision Support
==========================================

This module implements a focused multi-agent architecture for HR professionals
to make informed, evidence-based decisions.

Architecture:
- Statistics Agent: Performs statistical analysis of documents (especially CSV files)
- Visualization Agent: Creates user-friendly graphs/charts from statistics
- Knowledge Graph Agent: Extracts facts/knowledge from documents for traceability
- LLM Agent: Provides traceable answers using KG and statistics

Communication Flow:
- Statistics Agent ↔ Visualization Agent
- Statistics Agent ↔ KG Agent
- LLM Agent ↔ KG Agent
- LLM Agent ↔ Statistics Agent

Author: Research Brain Team
Last Updated: 2025-01-20
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import rdflib
import copy

# Import knowledge graph functions
from knowledge import (
    graph as kb_graph,
    add_to_graph as kb_add_to_graph,
    save_knowledge_graph as kb_save_knowledge_graph,
    load_knowledge_graph as kb_load_knowledge_graph,
)

# ============================================================================
# AGENT DATA STRUCTURES
# ============================================================================

@dataclass
class Agent:
    """Base agent structure"""
    id: str
    name: str
    type: str  # "statistics", "visualization", "kg", "llm"
    status: str  # "active", "idle", "processing"
    created_at: str
    metadata: Dict[str, Any]

@dataclass
class DocumentAgent(Agent):
    """Agent assigned to a specific document"""
    document_name: str
    document_id: str
    document_type: str  # "csv", "pdf", "docx", "txt"
    file_path: Optional[str] = None  # Store file path for CSV access
    facts_extracted: int = 0  # Track facts extracted for this document
    employee_names: List[str] = field(default_factory=list)  # Employees this agent processed (for orchestrator routing)
    data_range: Optional[Dict[str, Any]] = None  # Row ranges, column info, etc. (for orchestrator routing)
    columns_processed: List[str] = field(default_factory=list)  # Columns this agent processes (for orchestrator routing)

# ============================================================================
# GLOBAL STATE
# ============================================================================

AGENTS_FILE = "agents_store.json"

# Core agent IDs
ORCHESTRATOR_AGENT_ID = "orchestrator_agent"
STATISTICS_AGENT_ID = "statistics_agent"
VISUALIZATION_AGENT_ID = "visualization_agent"
KG_AGENT_ID = "kg_agent"
LLM_AGENT_ID = "llm_agent"
STRATEGIC_QUERY_AGENT_ID = "strategic_query_agent"
OPERATIONAL_QUERY_AGENT_ID = "operational_query_agent"

# Document-specific agents (created per document)
document_agents: Dict[str, DocumentAgent] = {}

# Agent storage (core agents)
agents_store: Dict[str, Agent] = {}

# ============================================================================
# STORAGE FUNCTIONS
# ============================================================================

def load_agents() -> Dict[str, Dict]:
    """Load agents from storage"""
    try:
        if os.path.exists(AGENTS_FILE):
            with open(AGENTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('agents', {})
        return {}
    except Exception as e:
        print(f"Error loading agents: {e}")
        return {}

def save_agents():
    """Save agents to storage - only core agents are persisted, document agents are ephemeral"""
    try:
        # Only save core agents
        all_agents = {}
        
        for agent_id, agent in agents_store.items():
            all_agents[agent_id] = asdict(agent)
        
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_agents': len(all_agents),
            'agents': all_agents,
            'note': 'Document agents are ephemeral and not persisted'
        }
        
        with open(AGENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving agents: {e}")
        return False

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_agents(clear_document_agents: bool = True):
    """Initialize core agents on startup
    
    Args:
        clear_document_agents: If True, clear document agents (default). Set to False to preserve existing agents.
    """
    global agents_store, document_agents
    
    # Clear document agents only if requested
    if clear_document_agents:
        doc_agent_count = len(document_agents)
        document_agents.clear()
        if doc_agent_count > 0:
            print(f"🗑️  Cleared {doc_agent_count} document agents during initialization")
    
    # Orchestrator Agent
    agents_store[ORCHESTRATOR_AGENT_ID] = Agent(
        id=ORCHESTRATOR_AGENT_ID,
        name="Orchestrator Agent",
        type="orchestrator",
        status="active",
        created_at=datetime.now().isoformat(),
        metadata={
            "description": "Coordinates queries and delegates tasks to appropriate agents",
            "role": "query_coordination",
            "capabilities": ["query_routing", "agent_delegation", "result_aggregation", "task_coordination"]
        }
    )
    
    # Statistics Agent
    agents_store[STATISTICS_AGENT_ID] = Agent(
        id=STATISTICS_AGENT_ID,
        name="Statistics Agent",
        type="statistics",
        status="active",
        created_at=datetime.now().isoformat(),
        metadata={
            "description": "Performs statistical analysis of documents (especially CSV files)",
            "role": "statistical_analysis",
            "capabilities": ["descriptive_stats", "correlations", "data_quality", "trends"]
        }
    )
    
    # Visualization Agent
    agents_store[VISUALIZATION_AGENT_ID] = Agent(
        id=VISUALIZATION_AGENT_ID,
        name="Visualization Agent",
        type="visualization",
        status="active",
        created_at=datetime.now().isoformat(),
        metadata={
            "description": "Creates user-friendly graphs/charts from statistics",
            "role": "data_visualization",
            "capabilities": ["bar_charts", "line_charts", "scatter_plots", "heatmaps"]
        }
    )
    
    # Knowledge Graph Agent
    agents_store[KG_AGENT_ID] = Agent(
        id=KG_AGENT_ID,
        name="Knowledge Graph Agent",
        type="kg",
        status="active",
        created_at=datetime.now().isoformat(),
        metadata={
            "description": "Extracts facts/knowledge from documents for traceability",
            "role": "knowledge_extraction",
            "capabilities": ["fact_extraction", "entity_recognition", "relation_extraction"]
        }
    )
    
    # LLM Agent
    stored_agents = load_agents()
    llm_agent_data = stored_agents.get(LLM_AGENT_ID)
    
    if llm_agent_data:
        agents_store[LLM_AGENT_ID] = Agent(
            id=LLM_AGENT_ID,
            name="HR Assistant",
            type="llm",
            status="active",
            created_at=llm_agent_data.get('created_at', datetime.now().isoformat()),
            metadata=llm_agent_data.get('metadata', {
                "description": "Provides traceable answers using KG and statistics",
                "role": "research_assistant",
                "capabilities": ["qa", "insights", "traceability"]
            })
        )
        print(f"✅ Restored HR Assistant agent")
    else:
        agents_store[LLM_AGENT_ID] = Agent(
            id=LLM_AGENT_ID,
            name="HR Assistant",
            type="llm",
            status="active",
            created_at=datetime.now().isoformat(),
            metadata={
                "description": "Generates intelligent responses using LLM with knowledge graph context",
                "role": "intelligent_responses",
                "capabilities": ["rag", "context_aware_responses", "insights", "explanations"]
            }
        )
    
    # Strategic Query Agent (handles S1, S2 queries)
    agents_store[STRATEGIC_QUERY_AGENT_ID] = Agent(
        id=STRATEGIC_QUERY_AGENT_ID,
        name="Strategic Query Agent",
        type="strategic_query",
        status="active",
        created_at=datetime.now().isoformat(),
        metadata={
            "description": "Processes strategic-level multi-variable queries (S1, S2) using statistics and knowledge graph",
            "role": "strategic_analysis",
            "capabilities": ["multi_variable_analysis", "risk_detection", "performance_monitoring", "recruitment_analysis"]
        }
    )
    
    # Operational Query Agent (handles O1, O2, O3 queries)
    agents_store[OPERATIONAL_QUERY_AGENT_ID] = Agent(
        id=OPERATIONAL_QUERY_AGENT_ID,
        name="Operational Query Agent",
        type="operational_query",
        status="active",
        created_at=datetime.now().isoformat(),
        metadata={
            "description": "Processes operational-level multi-variable queries (O1, O2, O3) using statistics and knowledge graph",
            "role": "operational_analysis",
            "capabilities": ["performance_monitoring", "absence_tracking", "engagement_monitoring", "departmental_analysis"]
        }
    )
    
    # Note: LLM agent is already created above (either restored from storage or newly created)
    
    save_agents()
    print(f"✅ Initialized {len(agents_store)} core agents:")
    print(f"   - Orchestrator Agent")
    print(f"   - Statistics Agent")
    print(f"   - Visualization Agent")
    print(f"   - Knowledge Graph Agent")
    print(f"   - HR Assistant (LLM)")
    print(f"   - Strategic Query Agent")
    print(f"   - Operational Query Agent")
    print(f"   Document agents will be created when documents are uploaded")

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def create_document_agent(document_name: str, document_id: str, document_type: str, file_path: Optional[str] = None) -> str:
    """Create a document-specific agent for processing a document"""
    agent_id = f"doc_{document_id}"
    
    if agent_id in document_agents:
        # Update file_path if provided
        if file_path and hasattr(document_agents[agent_id], 'file_path'):
            document_agents[agent_id].file_path = file_path
        return agent_id
    
    document_agents[agent_id] = DocumentAgent(
        id=agent_id,
        name=f"Document Agent - {document_name}",
        type="document",
        status="processing",
        created_at=datetime.now().isoformat(),
        document_name=document_name,
        document_id=document_id,
        document_type=document_type,
        file_path=file_path,
        facts_extracted=0,
        employee_names=[],  # Will be populated as data is processed
        data_range=None,
        columns_processed=[],  # Will be populated as columns are processed
        metadata={
            "description": f"Processes document: {document_name}",
            "assigned_to": [STATISTICS_AGENT_ID, KG_AGENT_ID],
            "statistics": None,
            "visualizations": None,
            "processed_at": None
        }
    )
    
    print(f"✅ Created document agent {agent_id} for '{document_name}'")
    return agent_id

def process_document_with_agents(document_id: str, document_name: str, document_type: str, 
                                 file_path: str, extracted_text: Optional[str] = None) -> Dict[str, Any]:
    """Process a document using all relevant agents"""
    doc_agent_id = create_document_agent(document_name, document_id, document_type, file_path=file_path)
    doc_agent = document_agents[doc_agent_id]
    
    results = {
        "document_id": document_id,
        "document_name": document_name,
        "statistics": None,
        "visualizations": None,
        "facts_extracted": 0,
        "kg_agent_id": KG_AGENT_ID,
        "statistics_agent_id": STATISTICS_AGENT_ID,
        "visualization_agent_id": VISUALIZATION_AGENT_ID
    }
    
    # OPTIMIZATION: For CSV files, read the file ONCE and pass DataFrame to all agents
    df = None
    if document_type.lower() == '.csv':
        try:
            import pandas as pd
            print(f"📖 Reading CSV file once (will be reused by all agents)...")
            # Detect separator
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')
                tab_count = first_line.count('\t')
                if semicolon_count > comma_count and semicolon_count > 0:
                    sep = ';'
                elif tab_count > comma_count and tab_count > 0:
                    sep = '\t'
                else:
                    sep = ','
            
            # Read full CSV file ONCE
            df = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
            
            print(f"✅ CSV loaded: {len(df)} rows × {len(df.columns)} columns")
        except Exception as csv_read_error:
            print(f"⚠️  Error reading CSV: {csv_read_error}")
            df = None
    
    # Step 1: Run Statistics, Visualization, and prepare KG text in parallel for CSV files
    statistics_result = None
    visualizations = None
    kg_text = extracted_text  # Initialize with provided text
    if document_type.lower() == '.csv' and df is not None:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time
            start_time = time.time()
            
            print(f"🚀 Running Statistics, Visualization, and KG text preparation in parallel...")
            
            # Prepare KG text extraction (doesn't depend on statistics) - uses DataFrame
            def prepare_kg_text():
                """Extract comprehensive text from CSV for KG agent - uses pre-loaded DataFrame"""
                try:
                    import pandas as pd
                    # Use pre-loaded DataFrame (sample for speed)
                    df_sample = df.head(200)  # Use first 200 rows from already-loaded DataFrame
                    
                    # Build comprehensive text representation for KG extraction
                    text_lines = []
                    text_lines.append(f"Human Resources Employee Dataset: {document_name}")
                    text_lines.append(f"Total records: {len(df)} rows in dataset (showing {len(df_sample)} sample rows)")
                    text_lines.append(f"Total attributes: {len(df.columns)} columns")
                    text_lines.append("")
                    text_lines.append("Dataset Attributes:")
                    for col in df_sample.columns:
                        col_type = "numeric" if pd.api.types.is_numeric_dtype(df_sample[col]) else "categorical"
                        unique_count = df_sample[col].nunique()
                        text_lines.append(f"- {col} ({col_type}): {unique_count} unique values")
                    text_lines.append("")
                    
                    # Sample records (limited for speed)
                    name_col = None
                    for col in df_sample.columns:
                        if any(keyword in col.lower() for keyword in ['name', 'employee', 'empname', 'employee_name','Employee_Name']):
                            name_col = col
                            break
                    
                    for idx, row in df_sample.head(50).iterrows():  # Reduced from 100 to 50 for speed
                        if name_col and name_col in df_sample.columns:
                            try:
                                name_val = row[name_col]
                                if pd.notna(name_val):
                                    employee_name = str(name_val).strip()
                                    fact_sentences = []
                                    for col, val in row.items():
                                        if pd.notna(val) and col != name_col:
                                            fact_sentences.append(f"{employee_name} has {col} {val}")
                                    if fact_sentences:
                                        text_lines.append(f"Record {idx + 1}: {' | '.join(fact_sentences[:5])}")  # Limit to 5 facts per record
                            except:
                                pass
                    
                    return "\n".join(text_lines)
                except Exception as e:
                    print(f"⚠️  Error preparing KG text: {e}")
                    return f"CSV Dataset: {document_name}\nColumns: {', '.join(df.columns.tolist()) if df is not None else 'unknown'}"
            
            # Run Statistics and KG text preparation in parallel - both use pre-loaded DataFrame
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit statistics agent and KG text preparation in parallel (both use DataFrame)
                stats_future = executor.submit(process_with_statistics_agent_from_df, df, document_name)
                kg_text_future = executor.submit(prepare_kg_text)
                
                # Wait for both to complete
                statistics_result = stats_future.result()
                results["statistics"] = statistics_result
                kg_text = kg_text_future.result()
                
                # Visualization depends on statistics, so run after stats completes
                if statistics_result:
                    visualizations = process_with_visualization_agent(statistics_result, document_name)
                    results["visualizations"] = visualizations
            
            elapsed = time.time() - start_time
            print(f"✅ Statistics, KG text preparation, and Visualization completed in {elapsed:.1f}s")
            
        except Exception as stats_error:
            print(f"⚠️  Statistics/Visualization/KG preparation error: {stats_error}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Knowledge Graph Agent - Extract facts from document
    # For CSV files, kg_text is already prepared in parallel above
    # For non-CSV files, use provided extracted_text
    if document_type.lower() != '.csv':
        kg_text = extracted_text
    
    # Step 3a: For CSV files, extract facts directly from DataFrame using parallel processing
    csv_facts_count = 0
    if document_type.lower() == '.csv' and df is not None:
        try:
            import time
            start_time = time.time()
            
            # Use pre-loaded DataFrame for fact extraction
            total_rows = len(df)
            total_cols = len(df.columns)
            complexity = total_rows * total_cols
            
            print(f"📊 CSV Fact Extraction Starting: {total_rows:,} rows × {total_cols} cols = {complexity:,} data points")
            
            # OPTIMIZED: Adaptive chunking with more aggressive parallelization
            # Target: ~500-1000 facts per chunk for better parallelization
            # Use more workers for larger files to maximize throughput
            if complexity < 5000:
                num_workers = 2
                chunk_size = max(25, total_rows // num_workers)
            elif complexity < 50000:
                # For medium files, use 4-6 workers
                num_workers = min(6, max(4, total_rows // 75))  # More workers, smaller chunks
                chunk_size = max(20, total_rows // num_workers)  # Smaller chunks for better load balancing
            else:
                # For large files, use 8-12 workers (if CPU cores allow)
                import os
                cpu_count = os.cpu_count() or 4
                max_workers = min(12, max(8, cpu_count))  # Use more workers if available
                num_workers = min(max_workers, max(6, total_rows // 150))  # More aggressive parallelization
                chunk_size = max(50, total_rows // num_workers)  # Smaller chunks for large files
            
            estimated_chunks = (total_rows + chunk_size - 1) // chunk_size
            print(f"🔄 Extracting facts using parallel processing:")
            print(f"   - Workers: {num_workers}")
            print(f"   - Chunk size: {chunk_size} rows")
            print(f"   - Estimated chunks: {estimated_chunks}")
            print(f"   - Data complexity: {complexity:,} data points")
            print(f"   - Started at: {time.strftime('%H:%M:%S')}")
            
            csv_facts_count = extract_csv_facts_directly_parallel_from_df(df, document_name, document_id, num_workers=num_workers, chunk_size=chunk_size, document_type=document_type)
            
            elapsed_time = time.time() - start_time
            print(f"✅ CSV extraction completed: {csv_facts_count} facts in {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
        except Exception as csv_facts_error:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            print(f"⚠️  Error in CSV fact extraction after {elapsed_time/60:.1f} minutes: {csv_facts_error}")
            import traceback
            traceback.print_exc()
    
    # Step 3b: Extract statistical facts directly from statistics (for CSV files)
    stats_facts_count = 0
    if document_type.lower() == '.csv' and results.get("statistics"):
        try:
            stats_facts_count = extract_statistical_facts(results["statistics"], document_name, document_id)
            print(f"📊 Extracted {stats_facts_count} statistical facts from correlations and statistics")
        except Exception as stats_facts_error:
            print(f"⚠️  Error extracting statistical facts: {stats_facts_error}")
            import traceback
            traceback.print_exc()
    
    # Step 3c: Skip redundant KG text processing for CSV files
    # Workers already construct triples directly from DataFrame rows
    # KG text processing is only needed for non-CSV files (PDF, DOCX, TXT)
    kg_result_facts = 0
    if document_type.lower() != '.csv' and kg_text and len(kg_text.strip()) > 10:
        try:
            print(f"🔄 Running KG agent for text extraction (non-CSV file)...")
            kg_result = process_with_kg_agent(kg_text, document_name, document_id)
            kg_result_facts = kg_result.get("facts_extracted", 0)
            results["kg_result"] = kg_result.get("result", "")
            print(f"✅ KG agent extracted {kg_result_facts} facts from text")
        except Exception as kg_error:
            print(f"⚠️  KG agent error: {kg_error}")
            import traceback
            traceback.print_exc()
    elif document_type.lower() == '.csv':
        print(f"⏭️  Skipping KG text processing for CSV (workers already construct triples directly)")
    
    # Total facts = direct CSV facts + statistical facts + text extraction facts (for non-CSV)
    results["facts_extracted"] = csv_facts_count + stats_facts_count + kg_result_facts
    print(f"📊 Total facts extracted: {results['facts_extracted']} (CSV: {csv_facts_count}, Stats: {stats_facts_count}, Text: {kg_result_facts})")
    
    # Step 4: Generate groupby insights and run operational queries in parallel
    operational_insights = {}
    groupby_insights = {}
    if document_type.lower() == '.csv' and results["facts_extracted"] > 0:
        try:
            print(f"🔄 Generating groupby insights and running operational queries in parallel...")
            from concurrent.futures import ThreadPoolExecutor
            import pandas as pd
            import time
            start_time = time.time()
            
            # Use pre-loaded DataFrame for groupby insights
            if df is not None:
                try:
                    def generate_groupby_insights():
                        """Generate insights using groupby operations - runs in parallel"""
                        insights = {}
                        numeric_cols = ['Salary', 'EmpSatisfaction', 'Absences', 'PerfScoreID', 'PerformanceScore', 'EngagementSurvey']
                        groupby_cols = ['Department', 'RecruitmentSource', 'ManagerName', 'EmploymentStatus']
                        
                        # Find actual column names (case-insensitive)
                        actual_numeric = []
                        for col in df.columns:
                            for nc in numeric_cols:
                                if nc.lower() in col.lower() or col.lower() in nc.lower():
                                    actual_numeric.append(col)
                                    break
                        
                        actual_groupby = []
                        for col in df.columns:
                            for gc in groupby_cols:
                                if gc.lower() in col.lower() or col.lower() in gc.lower():
                                    actual_groupby.append(col)
                                    break
                        
                        print(f"📊 Generating groupby insights: {len(actual_groupby)} grouping columns × {len(actual_numeric)} numeric columns")
                        
                        for group_col in actual_groupby:
                            if group_col not in df.columns:
                                continue
                            
                            group_insights = {}
                            for num_col in actual_numeric:
                                if num_col not in df.columns:
                                    continue
                                
                                try:
                                    # Convert to numeric
                                    df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
                                    # Group by and calculate statistics
                                    grouped = df.groupby(group_col)[num_col].agg(['mean', 'min', 'max', 'count']).round(2)
                                    group_insights[num_col] = grouped.to_dict('index')
                                except Exception as e:
                                    print(f"⚠️  Error grouping {group_col} by {num_col}: {e}")
                                    pass
                            
                            if group_insights:
                                insights[group_col] = group_insights
                        
                        return insights
                    
                    def run_operational_queries():
                        """Run operational queries - uses pre-loaded DataFrame instead of reconstructing from KG"""
                        ops_insights = {}
                    from strategic_query_agent import process_o1_1, process_o1_2, process_o2_1, process_o3_1
                    from strategic_query_agent import normalize_column_name
                    
                    queries = [
                        ("O1.1", process_o1_1, ["PerformanceScore", "PerfScoreID", "Department"]),
                        ("O1.2", process_o1_2, ["PerformanceScore", "PerfScoreID", "Department"]),
                        ("O2.1", process_o2_1, ["Absences", "EmploymentStatus", "Department", "Position"]),
                        ("O3.1", process_o3_1, ["EngagementSurvey", "ManagerName", "ManagerID"]),
                    ]
                    
                    for query_id, process_func, required_cols in queries:
                        try:
                            # Find actual column names
                            columns = {}
                            for var in required_cols:
                                actual_col = normalize_column_name(df, var)
                                if actual_col:
                                    columns[var] = actual_col
                            
                            # Check if all required columns are found
                            if len(columns) >= 2:  # At least 2 columns needed for operational queries
                                answer_parts, evidence_facts = process_func(df, columns)
                                answer = "".join(answer_parts) if isinstance(answer_parts, list) else str(answer_parts)
                                
                                if answer and len(answer) > 50 and "error" not in answer.lower() and "couldn't" not in answer.lower():
                                    ops_insights[query_id] = answer
                                    print(f"✅ {query_id} completed")
                                    
                                    # Store insight in knowledge graph for LLM access
                                    try:
                                        from knowledge import add_to_graph
                                        from datetime import datetime
                                        add_to_graph(
                                            f"Operational Insight {query_id}:\n{answer}",
                                            source_document="operational_insights",
                                            uploaded_at=datetime.now().isoformat(),
                                            agent_id="operational_query_agent"
                                        )
                                        print(f"✅ Stored operational insight in knowledge graph for LLM access")
                                    except Exception as store_error:
                                        print(f"⚠️  Error storing insight: {store_error}")
                            else:
                                print(f"⚠️  {query_id}: Missing required columns (found {len(columns)}/{len(required_cols)})")
                        except Exception as e:
                            print(f"⚠️  {query_id} query failed: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    return ops_insights
                
                    # Run groupby insights and operational queries in parallel
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        groupby_future = executor.submit(generate_groupby_insights)
                        ops_future = executor.submit(run_operational_queries)
                        
                        # Get results
                        groupby_insights = groupby_future.result()
                        operational_insights = ops_future.result()
                    
                    elapsed = time.time() - start_time
                    print(f"✅ Groupby insights and operational queries completed in {elapsed:.1f}s")
                    print(f"   Groupby insights: {len(groupby_insights)} grouping columns")
                    print(f"   Operational insights: {len(operational_insights)} queries")
                    
                    # Store groupby insights in results
                    results["groupby_insights"] = groupby_insights
                    
                    # Store groupby insights in knowledge graph for LLM access
                    if groupby_insights:
                        try:
                            from knowledge import add_to_graph
                            from datetime import datetime
                            
                            for group_col, metrics in groupby_insights.items():
                                for metric_col, group_data in metrics.items():
                                    insight_text = f"Groupby Analysis: {group_col} × {metric_col}\n"
                                    insight_text += f"Statistics by {group_col}:\n"
                                    for group_val, stats in group_data.items():
                                        insight_text += f"- {group_val}: mean={stats.get('mean', 'N/A')}, min={stats.get('min', 'N/A')}, max={stats.get('max', 'N/A')}, count={stats.get('count', 'N/A')}\n"
                                    
                                    add_to_graph(
                                        insight_text,
                                        source_document="groupby_insights",
                                        uploaded_at=datetime.now().isoformat(),
                                        agent_id="groupby_agent"
                                    )
                            
                            print(f"✅ Stored {sum(len(m) for m in groupby_insights.values())} groupby insights in knowledge graph")
                        except Exception as e:
                            print(f"⚠️  Error storing groupby insights: {e}")
                    
                    if operational_insights:
                        results["operational_insights"] = operational_insights
                        print(f"✅ Generated {len(operational_insights)} operational insights")
                    else:
                        print(f"⚠️  No operational insights generated (queries may have failed or data missing)")
                
                except Exception as df_error:
                    print(f"⚠️  Error in groupby/operational insights: {df_error}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as auto_query_error:
            print(f"⚠️  Error running insights generation: {auto_query_error}")
            import traceback
            traceback.print_exc()
    
    doc_agent.status = "completed"
    return results

def process_with_statistics_agent(file_path: str, document_name: str) -> Optional[Dict[str, Any]]:
    """Process document with Statistics Agent (legacy - reads file)"""
    stats_agent = agents_store.get(STATISTICS_AGENT_ID)
    if not stats_agent:
        return None
    
    stats_agent.status = "processing"
    
    try:
        # Import CSV analysis if available
        try:
            import pandas as pd
            
            # Detect separator
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')
                tab_count = first_line.count('\t')
                
                if semicolon_count > comma_count and semicolon_count > 0:
                    sep = ';'
                elif tab_count > comma_count and tab_count > 0:
                    sep = '\t'
                else:
                    sep = ','
            
            # Read CSV
            df = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
            
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
            
            return _process_statistics_from_df(df, document_name, stats_agent)
            
        except ImportError:
            print("⚠️  pandas not available, skipping statistical analysis")
            stats_agent.status = "active"
            return None
        except Exception as e:
            print(f"⚠️  Error in statistics agent: {e}")
            stats_agent.status = "active"
            import traceback
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"⚠️  Error in statistics agent (outer): {e}")
        stats_agent.status = "active"
        import traceback
        traceback.print_exc()
        return None

def process_with_statistics_agent_from_df(df, document_name: str) -> Optional[Dict[str, Any]]:
    """Process document with Statistics Agent using pre-loaded DataFrame"""
    stats_agent = agents_store.get(STATISTICS_AGENT_ID)
    if not stats_agent:
        return None
    
    stats_agent.status = "processing"
    
    try:
        return _process_statistics_from_df(df, document_name, stats_agent)
    except Exception as e:
        print(f"⚠️  Error in statistics agent: {e}")
        stats_agent.status = "active"
        import traceback
        traceback.print_exc()
        return None

def _process_statistics_from_df(df, document_name: str, stats_agent) -> Optional[Dict[str, Any]]:
    """Internal function to process statistics from DataFrame"""
    try:
        import pandas as pd
        
        # Perform statistical analysis
        analysis = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "column_types": {},
            "descriptive_stats": {},
            "missing_values": {},
            "data_quality": {}
        }
        
        # Analyze each column
        for col in df.columns:
                col_data = df[col]
                analysis["missing_values"][col] = int(col_data.isna().sum())
                analysis["data_quality"][col] = {
                    "completeness": float(1 - (col_data.isna().sum() / len(df))),
                    "unique_values": int(col_data.nunique())
                }
                
                # Detect column type and compute statistics
                if pd.api.types.is_numeric_dtype(col_data):
                    analysis["column_types"][col] = "numeric"
                    analysis["descriptive_stats"][col] = {
                        "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                        "median": float(col_data.median()) if not col_data.isna().all() else None,
                        "std": float(col_data.std()) if not col_data.isna().all() else None,
                        "min": float(col_data.min()) if not col_data.isna().all() else None,
                        "max": float(col_data.max()) if not col_data.isna().all() else None,
                        "q25": float(col_data.quantile(0.25)) if not col_data.isna().all() else None,
                        "q75": float(col_data.quantile(0.75)) if not col_data.isna().all() else None
                    }
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    analysis["column_types"][col] = "datetime"
                else:
                    analysis["column_types"][col] = "categorical"
                    value_counts = col_data.value_counts().head(10).to_dict()
                    analysis["descriptive_stats"][col] = {
                        "value_counts": {str(k): int(v) for k, v in value_counts.items()}
                    }
        
        # Compute correlations for numeric columns
        # Include ALL numeric columns in the correlation matrix
        numeric_cols = [col for col in df.columns if analysis["column_types"][col] == "numeric"]
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Build full symmetric correlation matrix (including self-correlations)
            analysis["correlations"] = {
                col: {
                    other_col: float(corr_matrix.loc[col, other_col])
                    for other_col in numeric_cols
                    if not pd.isna(corr_matrix.loc[col, other_col])
                }
                for col in numeric_cols
            }
            print(f"📊 Computed correlation matrix for {len(numeric_cols)} numeric columns")
        
        stats_agent.status = "active"
        print(f"✅ Statistics Agent analyzed {document_name}: {len(df)} rows, {len(df.columns)} columns")
        return analysis
    except Exception as e:
        print(f"⚠️  Error processing statistics from DataFrame: {e}")
        stats_agent.status = "active"
        import traceback
        traceback.print_exc()
        return None

def process_with_visualization_agent(statistics: Dict[str, Any], document_name: str) -> Dict[str, Any]:
    """Process statistics with Visualization Agent to create visualizations"""
    viz_agent = agents_store.get(VISUALIZATION_AGENT_ID)
    if not viz_agent:
        return {}
    
    viz_agent.status = "processing"
    
    visualizations = {
        "charts": [],
        "insights": []
    }
    
    try:
        # Generate visualization specifications based on statistics
        numeric_cols = [col for col, col_type in statistics.get("column_types", {}).items() 
                       if col_type == "numeric"]
        categorical_cols = [col for col, col_type in statistics.get("column_types", {}).items() 
                           if col_type == "categorical"]
        
        # Bar charts for categorical columns
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            value_counts = statistics.get("descriptive_stats", {}).get(col, {}).get("value_counts", {})
            if value_counts:
                visualizations["charts"].append({
                    "type": "bar",
                    "title": f"Distribution of {col}",
                    "data": value_counts,
                    "x_label": col,
                    "y_label": "Count"
                })
        
        # Histograms for numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            stats = statistics.get("descriptive_stats", {}).get(col, {})
            if stats:
                visualizations["charts"].append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "data": {
                        "mean": stats.get("mean"),
                        "median": stats.get("median"),
                        "std": stats.get("std"),
                        "min": stats.get("min"),
                        "max": stats.get("max")
                    },
                    "x_label": col,
                    "y_label": "Frequency"
                })
        
        # Correlation heatmap if correlations exist
        correlations = statistics.get("correlations", {})
        if correlations:
            visualizations["charts"].append({
                "type": "heatmap",
                "title": "Correlation Matrix",
                "data": correlations
            })
        
        # Generate insights
        insights = []
        if statistics.get("missing_values"):
            missing_cols = [col for col, count in statistics["missing_values"].items() if count > 0]
            if missing_cols:
                insights.append({
                    "type": "data_quality",
                    "message": f"{len(missing_cols)} columns have missing values",
                    "details": {col: statistics["missing_values"][col] for col in missing_cols[:5]}
                })
        
        visualizations["insights"] = insights
        
        viz_agent.status = "active"
        print(f"✅ Visualization Agent created {len(visualizations['charts'])} charts for {document_name}")
        return visualizations
        
    except Exception as e:
        print(f"❌ Visualization Agent error: {e}")
        import traceback
        traceback.print_exc()
        viz_agent.status = "active"
        return {}

def extract_csv_facts_directly_parallel_from_df(df, document_name: str, document_id: str, 
                                                 num_workers: int = 4, chunk_size: Optional[int] = None, 
                                                 document_type: str = ".csv") -> int:
    """
    Extract facts from CSV DataFrame using parallel processing with multiple worker agents.
    Uses pre-loaded DataFrame instead of reading file again.
    
    Args:
        df: Pre-loaded pandas DataFrame
        document_name: Name of the document
        document_id: ID of the document
        num_workers: Number of parallel workers (default: 4)
        chunk_size: Rows per chunk (auto-calculated if None)
        document_type: Type of document (e.g., ".csv")
    
    Returns:
        Total number of facts extracted
    """
    try:
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from knowledge import fact_exists, add_fact_source_document, graph, save_knowledge_graph
        from datetime import datetime
        import rdflib
        from urllib.parse import quote
        import threading
        
        # Thread-safe counter for facts
        facts_counter = {'added': 0, 'skipped': 0}
        facts_lock = threading.Lock()
        
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Find employee name column
        name_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['name', 'employee', 'empname', 'employee_name', 'employeename','Employee_Name']):
                name_col = col
                break
        
        if not name_col:
            name_col = df.columns[0] if len(df.columns) > 0 else None
        
        # Calculate optimal chunk size based on data complexity (rows × columns)
        # More columns = more facts per row = smaller chunks needed
        if chunk_size is None:
            # Calculate data complexity: rows × columns
            data_complexity = total_rows * total_cols
            
            # Target: ~10,000-20,000 data points per chunk (rows × columns)
            # This ensures balanced load across workers
            target_data_points_per_chunk = 15000  # Optimal for memory and processing
            
            # Calculate chunk size based on complexity
            if data_complexity > 0:
                calculated_chunk_size = max(25, min(200, target_data_points_per_chunk // total_cols))
            else:
                calculated_chunk_size = 100
            
            # Ensure we have enough chunks for parallelization
            # Minimum: num_workers chunks, maximum: reasonable chunk size
            min_chunks_needed = num_workers
            max_chunk_size_for_parallelization = max(25, total_rows // min_chunks_needed)
            
            # Use the smaller of: calculated size or max for parallelization
            chunk_size = min(calculated_chunk_size, max_chunk_size_for_parallelization)
            
            # Ensure minimum chunk size for efficiency (25 rows minimum for better parallelization)
            # Too small (< 25) creates excessive overhead from too many workers
            chunk_size = max(25, chunk_size)
            
            print(f"📊 Data complexity: {data_complexity:,} data points ({total_rows:,} rows × {total_cols} cols)")
            print(f"📊 Optimal chunk size: {chunk_size} rows (targeting ~{chunk_size * total_cols:,} data points per chunk)")
        
        # Split into chunks - CRITICAL: Ensure no rows are missed
        chunks = []
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunks.append((start_idx, end_idx))
        
        # Verify chunk coverage - ensure all rows are covered
        covered_rows = set()
        for start_idx, end_idx in chunks:
            for row_idx in range(start_idx, end_idx):
                covered_rows.add(row_idx)
        
        if len(covered_rows) != total_rows:
            missing_rows = set(range(total_rows)) - covered_rows
            print(f"⚠️  WARNING: Chunking may miss rows! Missing row indices: {sorted(missing_rows)[:20]}..." if len(missing_rows) > 20 else f"⚠️  WARNING: Chunking may miss rows! Missing row indices: {sorted(missing_rows)}")
        else:
            print(f"✅ Chunking verification: All {total_rows} rows are covered by chunks")
        
        print(f"📊 Splitting {total_rows} rows into {len(chunks)} chunks ({chunk_size} rows/chunk) for {num_workers} workers")
        
        # Ensure parent document agent tracks ALL columns upfront (before chunk processing)
        parent_agent_id = f"doc_{document_id}"
        if parent_agent_id in document_agents:
            parent_agent = document_agents[parent_agent_id]
            # Initialize with ALL columns from the DataFrame (ensures no columns are missed)
            with facts_lock:
                parent_agent.columns_processed = list(df.columns)
                parent_agent.data_range = {
                    "start": 0,
                    "end": total_rows,
                    "rows": total_rows,
                    "total_cols": total_cols,
                    "chunks": len(chunks),
                    "chunk_size": chunk_size
                }
            print(f"📊 Parent agent {parent_agent_id} initialized with {len(df.columns)} columns")
        
        uploaded_at = datetime.now().isoformat()
        
        def process_chunk(chunk_idx: int, start_idx: int, end_idx: int):
            """Process a single chunk of rows - each chunk is handled by a worker agent"""
            worker_id = f"doc_{document_id}_worker_{chunk_idx}"
            chunk_df = df.iloc[start_idx:end_idx].copy()
            chunk_facts_added = 0
            chunk_facts_skipped = 0
            
            # Create a worker agent for this chunk (visible in architecture)
            worker_agent_id = f"{worker_id}"
            worker_agent = DocumentAgent(
                id=worker_agent_id,
                name=f"Worker {chunk_idx + 1} - {document_name}",
                type="document_worker",
                status="processing",
                created_at=datetime.now().isoformat(),
                document_name=f"{document_name} (rows {start_idx}-{end_idx})",
                document_id=f"{document_id}_chunk_{chunk_idx}",
                document_type=document_type,
                facts_extracted=0,
                employee_names=[],  # Will be populated as rows are processed
                data_range={"start": start_idx, "end": end_idx, "rows": end_idx - start_idx},
                columns_processed=list(df.columns),  # Track all columns this worker processes
                metadata={
                    "description": f"Worker agent processing rows {start_idx}-{end_idx} of {document_name}",
                    "chunk_range": f"{start_idx}-{end_idx}",
                    "parent_document": document_id,
                    "worker_index": chunk_idx,
                    "chunk_size": end_idx - start_idx,
                    "total_rows": total_rows,
                    "total_cols": total_cols
                }
            )
            
            # Add worker agent to document_agents (thread-safe)
            with facts_lock:
                document_agents[worker_agent_id] = worker_agent
            
            chunk_employee_names = []  # Track employees processed by this worker
            chunk_columns_processed = set()  # Track all columns processed by this worker
            rows_processed_count = 0  # Track number of rows actually processed
            rows_skipped_count = 0  # Track number of rows skipped
            
            try:
                # First pass: track all columns in this chunk
                for col in chunk_df.columns:
                    chunk_columns_processed.add(col)
                
                # Log start of chunk processing
                expected_rows = end_idx - start_idx
                print(f"🔄 Worker {chunk_idx}: Processing rows {start_idx}-{end_idx-1} ({expected_rows} rows expected)")
                
                for idx, row in chunk_df.iterrows():
                    rows_processed_count += 1
                    
                    # Log progress every 100 rows or at start/end
                    if rows_processed_count == 1 or rows_processed_count % 100 == 0 or rows_processed_count == expected_rows:
                        print(f"   📊 Worker {chunk_idx}: Processing row {idx} ({rows_processed_count}/{expected_rows} rows processed)")
                    
                    # Extract employee name - handle quotes and ensure proper formatting
                    employee_name = None
                    if name_col and name_col in df.columns:
                        name_val = row[name_col]
                        if pd.notna(name_val):
                            employee_name = str(name_val).strip()
                            # Remove surrounding quotes if present (CSV might have "Name, First")
                            if employee_name.startswith('"') and employee_name.endswith('"'):
                                employee_name = employee_name[1:-1].strip()
                            # Also remove single quotes
                            if employee_name.startswith("'") and employee_name.endswith("'"):
                                employee_name = employee_name[1:-1].strip()
                    
                    # CRITICAL: Only skip if name is truly empty (not just whitespace)
                    # This ensures ALL employees are processed
                    if not employee_name or len(employee_name.strip()) == 0:
                        rows_skipped_count += 1
                        # Calculate absolute row number in the full dataset
                        absolute_row_num = start_idx + rows_processed_count - 1
                        print(f"⚠️  Worker {chunk_idx}: Skipping row {idx} (absolute row {absolute_row_num}) - No valid employee name")
                        continue
                    
                    # OPTIMIZATION: Pre-normalize employee name once per row (not per fact)
                    normalized_employee_name = employee_name.strip()
                    if normalized_employee_name.startswith('"') and normalized_employee_name.endswith('"'):
                        normalized_employee_name = normalized_employee_name[1:-1].strip()
                    if normalized_employee_name.startswith("'") and normalized_employee_name.endswith("'"):
                        normalized_employee_name = normalized_employee_name[1:-1].strip()
                    
                    # Track employee name for orchestrator routing (normalized, no quotes)
                    if employee_name not in chunk_employee_names:
                        chunk_employee_names.append(employee_name)
                    
                    # Also track in parent document agent
                    parent_agent_id = f"doc_{document_id}"
                    if parent_agent_id in document_agents:
                        parent_agent = document_agents[parent_agent_id]
                        if employee_name not in parent_agent.employee_names:
                            with facts_lock:
                                parent_agent.employee_names.append(employee_name)
                        # Columns are already tracked upfront, but ensure they're in the list
                        # (this is redundant but ensures thread-safety)
                        for col in chunk_df.columns:
                            if col not in parent_agent.columns_processed:
                                with facts_lock:
                                    parent_agent.columns_processed.append(col)
                    
                    # OPTIMIZATION: Collect facts for this row, then batch process
                    row_facts_to_check = []  # List of (employee_name, predicate, val_str) tuples
                    row_facts_data = []  # List of fact data for batch addition
                    row_columns_processed = []  # Track which columns were processed for this row
                    row_columns_skipped = []  # Track which columns were skipped (for verification)
                    
                    # CRITICAL: Process EVERY column in the row (row-by-row, column-by-column)
                    # This ensures ALL column values are maintained and converted to facts
                    for col, val in row.items():
                        # Skip only if this is the name column (it's the subject, not a fact)
                        if col == name_col:
                            row_columns_skipped.append(f"{col} (name column - subject)")
                            continue
                        
                        # Track that we're processing this column
                        row_columns_processed.append(col)
                        chunk_columns_processed.add(col)
                        
                        col_lower = str(col).lower().strip()
                        
                        # CRITICAL: Handle NaN values - convert to empty string but still process
                        # This ensures we track that the column exists even if value is missing
                        if pd.isna(val):
                            val_str = ""
                        else:
                            val_str = str(val).strip()
                        
                        # CRITICAL: Don't skip manager columns even if value seems empty
                        # ManagerName and ManagerID are important and should always be extracted
                        # For other columns, skip only if value is truly empty (not just whitespace)
                        is_manager_col = 'manager' in col_lower or 'managername' in col_lower or 'managerid' in col_lower
                        if not is_manager_col and (not val_str or len(val_str) == 0):
                            # Track skipped empty columns for verification
                            row_columns_skipped.append(f"{col} (empty value)")
                            continue
                        
                        # CRITICAL: Normalize value BEFORE creating predicate and collecting fact
                        # This ensures consistent fact checking and storage
                        # Convert numeric values to clean format (14.0 -> 14, but keep decimals if needed)
                        normalized_val = val_str
                        try:
                            if '.' in normalized_val:
                                float_val = float(normalized_val)
                                if float_val.is_integer():
                                    normalized_val = str(int(float_val))
                        except (ValueError, AttributeError):
                            pass  # Keep as string if conversion fails
                        
                        # Create predicate - handle all columns including absences, etc.
                        # Use standardized predicates for key columns to ensure consistent mapping
                        # IMPORTANT: Be specific to avoid duplicates (e.g., PositionID vs Position)
                        if 'salary' in col_lower:
                            predicate = "has salary"
                        elif ('position' in col_lower or 'job' in col_lower or 'title' in col_lower) and 'id' not in col_lower:
                            # Position, Job, Title (but NOT PositionID, JobID, etc.)
                            predicate = "has position"
                        elif ('position' in col_lower or 'job' in col_lower or 'positionid' in col_lower) and 'id' in col_lower:
                            # PositionID, JobID - use specific predicate to avoid duplicates
                            predicate = f"has {col.lower().replace('id', ' id')}"
                        elif 'department' in col_lower or 'dept' in col_lower or 'departmentid' in col_lower:
                            predicate = "works in department"
                        elif ('manager' in col_lower and 'name' in col_lower) or 'managername' in col_lower:
                            predicate = "has manager name"
                        elif ('manager' in col_lower and 'id' in col_lower) or 'managerid' in col_lower:
                            # ManagerID - use specific predicate
                            predicate = "has manager id"
                        elif 'manager' in col_lower and 'id' not in col_lower:
                            predicate = "has manager name"
                        elif 'age' in col_lower or 'dob' in col_lower:
                            predicate = "has age"
                        elif 'absence' in col_lower or 'absent' in col_lower or 'absences' in col_lower:
                            predicate = "has absences"
                        elif ('performance' in col_lower or 'perf' in col_lower or 'performancescore' in col_lower or 'perfscoreid' in col_lower) and ('score' in col_lower or 'id' in col_lower):
                            # Handle both PerformanceScore and PerfScoreID
                            predicate = "has performance score"
                        elif 'engagement' in col_lower and ('survey' in col_lower or 'score' in col_lower or 'engagementsurvey' in col_lower):
                            predicate = "has engagement survey"
                        elif 'employment' in col_lower and 'status' in col_lower:
                            predicate = "has employment status"
                        elif 'recruitment' in col_lower or ('source' in col_lower and 'recruit' in col_lower or 'recruitmentsource' in col_lower):
                            predicate = "has recruitment source"
                        elif 'date' in col_lower and ('hire' in col_lower or 'of' in col_lower):
                            predicate = "has dateofhire"
                        elif 'special' in col_lower and 'project' in col_lower:
                            predicate = "has specialprojectscount"
                        elif 'gender' in col_lower or 'sex' in col_lower:
                            predicate = "has gender"
                        elif 'marital' in col_lower:
                            predicate = "has marital status"
                        elif 'state' in col_lower and 'marital' not in col_lower:
                            predicate = "lives in state"
                        elif 'city' in col_lower:
                            predicate = "lives in city"
                        elif 'zip' in col_lower or 'postal' in col_lower:
                            predicate = "has zip code"
                        elif 'phone' in col_lower:
                            predicate = "has phone"
                        elif 'email' in col_lower:
                            predicate = "has email"
                        elif 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
                            predicate = f"has {col}"
                        elif 'id' in col_lower and 'employee' not in col_lower:
                            # ID columns get their own predicate to avoid duplicates
                            predicate = f"has {col}"
                        else:
                            # Default: use column name as predicate (ensures all columns are captured)
                            predicate = f"has {col}"
                        
                        # Collect fact for batch checking (use normalized value)
                        row_facts_to_check.append((normalized_employee_name, predicate, normalized_val))
                        row_facts_data.append((normalized_employee_name, predicate, normalized_val, col_lower))
                    
                    # OPTIMIZATION: Batch check all facts for this row in one lock acquisition
                    facts_to_add = []
                    facts_to_skip = []
                    with facts_lock:
                        from knowledge import fact_exists, _fact_lookup_set, normalize_entity
                        for emp_name, pred, val in row_facts_to_check:
                            if fact_exists(emp_name, pred, val):
                                facts_to_skip.append((emp_name, pred, val))
                            else:
                                facts_to_add.append((emp_name, pred, val))
                    
                    # OPTIMIZATION: Batch add all facts for this row in one lock acquisition
                    if facts_to_add:
                        with facts_lock:
                            from knowledge import graph, _fact_lookup_set, normalize_entity
                            triples_to_add = []
                            index_entries_to_add = []
                            source_docs_to_add = []
                            
                            for emp_name, pred, val in facts_to_add:
                                subject_clean = emp_name.replace(' ', '_')
                                predicate_clean = pred.strip().replace(' ', '_')
                                
                                # Value is already normalized (done before collection)
                                # Just ensure it's clean
                                object_clean = val.strip()
                                
                                subject_uri = rdflib.URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
                                predicate_uri = rdflib.URIRef(f"urn:predicate:{quote(predicate_clean, safe='')}")
                                object_literal = rdflib.Literal(object_clean)
                                
                                triples_to_add.append((subject_uri, predicate_uri, object_literal))
                                
                                # Update in-memory index
                                try:
                                    s_norm = normalize_entity(subject_clean.lower().replace('_', ' '))
                                    p_norm = predicate_clean.lower().replace('_', ' ')
                                    o_norm = normalize_entity(object_clean.lower())
                                    index_entries_to_add.append((s_norm, p_norm, o_norm))
                                except:
                                    pass
                                
                                # Collect source document info for batch processing
                                source_docs_to_add.append((emp_name, pred, val))
                            
                            # Batch add all triples to graph
                            for triple in triples_to_add:
                                graph.add(triple)
                            
                            # Batch update index
                            for entry in index_entries_to_add:
                                _fact_lookup_set.add(entry)
                            
                            chunk_facts_added += len(facts_to_add)
                            facts_counter['added'] += len(facts_to_add)
                        
                        # OPTIMIZATION: Batch process source documents outside the lock
                        # (add_fact_source_document has its own internal checks, so it's safe to call outside lock)
                        for emp_name, pred, val in source_docs_to_add:
                            try:
                                add_fact_source_document(emp_name, pred, val, document_name, uploaded_at)
                            except:
                                pass  # Don't fail if source tracking fails
                    
                    if facts_to_skip:
                        chunk_facts_skipped += len(facts_to_skip)
                        with facts_lock:
                            facts_counter['skipped'] += len(facts_to_skip)
                    
                    # CRITICAL: Verify that we processed facts for this row
                    # Ensure ALL columns were either processed or intentionally skipped
                    total_columns_in_row = len([c for c in row.index if c != name_col])
                    columns_processed_count = len(row_columns_processed)
                    columns_skipped_count = len(row_columns_skipped)
                    
                    # Verify column coverage: processed + skipped should equal total (excluding name column)
                    # Only log if there's an actual problem (missing columns)
                    if columns_processed_count + columns_skipped_count != total_columns_in_row:
                        skipped_col_names = [c.split(' (')[0] for c in row_columns_skipped]
                        processed_set = set(row_columns_processed)
                        skipped_set = set(skipped_col_names)
                        all_cols_set = set(row.index)
                        missing_cols = all_cols_set - skipped_set - processed_set - {name_col}
                        if missing_cols:
                            print(f"   ⚠️  Worker {chunk_idx}: Row {idx} ({normalized_employee_name}): Missing columns: {missing_cols}")
                    
                    # Only log if no facts were added (indicates a problem)
                    if len(facts_to_add) == 0 and len(facts_to_skip) == 0 and columns_processed_count > 0:
                        print(f"   ⚠️  Worker {chunk_idx}: Row {idx} ({normalized_employee_name}): No facts collected from {columns_processed_count} columns!")
                
                # Verify all rows were processed
                if rows_processed_count != expected_rows:
                    print(f"⚠️  Worker {chunk_idx}: WARNING - Expected {expected_rows} rows but processed {rows_processed_count} rows!")
                else:
                    print(f"✅ Worker {chunk_idx}: Successfully processed all {rows_processed_count} rows ({rows_skipped_count} skipped, {rows_processed_count - rows_skipped_count} with valid names)")
                
                # Update worker agent status and track employees and columns
                with facts_lock:
                    worker_agent.employee_names = chunk_employee_names
                    worker_agent.columns_processed = list(chunk_columns_processed)
                    worker_agent.facts_extracted = chunk_facts_added
                    worker_agent.status = "completed"
                    # Store row processing stats in metadata
                    worker_agent.metadata["rows_processed"] = rows_processed_count
                    worker_agent.metadata["rows_expected"] = expected_rows
                    worker_agent.metadata["rows_skipped"] = rows_skipped_count
                    worker_agent.metadata["rows_with_valid_names"] = rows_processed_count - rows_skipped_count
                
                return chunk_facts_added, chunk_facts_skipped
                
            except Exception as chunk_error:
                print(f"❌ Worker {chunk_idx} (rows {start_idx}-{end_idx}) failed: {chunk_error}")
                import traceback
                traceback.print_exc()
                with facts_lock:
                    worker_agent.status = "error"
                return 0, 0
        
        # Process chunks in parallel
        import time
        parallel_start_time = time.time()
        print(f"🚀 Starting parallel processing with {num_workers} workers at {time.strftime('%H:%M:%S')}...")
        print(f"📊 Workers will process {len(chunks)} chunks:")
        for idx, (start, end) in enumerate(chunks):
            rows_in_chunk = end - start
            print(f"   Worker {idx + 1}: rows {start}-{end} ({rows_in_chunk} rows)")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_chunk, idx, start, end): (idx, start, end) 
                      for idx, (start, end) in enumerate(chunks)}
            
            completed = 0
            total_rows_processed = 0
            total_rows_expected = 0
            last_progress_time = parallel_start_time
            for future in as_completed(futures):
                chunk_idx, start_idx, end_idx = futures[future]
                rows_in_chunk = end_idx - start_idx
                total_rows_expected += rows_in_chunk
                completed += 1
                try:
                    added, skipped = future.result()
                    facts_counter['added'] += added
                    facts_counter['skipped'] += skipped
                    
                    # Get row processing stats from worker agent
                    worker_agent_id = f"doc_{document_id}_worker_{chunk_idx}"
                    elapsed = time.time() - parallel_start_time
                    progress_pct = (completed / len(chunks)) * 100
                    if worker_agent_id in document_agents:
                        worker_agent = document_agents[worker_agent_id]
                        rows_processed = worker_agent.metadata.get("rows_processed", rows_in_chunk)
                        rows_skipped = worker_agent.metadata.get("rows_skipped", 0)
                        total_rows_processed += rows_processed
                        print(f"✅ Worker {chunk_idx + 1} completed: rows {start_idx}-{end_idx-1} ({rows_in_chunk} expected, {rows_processed} processed, {rows_skipped} skipped) → {added} facts added, {skipped} skipped [{completed}/{len(chunks)} = {progress_pct:.1f}%] ({elapsed/60:.1f}m elapsed)")
                    else:
                        total_rows_processed += rows_in_chunk
                        print(f"✅ Worker {chunk_idx + 1} completed: rows {start_idx}-{end_idx-1} ({rows_in_chunk} rows) → {added} facts added, {skipped} skipped [{completed}/{len(chunks)} = {progress_pct:.1f}%] ({elapsed/60:.1f}m elapsed)")
                except Exception as e:
                    print(f"❌ Worker {chunk_idx + 1} (rows {start_idx}-{end_idx-1}) failed: {e}")
            
            # Verify all rows were processed
            parallel_elapsed = time.time() - parallel_start_time
            print(f"\n📊 ROW PROCESSING VERIFICATION:")
            print(f"   Total rows in dataset: {total_rows}")
            print(f"   Total rows expected across all chunks: {total_rows_expected}")
            print(f"   Total rows processed: {total_rows_processed}")
            if total_rows_processed == total_rows_expected == total_rows:
                print(f"   ✅ SUCCESS: All {total_rows} rows were processed!")
            else:
                missing = total_rows - total_rows_processed
                print(f"   ⚠️  WARNING: {missing} rows may not have been processed!")
                print(f"   Expected: {total_rows}, Processed: {total_rows_processed}, Difference: {missing}")
            print(f"⏱️  Parallel processing completed in {parallel_elapsed/60:.1f} minutes ({parallel_elapsed:.1f} seconds)")
        
        # Aggregate results from all workers
        parent_agent_id = f"doc_{document_id}"
        if parent_agent_id in document_agents:
            parent_agent = document_agents[parent_agent_id]
            # Aggregate employee names and columns from all workers
            all_employee_names = set(parent_agent.employee_names)
            all_columns = set(parent_agent.columns_processed)
            
            for worker_id, worker_agent in document_agents.items():
                if (worker_agent.type == "document_worker" and 
                    worker_agent.metadata.get("parent_document") == document_id):
                    all_employee_names.update(worker_agent.employee_names)
                    all_columns.update(worker_agent.columns_processed)
            
            with facts_lock:
                parent_agent.employee_names = list(all_employee_names)
                parent_agent.columns_processed = list(all_columns)
            
            print(f"✅ Parent agent {parent_agent_id} aggregated: {len(all_employee_names)} employees, {len(all_columns)} columns")
        
        print(f"✅ Parallel CSV extraction completed: {facts_counter['added']} facts added, {facts_counter['skipped']} duplicates skipped")
        print(f"📊 Document split into {len(chunks)} chunks, processed by {num_workers} workers")
        
        return facts_counter['added']
        
    except Exception as e:
        print(f"❌ Error in parallel CSV fact extraction: {e}")
        import traceback
        traceback.print_exc()
        return 0

def extract_csv_facts_directly_parallel(file_path: str, document_name: str, document_id: str, 
                                        num_workers: int = 4, chunk_size: Optional[int] = None, 
                                        document_type: str = ".csv") -> int:
    """
    Extract facts from CSV using parallel processing with multiple worker agents.
    Each worker processes a different chunk of rows concurrently.
    
    Args:
        file_path: Path to CSV file
        document_name: Name of the document
        document_id: ID of the document
        num_workers: Number of parallel workers (default: 4)
        chunk_size: Rows per chunk (auto-calculated if None)
    
    Returns:
        Total number of facts extracted
    """
    try:
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from knowledge import fact_exists, add_fact_source_document, graph, save_knowledge_graph
        from datetime import datetime
        import rdflib
        from urllib.parse import quote
        import threading
        
        # Thread-safe counter for facts
        facts_counter = {'added': 0, 'skipped': 0}
        facts_lock = threading.Lock()
        
        # Detect separator
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            tab_count = first_line.count('\t')
            if semicolon_count > comma_count and semicolon_count > 0:
                sep = ';'
            elif tab_count > comma_count and tab_count > 0:
                sep = '\t'
            else:
                sep = ','
        
        # Read entire CSV to get total rows
        print(f"📊 Reading CSV file for parallel processing...")
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
        
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Find employee name column
        name_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['name', 'employee', 'empname', 'employee_name', 'employeename']):
                name_col = col
                break
        
        if not name_col:
            name_col = df.columns[0] if len(df.columns) > 0 else None
        
        # Calculate optimal chunk size based on data complexity (rows × columns)
        # More columns = more facts per row = smaller chunks needed
        if chunk_size is None:
            # Calculate data complexity: rows × columns
            data_complexity = total_rows * total_cols
            
            # Target: ~10,000-20,000 data points per chunk (rows × columns)
            # This ensures balanced load across workers
            target_data_points_per_chunk = 15000  # Optimal for memory and processing
            
            # Calculate chunk size based on complexity
            if data_complexity > 0:
                calculated_chunk_size = max(25, min(200, target_data_points_per_chunk // total_cols))
            else:
                calculated_chunk_size = 100
            
            # Ensure we have enough chunks for parallelization
            # Minimum: num_workers chunks, maximum: reasonable chunk size
            min_chunks_needed = num_workers
            max_chunk_size_for_parallelization = max(25, total_rows // min_chunks_needed)
            
            # Use the smaller of: calculated size or max for parallelization
            chunk_size = min(calculated_chunk_size, max_chunk_size_for_parallelization)
            
            # Ensure minimum chunk size for efficiency (25 rows minimum for better parallelization)
            # Too small (< 25) creates excessive overhead from too many workers
            chunk_size = max(25, chunk_size)
            
            print(f"📊 Data complexity: {data_complexity:,} data points ({total_rows:,} rows × {total_cols} cols)")
            print(f"📊 Optimal chunk size: {chunk_size} rows (targeting ~{chunk_size * total_cols:,} data points per chunk)")
        
        # Split into chunks - CRITICAL: Ensure no rows are missed
        chunks = []
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunks.append((start_idx, end_idx))
        
        # Verify chunk coverage - ensure all rows are covered
        covered_rows = set()
        for start_idx, end_idx in chunks:
            for row_idx in range(start_idx, end_idx):
                covered_rows.add(row_idx)
        
        if len(covered_rows) != total_rows:
            missing_rows = set(range(total_rows)) - covered_rows
            print(f"⚠️  WARNING: Chunking may miss rows! Missing row indices: {sorted(missing_rows)[:20]}..." if len(missing_rows) > 20 else f"⚠️  WARNING: Chunking may miss rows! Missing row indices: {sorted(missing_rows)}")
        else:
            print(f"✅ Chunking verification: All {total_rows} rows are covered by chunks")
        
        print(f"📊 Splitting {total_rows} rows into {len(chunks)} chunks ({chunk_size} rows/chunk) for {num_workers} workers")
        
        # Ensure parent document agent tracks ALL columns upfront (before chunk processing)
        parent_agent_id = f"doc_{document_id}"
        if parent_agent_id in document_agents:
            parent_agent = document_agents[parent_agent_id]
            # Initialize with ALL columns from the DataFrame (ensures no columns are missed)
            with facts_lock:
                parent_agent.columns_processed = list(df.columns)
                parent_agent.data_range = {
                    "start": 0,
                    "end": total_rows,
                    "rows": total_rows,
                    "total_cols": total_cols,
                    "chunks": len(chunks),
                    "chunk_size": chunk_size
                }
            print(f"📊 Parent agent {parent_agent_id} initialized with {len(df.columns)} columns")
        
        uploaded_at = datetime.now().isoformat()
        
        def process_chunk(chunk_idx: int, start_idx: int, end_idx: int):
            """Process a single chunk of rows - each chunk is handled by a worker agent"""
            worker_id = f"doc_{document_id}_worker_{chunk_idx}"
            chunk_df = df.iloc[start_idx:end_idx].copy()
            chunk_facts_added = 0
            chunk_facts_skipped = 0
            
            # Create a worker agent for this chunk (visible in architecture)
            worker_agent_id = f"{worker_id}"
            worker_agent = DocumentAgent(
                id=worker_agent_id,
                name=f"Worker {chunk_idx + 1} - {document_name}",
                type="document_worker",
                status="processing",
                created_at=datetime.now().isoformat(),
                document_name=f"{document_name} (rows {start_idx}-{end_idx})",
                document_id=f"{document_id}_chunk_{chunk_idx}",
                document_type=document_type,
                facts_extracted=0,
                employee_names=[],  # Will be populated as rows are processed
                data_range={"start": start_idx, "end": end_idx, "rows": end_idx - start_idx},
                columns_processed=list(df.columns),  # Track all columns this worker processes
                metadata={
                    "description": f"Worker agent processing rows {start_idx}-{end_idx} of {document_name}",
                    "chunk_range": f"{start_idx}-{end_idx}",
                    "parent_document": document_id,
                    "worker_index": chunk_idx,
                    "chunk_size": end_idx - start_idx,
                    "total_rows": total_rows,
                    "total_cols": total_cols
                }
            )
            
            # Add worker agent to document_agents (thread-safe)
            with facts_lock:
                document_agents[worker_agent_id] = worker_agent
            
            chunk_employee_names = []  # Track employees processed by this worker
            chunk_columns_processed = set()  # Track all columns processed by this worker
            rows_processed_count = 0  # Track number of rows actually processed
            rows_skipped_count = 0  # Track number of rows skipped
            
            try:
                # First pass: track all columns in this chunk
                for col in chunk_df.columns:
                    chunk_columns_processed.add(col)
                
                # Log start of chunk processing
                expected_rows = end_idx - start_idx
                print(f"🔄 Worker {chunk_idx}: Processing rows {start_idx}-{end_idx-1} ({expected_rows} rows expected)")
                
                for idx, row in chunk_df.iterrows():
                    rows_processed_count += 1
                    
                    # Log progress every 100 rows or at start/end
                    if rows_processed_count == 1 or rows_processed_count % 100 == 0 or rows_processed_count == expected_rows:
                        print(f"   📊 Worker {chunk_idx}: Processing row {idx} ({rows_processed_count}/{expected_rows} rows processed)")
                    
                    # Extract employee name - handle quotes and ensure proper formatting
                    employee_name = None
                    if name_col and name_col in df.columns:
                        name_val = row[name_col]
                        if pd.notna(name_val):
                            employee_name = str(name_val).strip()
                            # Remove surrounding quotes if present (CSV might have "Name, First")
                            if employee_name.startswith('"') and employee_name.endswith('"'):
                                employee_name = employee_name[1:-1].strip()
                            # Also remove single quotes
                            if employee_name.startswith("'") and employee_name.endswith("'"):
                                employee_name = employee_name[1:-1].strip()
                    
                    # CRITICAL: Only skip if name is truly empty (not just whitespace)
                    # This ensures ALL employees are processed
                    if not employee_name or len(employee_name.strip()) == 0:
                        rows_skipped_count += 1
                        # Calculate absolute row number in the full dataset
                        absolute_row_num = start_idx + rows_processed_count - 1
                        print(f"⚠️  Worker {chunk_idx}: Skipping row {idx} (absolute row {absolute_row_num}) - No valid employee name")
                        continue
                    
                    # Track employee name for orchestrator routing (normalized, no quotes)
                    if employee_name not in chunk_employee_names:
                        chunk_employee_names.append(employee_name)
                    
                    # Also track in parent document agent
                    parent_agent_id = f"doc_{document_id}"
                    if parent_agent_id in document_agents:
                        parent_agent = document_agents[parent_agent_id]
                        if employee_name not in parent_agent.employee_names:
                            with facts_lock:
                                parent_agent.employee_names.append(employee_name)
                        # Columns are already tracked upfront, but ensure they're in the list
                        # (this is redundant but ensures thread-safety)
                        for col in chunk_df.columns:
                            if col not in parent_agent.columns_processed:
                                with facts_lock:
                                    parent_agent.columns_processed.append(col)
                    
                    # Create facts for each column (ensure ALL columns are processed, including last column)
                    for col, val in row.items():
                        # Skip only if value is NaN or this is the name column
                        if pd.isna(val) or col == name_col:
                            continue
                        
                        # Ensure column is tracked
                        chunk_columns_processed.add(col)
                        
                        col_lower = str(col).lower().strip()
                        val_str = str(val).strip()
                        
                        # CRITICAL: Don't skip manager columns even if value seems empty
                        # ManagerName and ManagerID are important and should always be extracted
                        # Only skip if value is truly empty (not just whitespace or "0")
                        is_manager_col = 'manager' in col_lower
                        if not is_manager_col and (not val_str or len(val_str) == 0):
                            continue
                        
                        # Create predicate - handle all columns including absences, etc.
                        # Use standardized predicates for key columns to ensure consistent mapping
                        if 'salary' in col_lower:
                            predicate = "has salary"
                        elif 'position' in col_lower or 'job' in col_lower or 'title' in col_lower:
                            predicate = "has position"
                        elif 'department' in col_lower or 'dept' in col_lower:
                            predicate = "works in department"
                        elif 'age' in col_lower:
                            predicate = "has age"
                        elif 'absence' in col_lower or 'absent' in col_lower:
                            predicate = "has absences"
                        elif ('performance' in col_lower or 'perf' in col_lower) and ('score' in col_lower or 'id' in col_lower):
                            # Handle both PerformanceScore and PerfScoreID
                            predicate = "has performance score"
                        elif 'engagement' in col_lower and ('survey' in col_lower or 'score' in col_lower):
                            predicate = "has engagement survey"
                            # Also store with exact column name for direct matching
                            # This ensures "EngagementSurvey" column is preserved
                        elif 'employment' in col_lower and 'status' in col_lower:
                            predicate = "has employment status"
                        elif 'recruitment' in col_lower or ('source' in col_lower and 'recruit' in col_lower):
                            predicate = "has recruitment source"
                        elif 'manager' in col_lower and 'name' in col_lower:
                            predicate = "has manager name"
                        elif 'manager' in col_lower and 'id' in col_lower:
                            # ManagerID - use specific predicate
                            predicate = "has manager id"
                        elif 'manager' in col_lower:
                            predicate = "has manager name"
                        elif 'date' in col_lower and ('hire' in col_lower or 'of' in col_lower):
                            predicate = "has dateofhire"
                        elif 'special' in col_lower and 'project' in col_lower:
                            predicate = "has specialprojectscount"
                        elif 'gender' in col_lower or 'sex' in col_lower:
                            predicate = "has gender"
                        elif 'marital' in col_lower:
                            predicate = "has marital status"
                        elif 'state' in col_lower and 'marital' not in col_lower:
                            predicate = "lives in state"
                        elif 'city' in col_lower:
                            predicate = "lives in city"
                        elif 'zip' in col_lower or 'postal' in col_lower:
                            predicate = "has zip code"
                        elif 'phone' in col_lower:
                            predicate = "has phone"
                        elif 'email' in col_lower:
                            predicate = "has email"
                        elif 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
                            predicate = f"has {col}"
                        elif 'id' in col_lower and 'employee' not in col_lower:
                            predicate = f"has {col}"
                        else:
                            # Default: use column name as predicate (ensures all columns are captured)
                            predicate = f"has {col}"
                        
                        # CRITICAL: Normalize employee name (remove quotes, ensure consistent format)
                        # This ensures names are stored consistently for reliable lookup
                        normalized_employee_name = employee_name.strip()
                        if normalized_employee_name.startswith('"') and normalized_employee_name.endswith('"'):
                            normalized_employee_name = normalized_employee_name[1:-1].strip()
                        if normalized_employee_name.startswith("'") and normalized_employee_name.endswith("'"):
                            normalized_employee_name = normalized_employee_name[1:-1].strip()
                        
                        # IMPROVED: Normalize value before checking (14.0 -> 14)
                        normalized_val = val_str.strip()
                        try:
                            if '.' in normalized_val:
                                float_val = float(normalized_val)
                                if float_val.is_integer():
                                    normalized_val = str(int(float_val))
                        except (ValueError, AttributeError):
                            pass  # Keep as string if conversion fails
                        
                        # Check if fact already exists (thread-safe, use normalized values)
                        with facts_lock:
                            fact_exists_check = fact_exists(normalized_employee_name, predicate, normalized_val)
                        
                        if not fact_exists_check:
                            # Add fact directly to graph (thread-safe)
                            with facts_lock:
                                subject_clean = normalized_employee_name.replace(' ', '_')
                                predicate_clean = predicate.strip().replace(' ', '_')
                                object_clean = normalized_val
                                
                                subject_uri = rdflib.URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
                                predicate_uri = rdflib.URIRef(f"urn:predicate:{quote(predicate_clean, safe='')}")
                                object_literal = rdflib.Literal(object_clean)
                                
                                # WORKER CONSTRUCTS TRIPLE DIRECTLY - This is parallel KG construction!
                                graph.add((subject_uri, predicate_uri, object_literal))
                                
                                # Update in-memory index for fast future lookups (O(1) instead of O(n))
                                try:
                                    from knowledge import _fact_lookup_set, normalize_entity
                                    s_norm = normalize_entity(subject_clean.lower().replace('_', ' '))
                                    p_norm = predicate_clean.lower().replace('_', ' ')
                                    o_norm = normalize_entity(object_clean.lower())
                                    _fact_lookup_set.add((s_norm, p_norm, o_norm))
                                except:
                                    pass  # Index update is optional, don't fail if it errors
                                
                                # Use normalized name for source document tracking
                                add_fact_source_document(normalized_employee_name, predicate, val_str, document_name, uploaded_at)
                                
                                chunk_facts_added += 1
                                facts_counter['added'] += 1
                        else:
                            chunk_facts_skipped += 1
                            with facts_lock:
                                facts_counter['skipped'] += 1
                
                # Update worker agent status and track employees and columns
                with facts_lock:
                    if worker_agent_id in document_agents:
                        document_agents[worker_agent_id].facts_extracted = chunk_facts_added
                        document_agents[worker_agent_id].employee_names = chunk_employee_names
                        document_agents[worker_agent_id].columns_processed = list(chunk_columns_processed)
                        document_agents[worker_agent_id].status = "completed"
                
                return (worker_id, chunk_facts_added, chunk_facts_skipped, None)
            except Exception as e:
                # Mark worker as failed
                with facts_lock:
                    if worker_agent_id in document_agents:
                        document_agents[worker_agent_id].status = "failed"
                        document_agents[worker_agent_id].metadata["error"] = str(e)
                return (worker_id, chunk_facts_added, chunk_facts_skipped, str(e))
        
        # Process chunks in parallel
        print(f"🚀 Starting parallel processing with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            futures = {executor.submit(process_chunk, idx, start, end): (idx, start, end) 
                      for idx, (start, end) in enumerate(chunks)}
            
            # Collect results as they complete
            completed_chunks = 0
            for future in as_completed(futures):
                chunk_idx, start_idx, end_idx = futures[future]
                try:
                    worker_id, added, skipped, error = future.result()
                    completed_chunks += 1
                    if error:
                        print(f"⚠️  Worker {worker_id} (rows {start_idx}-{end_idx}) error: {error}")
                    else:
                        print(f"✅ Worker {worker_id} (rows {start_idx}-{end_idx}): {added} facts added, {skipped} skipped [{completed_chunks}/{len(chunks)}]")
                except Exception as e:
                    print(f"❌ Worker {chunk_idx} (rows {start_idx}-{end_idx}) failed: {e}")
        
        # Aggregate results from all workers into parent document agent
        parent_agent_id = f"doc_{document_id}"
        if parent_agent_id in document_agents:
            parent_agent = document_agents[parent_agent_id]
            # Aggregate employee names and columns from all workers
            all_employee_names = set(parent_agent.employee_names)
            all_columns = set(parent_agent.columns_processed)
            
            # Collect from all worker agents
            for worker_agent_id, worker_agent in document_agents.items():
                if (worker_agent.type == "document_worker" and 
                    worker_agent.metadata.get("parent_document") == document_id):
                    all_employee_names.update(worker_agent.employee_names)
                    all_columns.update(worker_agent.columns_processed)
            
            # Update parent agent with aggregated data
            parent_agent.employee_names = sorted(list(all_employee_names))
            parent_agent.columns_processed = sorted(list(all_columns))
            parent_agent.facts_extracted = facts_counter['added']
            parent_agent.status = "completed"
            
            print(f"✅ Parent agent {parent_agent_id} aggregated: {len(parent_agent.employee_names)} employees, {len(parent_agent.columns_processed)} columns")
        
        # Save the graph after all workers complete
        save_knowledge_graph()
        
        total_added = facts_counter['added']
        total_skipped = facts_counter['skipped']
        print(f"✅ Parallel CSV extraction completed: {total_added} facts added, {total_skipped} duplicates skipped")
        print(f"📊 Document split into {len(chunks)} chunks, processed by {num_workers} workers")
        return total_added
        
    except Exception as e:
        print(f"⚠️  Error in parallel CSV fact extraction: {e}")
        import traceback
        traceback.print_exc()
        return 0

def extract_csv_facts_directly(file_path: str, document_name: str, document_id: str) -> int:
    """
    Extract facts directly from CSV DataFrame for better performance and completeness.
    Processes ALL rows and creates facts directly without text conversion.
    """
    try:
        import pandas as pd
        from knowledge import fact_exists, add_fact_source_document
        from datetime import datetime
        import rdflib
        from urllib.parse import quote
        
        # Detect separator
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            tab_count = first_line.count('\t')
            if semicolon_count > comma_count and semicolon_count > 0:
                sep = ';'
            elif tab_count > comma_count and tab_count > 0:
                sep = '\t'
            else:
                sep = ','
        
        # Read ALL rows (no limit for direct fact creation)
        print(f"📊 Reading CSV file for direct fact extraction...")
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
        
        total_rows = len(df)
        print(f"📊 Processing {total_rows} rows with {len(df.columns)} columns for direct fact extraction")
        
        # Find employee name column
        name_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['name', 'employee', 'empname', 'employee_name', 'employeename']):
                name_col = col
                break
        
        if not name_col:
            print(f"⚠️  No employee name column found, using first column as identifier")
            name_col = df.columns[0] if len(df.columns) > 0 else None
        
        uploaded_at = datetime.now().isoformat()
        facts_added = 0
        facts_skipped = 0
        
        # Process rows in batches for progress logging
        batch_size = 100
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            if batch_start % 500 == 0 or batch_start == 0:
                print(f"   📊 Processing rows {batch_start + 1} to {batch_end} of {total_rows}...")
            
            for idx in range(batch_start, batch_end):
                row = df.iloc[idx]
                
                # Extract employee name - handle quotes and ensure proper formatting
                employee_name = None
                if name_col and name_col in df.columns:
                    name_val = row[name_col]
                    if pd.notna(name_val):
                        employee_name = str(name_val).strip()
                        # Remove surrounding quotes if present (CSV might have "Name, First")
                        if employee_name.startswith('"') and employee_name.endswith('"'):
                            employee_name = employee_name[1:-1].strip()
                        # Also remove single quotes
                        if employee_name.startswith("'") and employee_name.endswith("'"):
                            employee_name = employee_name[1:-1].strip()
                
                # CRITICAL: Only skip if name is truly empty (not just whitespace)
                # This ensures ALL employees are processed
                if not employee_name or len(employee_name.strip()) == 0:
                    print(f"⚠️  Skipping row {idx}: No valid employee name found")
                    continue
                
                # CRITICAL: Normalize employee name (remove quotes, ensure consistent format)
                normalized_employee_name = employee_name.strip()
                if normalized_employee_name.startswith('"') and normalized_employee_name.endswith('"'):
                    normalized_employee_name = normalized_employee_name[1:-1].strip()
                if normalized_employee_name.startswith("'") and normalized_employee_name.endswith("'"):
                    normalized_employee_name = normalized_employee_name[1:-1].strip()
                
                # Create facts for each column
                for col, val in row.items():
                    if pd.isna(val) or col == name_col:
                        continue
                    
                    col_lower = str(col).lower()
                    val_str = str(val).strip()
                    
                    # CRITICAL: Don't skip manager columns even if value seems empty
                    # ManagerName and ManagerID are important and should always be extracted
                    is_manager_col = 'manager' in col_lower
                    if not is_manager_col and (not val_str or len(val_str) == 0):
                        continue
                    
                    # IMPROVED: Normalize value (14.0 -> 14) before processing
                    normalized_val = val_str
                    try:
                        if '.' in normalized_val:
                            float_val = float(normalized_val)
                            if float_val.is_integer():
                                normalized_val = str(int(float_val))
                    except (ValueError, AttributeError):
                        pass  # Keep as string if conversion fails
                    
                    # Create predicate based on column type - match parallel processing logic
                    if 'salary' in col_lower:
                        predicate = "has salary"
                    elif 'position' in col_lower or 'job' in col_lower or 'title' in col_lower:
                        predicate = "has position"
                    elif 'department' in col_lower or 'dept' in col_lower:
                        predicate = "works in department"
                    elif 'age' in col_lower:
                        predicate = "has age"
                    elif 'absence' in col_lower or 'absent' in col_lower:
                        predicate = "has absences"
                    elif ('performance' in col_lower or 'perf' in col_lower) and ('score' in col_lower or 'id' in col_lower):
                        predicate = "has performance score"
                    elif 'engagement' in col_lower and ('survey' in col_lower or 'score' in col_lower):
                        predicate = "has engagement survey"
                    elif 'employment' in col_lower and 'status' in col_lower:
                        predicate = "has employment status"
                    elif 'manager' in col_lower and 'name' in col_lower:
                        predicate = "has manager name"
                    elif 'manager' in col_lower and 'id' in col_lower:
                        predicate = "has manager id"
                    elif 'manager' in col_lower:
                        predicate = "has manager name"
                    else:
                        # For columns that match exact CSV names (capitalized), use the column name directly
                        if col[0].isupper() or col.isupper():
                            predicate = f"has {col}"  # e.g., "has EngagementSurvey"
                        else:
                            predicate = f"has {col}"
                    
                    # Check if fact already exists (use normalized values)
                    if not fact_exists(normalized_employee_name, predicate, normalized_val):
                        # Add fact directly to graph
                        from knowledge import graph
                        subject_clean = normalized_employee_name.replace(' ', '_')
                        predicate_clean = predicate.strip().replace(' ', '_')
                        object_clean = normalized_val
                        
                        subject_uri = rdflib.URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
                        predicate_uri = rdflib.URIRef(f"urn:predicate:{quote(predicate_clean, safe='')}")
                        object_literal = rdflib.Literal(object_clean)
                        
                        graph.add((subject_uri, predicate_uri, object_literal))
                        
                        # Add source document (use normalized name and value)
                        add_fact_source_document(normalized_employee_name, predicate, object_clean, document_name, uploaded_at)
                        
                        facts_added += 1
                    else:
                        facts_skipped += 1
        
        # Save the graph after adding facts
        from knowledge import save_knowledge_graph
        save_knowledge_graph()
        
        print(f"✅ Direct CSV extraction: Added {facts_added} facts, skipped {facts_skipped} duplicates")
        return facts_added
        
    except Exception as e:
        print(f"⚠️  Error in direct CSV fact extraction: {e}")
        import traceback
        traceback.print_exc()
        return 0

def extract_column_names_from_query(query: str, available_columns: List[str]) -> Optional[Tuple[str, str]]:
    """
    Extract two column names from a correlation query.
    Handles patterns like "correlation between X and Y", "X and Y correlation", etc.
    Returns (col1, col2) if found, None otherwise.
    """
    from typing import List, Optional, Tuple
    import re
    
    query_lower = query.lower()
    
    # Pattern: "correlation between X and Y"
    patterns = [
        r'correlation\s+between\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)\s+and\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)(?:\s|$|\?|\.)',
        r'([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)\s+and\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)\s+correlation',
        r'correlation\s+of\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)\s+and\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)(?:\s|$|\?|\.)',
        r'([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)\s+vs\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)\s+correlation',
        r'what\s+is\s+the\s+correlation\s+between\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)\s+and\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*?)(?:\s|$|\?|\.)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            col1_candidate = match.group(1).strip()
            col2_candidate = match.group(2).strip()
            
            # Try to match to actual column names (case-insensitive, handle variations)
            col1_matched = None
            col2_matched = None
            
            for col in available_columns:
                col_lower = col.lower().replace('_', '').replace(' ', '').replace('-', '')
                col1_lower = col1_candidate.lower().replace('_', '').replace(' ', '').replace('-', '')
                col2_lower = col2_candidate.lower().replace('_', '').replace(' ', '').replace('-', '')
                
                # More flexible matching - check if one contains the other or they're very similar
                if not col1_matched and (col1_lower == col_lower or col1_lower in col_lower or col_lower in col1_lower or 
                                        (len(col1_lower) > 3 and col1_lower[:4] in col_lower) or
                                        (len(col_lower) > 3 and col_lower[:4] in col1_lower)):
                    col1_matched = col
                if not col2_matched and (col2_lower == col_lower or col2_lower in col_lower or col_lower in col2_lower or 
                                        (len(col2_lower) > 3 and col2_lower[:4] in col_lower) or
                                        (len(col_lower) > 3 and col_lower[:4] in col2_lower)):
                    col2_matched = col
            
            if col1_matched and col2_matched:
                return (col1_matched, col2_matched)
    
    return None


def format_statistics_context_for_llm(query: str, statistics_list: List[Dict[str, Any]]) -> str:
    """
    Format statistics (correlations, distributions, min/max) as context for LLM.
    Returns a formatted string with relevant statistical information.
    Statistics Agent function for providing correlation and distribution data to LLM.
    """
    from typing import List, Dict, Any
    
    if not statistics_list:
        return ""
    
    query_lower = query.lower()
    context_parts = []
    
    for stats in statistics_list:
        doc_name = stats.get("document_name", "Unknown Document")
        context_parts.append(f"\n## Statistical Analysis for {doc_name}\n")
        
        # Correlations
        if "correlation" in query_lower or "correlate" in query_lower or "relationship" in query_lower:
            correlations = stats.get("correlations", {})
            if correlations:
                # Get all available column names from correlations
                all_columns = set()
                for col1, corr_dict in correlations.items():
                    all_columns.add(col1)
                    for col2 in corr_dict.keys():
                        all_columns.add(col2)
                
                # Try to extract specific column pair from query
                specific_pair = extract_column_names_from_query(query, list(all_columns))
                specific_correlation = None
                
                if specific_pair:
                    col1, col2 = specific_pair
                    # Check both directions (correlation matrix is symmetric)
                    if col1 in correlations and col2 in correlations.get(col1, {}):
                        specific_correlation = (col1, col2, correlations[col1][col2])
                    elif col2 in correlations and col1 in correlations.get(col2, {}):
                        specific_correlation = (col2, col1, correlations[col2][col1])
                
                # If specific correlation found, show it prominently first
                if specific_correlation:
                    col1, col2, corr_value = specific_correlation
                    strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.5 else "weak"
                    direction = "positive" if corr_value > 0 else "negative"
                    context_parts.append("### Specific Correlation Requested:")
                    context_parts.append(f"**{col1} ↔ {col2}: {strength} {direction} correlation = {corr_value:.3f}**")
                    context_parts.append("")
                
                context_parts.append("### All Column Correlations:")
                strong_correlations = []
                for col1, corr_dict in correlations.items():
                    for col2, corr_value in corr_dict.items():
                        if col1 != col2:
                            # Include ALL correlations, not just strong ones
                            # But prioritize showing strong ones first
                            strong_correlations.append((col1, col2, corr_value))
                
                # Sort by absolute correlation value
                strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Show all correlations (not just top 20) if specific pair was requested
                limit = len(strong_correlations) if specific_correlation else 20
                for col1, col2, corr_value in strong_correlations[:limit]:
                    strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.5 else "weak"
                    direction = "positive" if corr_value > 0 else "negative"
                    # Highlight the specific correlation if it's in the list
                    marker = " ⭐" if specific_correlation and col1 == specific_correlation[0] and col2 == specific_correlation[1] else ""
                    context_parts.append(f"- {col1} ↔ {col2}: {strength} {direction} correlation ({corr_value:.3f}){marker}")
        
        # Distributions
        if "distribution" in query_lower or "distribute" in query_lower or "how many" in query_lower:
            descriptive_stats = stats.get("descriptive_stats", {})
            column_types = stats.get("column_types", {})
            
            context_parts.append("\n### Value Distributions:")
            for col, col_type in list(column_types.items())[:10]:  # Limit to 10 columns
                desc_stats = descriptive_stats.get(col, {})
                if col_type == "categorical" and "value_counts" in desc_stats:
                    value_counts = desc_stats["value_counts"]
                    top_values = list(value_counts.items())[:5]
                    if top_values:
                        context_parts.append(f"\n**{col}** (categorical):")
                        for value, count in top_values:
                            context_parts.append(f"  - {value}: {count} occurrences")
                elif col_type == "numeric" and desc_stats:
                    context_parts.append(f"\n**{col}** (numeric):")
                    if desc_stats.get("min") is not None:
                        context_parts.append(f"  - Min: {desc_stats['min']:.2f}")
                    if desc_stats.get("max") is not None:
                        context_parts.append(f"  - Max: {desc_stats['max']:.2f}")
                    if desc_stats.get("mean") is not None:
                        context_parts.append(f"  - Mean: {desc_stats['mean']:.2f}")
                    if desc_stats.get("median") is not None:
                        context_parts.append(f"  - Median: {desc_stats['median']:.2f}")
        
        # Min/Max values
        if any(keyword in query_lower for keyword in ["min", "minimum", "max", "maximum", "range", "lowest", "highest"]):
            descriptive_stats = stats.get("descriptive_stats", {})
            column_types = stats.get("column_types", {})
            
            context_parts.append("\n### Min/Max Values by Column:")
            for col, col_type in column_types.items():
                if col_type == "numeric":
                    desc_stats = descriptive_stats.get(col, {})
                    if desc_stats.get("min") is not None and desc_stats.get("max") is not None:
                        context_parts.append(f"- **{col}**: Min = {desc_stats['min']:.2f}, Max = {desc_stats['max']:.2f}, Range = {desc_stats['max'] - desc_stats['min']:.2f}")
        
        # General statistics summary
        if not any(keyword in query_lower for keyword in ["correlation", "distribution", "min", "max", "minimum", "maximum"]):
            context_parts.append("\n### Summary Statistics:")
            if stats.get("total_rows"):
                context_parts.append(f"- Total records: {stats['total_rows']}")
            if stats.get("total_columns"):
                context_parts.append(f"- Total columns: {stats['total_columns']}")
            
            # Show key numeric columns with their ranges
            descriptive_stats = stats.get("descriptive_stats", {})
            column_types = stats.get("column_types", {})
            numeric_cols = [col for col, ct in column_types.items() if ct == "numeric"][:5]
            if numeric_cols:
                context_parts.append("\nKey numeric columns:")
                for col in numeric_cols:
                    desc_stats = descriptive_stats.get(col, {})
                    if desc_stats.get("min") is not None and desc_stats.get("max") is not None:
                        context_parts.append(f"  - {col}: [{desc_stats['min']:.2f}, {desc_stats['max']:.2f}]")
    
    return "\n".join(context_parts) if context_parts else ""


def extract_statistical_facts(statistics: Dict[str, Any], document_name: str, document_id: str) -> int:
    """Extract facts directly from statistics (correlations, statistical summaries, etc.)"""
    if not statistics:
        return 0
    
    from knowledge import add_to_graph as kb_add_to_graph
    from datetime import datetime
    
    timestamp = datetime.now().isoformat()
    facts_count = 0
    
    try:
        # Extract facts about correlations
        correlations = statistics.get("correlations", {})
        if correlations:
            correlation_text = []
            correlation_text.append(f"Correlation Analysis for {document_name}:")
            correlation_text.append("")
            
            strong_correlations = []
            for col1, corr_dict in correlations.items():
                for col2, corr_value in corr_dict.items():
                    if col1 != col2 and abs(corr_value) > 0.5:  # Only strong correlations
                        strong_correlations.append((col1, col2, corr_value))
                        strength = "strong" if abs(corr_value) > 0.7 else "moderate"
                        direction = "positive" if corr_value > 0 else "negative"
                        correlation_text.append(f"{col1} has {strength} {direction} correlation ({corr_value:.3f}) with {col2}")
            
            if correlation_text:
                correlation_text.append("")
                correlation_text.append("These correlations indicate relationships between employee attributes that can inform HR decisions.")
                corr_facts = kb_add_to_graph(
                    "\n".join(correlation_text),
                    source_document=document_name,
                    uploaded_at=timestamp,
                    agent_id=KG_AGENT_ID
                )
                # Count facts from correlation text
                facts_count += len([line for line in correlation_text if 'correlation' in line.lower()])
        
        # Extract facts about statistical summaries
        descriptive_stats = statistics.get("descriptive_stats", {})
        column_types = statistics.get("column_types", {})
        
        if descriptive_stats and column_types:
            stats_text = []
            stats_text.append(f"Statistical Summary for {document_name}:")
            stats_text.append("")
            
            # Numeric column statistics
            numeric_cols = [col for col, col_type in column_types.items() if col_type == "numeric"]
            if numeric_cols:
                stats_text.append("Numeric Column Statistics:")
                for col in numeric_cols[:15]:  # Top 15 numeric columns
                    stats = descriptive_stats.get(col, {})
                    if stats and stats.get("mean") is not None:
                        mean_val = stats.get("mean", 0)
                        median_val = stats.get("median", 0)
                        min_val = stats.get("min", 0)
                        max_val = stats.get("max", 0)
                        std_val = stats.get("std", 0)
                        
                        stats_text.append(f"{col} has mean {mean_val:.2f}, median {median_val:.2f}, range from {min_val:.2f} to {max_val:.2f}, standard deviation {std_val:.2f}")
                        stats_text.append(f"{col} average value is {mean_val:.2f}")
                        stats_text.append(f"{col} median value is {median_val:.2f}")
                        if max_val - min_val > 0:
                            stats_text.append(f"{col} ranges from {min_val:.2f} to {max_val:.2f}")
                stats_text.append("")
            
            # Categorical column distributions
            categorical_cols = [col for col, col_type in column_types.items() if col_type == "categorical"]
            if categorical_cols:
                stats_text.append("Categorical Column Distributions:")
                for col in categorical_cols[:15]:  # Top 15 categorical columns
                    stats = descriptive_stats.get(col, {})
                    value_counts = stats.get("value_counts", {})
                    if value_counts:
                        top_values = list(value_counts.items())[:5]
                        if top_values:
                            top_val, top_count = top_values[0]
                            total_count = sum(count for _, count in value_counts.items())
                            percentage = (top_count / total_count * 100) if total_count > 0 else 0
                            stats_text.append(f"{col} most common value is {top_val} with {percentage:.1f}% of records")
                            stats_text.append(f"{col} distribution shows {top_val} as the top category")
                stats_text.append("")
            
            # Data quality insights
            missing_values = statistics.get("missing_values", {})
            if missing_values:
                missing_cols = [col for col, count in missing_values.items() if count > 0]
                if missing_cols:
                    stats_text.append("Data Quality:")
                    for col in missing_cols[:10]:
                        missing_count = missing_values[col]
                        total_rows = statistics.get("total_rows", 0)
                        if total_rows > 0:
                            completeness = ((total_rows - missing_count) / total_rows * 100)
                            stats_text.append(f"{col} has {missing_count} missing values, completeness {completeness:.1f}%")
                    stats_text.append("")
            
            if len(stats_text) > 2:  # More than just header
                stats_facts = kb_add_to_graph(
                    "\n".join(stats_text),
                    source_document=document_name,
                    uploaded_at=timestamp,
                    agent_id=KG_AGENT_ID
                )
                # Count facts from statistics text
                facts_count += len([line for line in stats_text if ':' in line or 'is' in line.lower() or 'has' in line.lower()])
        
        return facts_count
        
    except Exception as e:
        print(f"⚠️  Error in extract_statistical_facts: {e}")
        import traceback
        traceback.print_exc()
        return 0

def process_with_kg_agent(extracted_text: str, document_name: str, document_id: str) -> Dict[str, Any]:
    """Process document with Knowledge Graph Agent"""
    kg_agent = agents_store.get(KG_AGENT_ID)
    if not kg_agent:
        return {"facts_extracted": 0, "result": "KG agent not found"}
    
    kg_agent.status = "processing"
    
    try:
        facts_before = len(kb_graph)
        timestamp = datetime.now().isoformat()
        
        # Process text with knowledge extraction pipeline
        result = kb_add_to_graph(extracted_text, source_document=document_name, uploaded_at=timestamp, agent_id=KG_AGENT_ID)
        
        facts_after = len(kb_graph)
        facts_extracted = facts_after - facts_before
        
        kg_agent.status = "active"
        print(f"✅ KG Agent extracted {facts_extracted} facts from {document_name}")
        
        return {
            "facts_extracted": facts_extracted,
            "result": result
        }
        
    except Exception as e:
        print(f"❌ KG Agent error: {e}")
        import traceback
        traceback.print_exc()
        kg_agent.status = "active"
        return {"facts_extracted": 0, "result": f"Error: {str(e)}"}

# ============================================================================
# AGENT INFORMATION
# ============================================================================

def get_all_agents() -> Dict[str, Any]:
    """Get all agents (core + document agents)"""
    all_agents = {}
    
    # Add core agents
    for agent_id, agent in agents_store.items():
        all_agents[agent_id] = asdict(agent)
    
    # Add document agents (exclude any non-serializable data)
    for agent_id, agent in document_agents.items():
        agent_dict = asdict(agent)
        all_agents[agent_id] = agent_dict
    
    return all_agents

def get_agent_architecture() -> Dict[str, Any]:
    """Get agent architecture for visualization"""
    architecture = {
        "orchestrator_agents": [],
        "statistics_agents": [],
        "visualization_agents": [],
        "kg_agents": [],
        "llm_agents": [],
        "strategic_query_agents": [],
        "operational_query_agents": [],
        "document_agents": []
    }
    
    # Add core agents
    for agent_id, agent in agents_store.items():
        agent_dict = asdict(agent)
        if agent.type == "orchestrator":
            architecture["orchestrator_agents"].append(agent_dict)
        elif agent.type == "statistics":
            architecture["statistics_agents"].append(agent_dict)
        elif agent.type == "visualization":
            architecture["visualization_agents"].append(agent_dict)
        elif agent.type == "kg":
            architecture["kg_agents"].append(agent_dict)
        elif agent.type == "llm":
            architecture["llm_agents"].append(agent_dict)
        elif agent.type == "strategic_query":
            architecture["strategic_query_agents"].append(agent_dict)
        elif agent.type == "operational_query":
            architecture["operational_query_agents"].append(agent_dict)
    
    # Add document agents (including worker agents)
    for agent_id, agent in document_agents.items():
        agent_dict = asdict(agent)
        # Separate main document agents from worker agents
        if agent.type == "document_worker":
            # Worker agents are included in document_agents but can be filtered if needed
            architecture["document_agents"].append(agent_dict)
        else:
            architecture["document_agents"].append(agent_dict)
    
    return architecture

def get_agent_by_id(agent_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific agent by ID"""
    if agent_id in agents_store:
        return asdict(agents_store[agent_id])
    if agent_id in document_agents:
        return asdict(document_agents[agent_id])
    return None

def get_document_statistics(document_id: str) -> Optional[Dict[str, Any]]:
    """Get statistics for a specific document"""
    doc_agent_id = f"doc_{document_id}"
    doc_agent = document_agents.get(doc_agent_id)
    if not doc_agent:
        return None
    
    # Statistics are stored in document metadata or retrieved from Statistics Agent
    return doc_agent.metadata.get("statistics")

def get_document_visualizations(document_id: str) -> Optional[Dict[str, Any]]:
    """Get visualizations for a specific document"""
    doc_agent_id = f"doc_{document_id}"
    doc_agent = document_agents.get(doc_agent_id)
    if not doc_agent:
        return None
    
    return doc_agent.metadata.get("visualizations")

def summarize_document(document_id: str, document_name: str) -> Optional[str]:
    """Generate a summary of a document using the LLM agent"""
    llm_agent = agents_store.get(LLM_AGENT_ID)
    if not llm_agent:
        return None
    
    try:
        # Get document agent to access statistics and metadata
        doc_agent_id = f"doc_{document_id}"
        doc_agent = document_agents.get(doc_agent_id)
        
        # Get statistics if available
        statistics = None
        if doc_agent:
            statistics = doc_agent.metadata.get("statistics")
        
        # Retrieve facts related to this document
        from knowledge import retrieve_context
        from responses import generate_llm_response, USE_LLM, USE_OLLAMA, USE_OPENAI, OPENAI_API_KEY
        
        # Build comprehensive context for summary
        context_parts = []
        
        # Add document-specific facts from KG
        query = f"employees data human resources {document_name}"
        kg_context = retrieve_context(query)
        if kg_context and "No directly relevant facts found" not in kg_context:
            # Filter to facts from this document
            doc_facts = []
            for line in kg_context.split('\n'):
                if document_name.lower() in line.lower() or document_id.lower() in line.lower():
                    doc_facts.append(line)
            if doc_facts:
                context_parts.append("**Knowledge Graph Facts:**")
                context_parts.extend(doc_facts[:10])  # Top 10 facts
        
        # Add statistics information
        if statistics:
            context_parts.append("\n**Statistical Analysis:**")
            if statistics.get("total_rows"):
                context_parts.append(f"- Total records: {statistics.get('total_rows')}")
            if statistics.get("total_columns"):
                context_parts.append(f"- Number of attributes: {statistics.get('total_columns')}")
            if statistics.get("column_types"):
                col_types = statistics.get("column_types", {})
                numeric_count = sum(1 for t in col_types.values() if t == "numeric")
                categorical_count = sum(1 for t in col_types.values() if t == "categorical")
                if numeric_count > 0:
                    context_parts.append(f"- Numeric attributes: {numeric_count}")
                if categorical_count > 0:
                    context_parts.append(f"- Categorical attributes: {categorical_count}")
            
            # Add key column names
            if statistics.get("column_types"):
                key_columns = list(statistics.get("column_types", {}).keys())[:10]
                context_parts.append(f"- Key attributes: {', '.join(key_columns)}")
            
            # Add descriptive statistics highlights
            if statistics.get("descriptive_stats"):
                desc_stats = statistics.get("descriptive_stats", {})
                for col, stats in list(desc_stats.items())[:3]:
                    if isinstance(stats, dict):
                        if "value_counts" in stats:
                            top_values = list(stats["value_counts"].items())[:3]
                            if top_values:
                                context_parts.append(f"- {col}: Top values include {', '.join([str(v) for v, _ in top_values])}")
        
        # Build context text
        context_text = '\n'.join(context_parts) if context_parts else ""
        
        # Determine document type and build appropriate prompt
        document_type = "CSV file"
        if doc_agent:
            doc_type = doc_agent.document_type
            if doc_type == '.csv':
                document_type = "CSV dataset"
            elif doc_type == '.pdf':
                document_type = "PDF document"
            elif doc_type == '.docx':
                document_type = "Word document"
        
        # Build enhanced prompt for HR data
        prompt = f"""Please provide a concise summary (2-4 sentences) of the document '{document_name}' which is a {document_type} containing human resources and employee data.

This document contains information about employees, their attributes, and organizational data. Based on the following information, summarize:
1. What type of HR/employee data this contains
2. Key characteristics or patterns in the data
3. What insights can be drawn about the workforce

Information available:
{context_text if context_text else "Limited information available - this appears to be an HR/employee dataset."}

Provide a clear, informative summary focusing on the employee and HR aspects of the data."""

        # Use LLM to generate summary if available
        use_llm = USE_OLLAMA or (USE_OPENAI and OPENAI_API_KEY)
        if use_llm and USE_LLM:
            summary = generate_llm_response(prompt, context_text if context_text else f"Document: {document_name}, Type: {document_type}")
            if summary and len(summary.strip()) > 10:
                return summary.strip()
        
        # Fallback: Generate summary from statistics
        if statistics:
            summary_parts = []
            summary_parts.append(f"This document contains human resources and employee data with {statistics.get('total_rows', 0)} employee records.")
            
            if statistics.get("column_types"):
                col_types = statistics.get("column_types", {})
                key_cols = list(col_types.keys())[:8]
                if key_cols:
                    summary_parts.append(f"The dataset includes attributes such as: {', '.join(key_cols)}.")
                
                # Add insights about data types
                numeric_cols = [col for col, t in col_types.items() if t == "numeric"]
                categorical_cols = [col for col, t in col_types.items() if t == "categorical"]
                
                if numeric_cols:
                    summary_parts.append(f"It contains {len(numeric_cols)} numeric attributes (likely including salary, age, tenure, etc.).")
                if categorical_cols:
                    summary_parts.append(f"It includes {len(categorical_cols)} categorical attributes (such as department, role, location, etc.).")
            
            # Add data quality note
            if statistics.get("missing_values"):
                missing_cols = [col for col, count in statistics.get("missing_values", {}).items() if count > 0]
                if missing_cols:
                    summary_parts.append(f"Note: Some attributes have missing values, indicating incomplete records.")
            
            return " ".join(summary_parts)
        
        # Final fallback
        if doc_agent and doc_agent.facts_extracted > 0:
            return f"This document '{document_name}' contains human resources and employee data. {doc_agent.facts_extracted} facts have been extracted from the dataset, covering employee attributes, organizational information, and workforce characteristics."
        
        return f"This document '{document_name}' is a {document_type} containing human resources and employee data. The document has been processed and is ready for analysis."
        
    except Exception as e:
        print(f"⚠️  Error generating document summary: {e}")
        import traceback
        traceback.print_exc()
        return None
