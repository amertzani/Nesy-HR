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
from typing import Dict, List, Optional, Any
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
            name="LLM Research Assistant",
            type="llm",
            status="active",
            created_at=llm_agent_data.get('created_at', datetime.now().isoformat()),
            metadata=llm_agent_data.get('metadata', {
                "description": "Provides traceable answers using KG and statistics",
                "role": "research_assistant",
                "capabilities": ["qa", "insights", "traceability"]
            })
        )
        print(f"✅ Restored LLM agent")
    else:
        agents_store[LLM_AGENT_ID] = Agent(
            id=LLM_AGENT_ID,
            name="LLM Research Assistant",
            type="llm",
            status="active",
            created_at=datetime.now().isoformat(),
            metadata={
                "description": "Provides traceable answers using KG and statistics",
                "role": "research_assistant",
                "capabilities": ["qa", "insights", "traceability"]
            }
        )
    
    save_agents()
    print(f"✅ Initialized {len(agents_store)} core agents:")
    print(f"   - Statistics Agent")
    print(f"   - Visualization Agent")
    print(f"   - Knowledge Graph Agent")
    print(f"   - LLM Research Assistant")
    print(f"   Document agents will be created when documents are uploaded")

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def create_document_agent(document_name: str, document_id: str, document_type: str) -> str:
    """Create a document-specific agent for processing a document"""
    agent_id = f"doc_{document_id}"
    
    if agent_id in document_agents:
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
    doc_agent_id = create_document_agent(document_name, document_id, document_type)
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
    
    # Step 1: Statistics Agent - Analyze document (especially for CSV files)
    statistics_result = None
    if document_type.lower() == '.csv':
        try:
            statistics_result = process_with_statistics_agent(file_path, document_name)
            results["statistics"] = statistics_result
            
            # Step 2: Visualization Agent - Create visualizations from statistics
            if statistics_result:
                try:
                    visualizations = process_with_visualization_agent(statistics_result, document_name)
                    results["visualizations"] = visualizations
                except Exception as viz_error:
                    print(f"⚠️  Visualization agent error: {viz_error}")
                    import traceback
                    traceback.print_exc()
        except Exception as stats_error:
            print(f"⚠️  Statistics agent error: {stats_error}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Knowledge Graph Agent - Extract facts from document
    # For CSV files, use provided text or extract comprehensive sample
    kg_text = extracted_text
    if document_type.lower() == '.csv' and (not kg_text or len(kg_text.strip()) < 10):
        # If CSV and no text provided, extract comprehensive sample from CSV for KG agent
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
            
            # Read more rows for better extraction (increased from 100 to 200)
            df_sample = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python', nrows=200)
            if len(df_sample.columns) == 1:
                df_sample = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python', nrows=200)
            
            # Build comprehensive text representation for KG extraction
            text_lines = []
            
            # Header: Dataset overview
            text_lines.append(f"Human Resources Employee Dataset: {document_name}")
            text_lines.append(f"Total records: {len(df_sample)} sample rows from dataset")
            text_lines.append(f"Total attributes: {len(df_sample.columns)} columns")
            text_lines.append("")
            
            # Column descriptions
            text_lines.append("Dataset Attributes:")
            for col in df_sample.columns:
                col_type = "numeric" if pd.api.types.is_numeric_dtype(df_sample[col]) else "categorical"
                unique_count = df_sample[col].nunique()
                text_lines.append(f"- {col} ({col_type}): {unique_count} unique values")
            text_lines.append("")
            
            # Statistical summaries for numeric columns
            numeric_cols = [col for col in df_sample.columns if pd.api.types.is_numeric_dtype(df_sample[col])]
            if numeric_cols:
                text_lines.append("Numeric Attribute Statistics:")
                for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                    stats = df_sample[col].describe()
                    text_lines.append(f"- {col}: mean {stats['mean']:.2f}, median {stats['50%']:.2f}, range [{stats['min']:.2f}, {stats['max']:.2f}]")
                text_lines.append("")
            
            # Categorical value distributions
            categorical_cols = [col for col in df_sample.columns if not pd.api.types.is_numeric_dtype(df_sample[col])]
            if categorical_cols:
                text_lines.append("Categorical Attribute Distributions:")
                for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
                    top_values = df_sample[col].value_counts().head(5)
                    value_list = ", ".join([f"{val} ({count})" for val, count in top_values.items()])
                    text_lines.append(f"- {col}: top values are {value_list}")
                text_lines.append("")
            
            # Sample records with structured format for better extraction
            text_lines.append("Sample Employee Records:")
            
            # Find employee name column
            name_col = None
            for col in df_sample.columns:
                if any(keyword in col.lower() for keyword in ['name', 'employee', 'empname', 'employee_name']):
                    name_col = col
                    break
            
            try:
                for idx, row in df_sample.head(100).iterrows():  # Increased from 50 to 100 rows
                    # Extract employee name if available
                    employee_name = None
                    if name_col and name_col in df_sample.columns:
                        try:
                            name_val = row[name_col]
                            if pd.notna(name_val):
                                employee_name = str(name_val).strip()
                        except (KeyError, IndexError):
                            pass
                    
                    # Create structured fact-like sentences with employee name as subject
                    if employee_name:
                        # Format: "Employee Name has attribute value"
                        fact_sentences = []
                        for col, val in row.items():
                            try:
                                if pd.notna(val) and col != name_col:
                                    col_lower = str(col).lower()
                                    val_str = str(val).strip()
                                    
                                    # Format based on attribute type
                                    if 'salary' in col_lower:
                                        fact_sentences.append(f"{employee_name} has salary {val_str}")
                                        fact_sentences.append(f"{employee_name} salary is {val_str}")
                                    elif 'position' in col_lower:
                                        fact_sentences.append(f"{employee_name} has position {val_str}")
                                        fact_sentences.append(f"{employee_name} position is {val_str}")
                                    elif 'department' in col_lower or 'dept' in col_lower:
                                        fact_sentences.append(f"{employee_name} works in department {val_str}")
                                        fact_sentences.append(f"{employee_name} department is {val_str}")
                                    elif 'age' in col_lower:
                                        fact_sentences.append(f"{employee_name} has age {val_str}")
                                        fact_sentences.append(f"{employee_name} age is {val_str}")
                                    else:
                                        # Generic format
                                        fact_sentences.append(f"{employee_name} has {col} {val_str}")
                                        fact_sentences.append(f"{employee_name} {col} is {val_str}")
                            except Exception:
                                continue  # Skip problematic columns
                        
                        if fact_sentences:
                            text_lines.append(f"Record {idx + 1}: {' | '.join(fact_sentences[:10])}")  # Limit to 10 facts per record
                    else:
                        # Fallback: original format
                        record_parts = []
                        for col, val in row.items():
                            try:
                                if pd.notna(val):
                                    col_lower = str(col).lower()
                                    if col_lower in ['name', 'employee', 'id', 'employeeid']:
                                        record_parts.append(f"Employee {val}")
                                    else:
                                        record_parts.append(f"{col} is {val}")
                            except Exception:
                                continue  # Skip problematic columns
                        
                        if record_parts:
                            text_lines.append(f"Record {idx + 1}: {' | '.join(record_parts)}")
            except Exception as e:
                print(f"⚠️  Error processing employee records: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simple format
                try:
                    for idx, row in df_sample.head(50).iterrows():
                        try:
                            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                            if row_text:
                                text_lines.append(f"Record {idx + 1}: {row_text}")
                        except Exception:
                            continue
                except Exception as fallback_error:
                    print(f"⚠️  Fallback format also failed: {fallback_error}")
                    # Last resort: just add column names
                    if 'text_lines' in locals():
                        text_lines.append(f"Columns: {', '.join(df_sample.columns.tolist())}")
        
            # Add correlation information if statistics are available
            if statistics_result and statistics_result.get("correlations"):
                try:
                    correlations = statistics_result["correlations"]
                    text_lines.append("")
                    text_lines.append("Correlation Analysis:")
                    strong_corrs = []
                    for col1, corr_dict in correlations.items():
                        for col2, corr_value in corr_dict.items():
                            if col1 != col2 and abs(corr_value) > 0.5:
                                strong_corrs.append((col1, col2, corr_value))
                    
                    if strong_corrs:
                        text_lines.append(f"Found {len(strong_corrs)} strong correlations (|r| > 0.5):")
                        for col1, col2, corr_value in strong_corrs[:10]:  # Top 10 correlations
                            strength = "strong" if abs(corr_value) > 0.7 else "moderate"
                            direction = "positive" if corr_value > 0 else "negative"
                            text_lines.append(f"- {col1} has {strength} {direction} correlation ({corr_value:.3f}) with {col2}")
                            text_lines.append(f"- {col1} correlates with {col2} with coefficient {corr_value:.3f}")
                    text_lines.append("")
                except Exception as corr_error:
                    print(f"⚠️  Error adding correlation info: {corr_error}")
            
            # Add summary insights
            try:
                text_lines.append("")
                text_lines.append("Dataset Insights:")
                if 'numeric_cols' in locals() and numeric_cols:
                    text_lines.append(f"- Contains {len(numeric_cols)} numeric attributes including salary, age, tenure, and performance metrics")
                if 'categorical_cols' in locals() and categorical_cols:
                    text_lines.append(f"- Contains {len(categorical_cols)} categorical attributes including department, role, location, and status")
                text_lines.append(f"- Dataset represents employee information and organizational data")
            except Exception as insight_error:
                print(f"⚠️  Error adding insights: {insight_error}")
            
            kg_text = "\n".join(text_lines)
            if 'df_sample' in locals():
                print(f"📊 Extracted comprehensive text from CSV for KG agent ({len(df_sample)} rows, {len(df_sample.columns)} columns)")
        except Exception as csv_text_error:
            print(f"⚠️  Could not extract CSV text for KG: {csv_text_error}")
            import traceback
            traceback.print_exc()
            # Minimal fallback
            try:
                if 'df_sample' in locals():
                    kg_text = f"CSV Dataset: {document_name}\nColumns: {', '.join(df_sample.columns.tolist())}"
                else:
                    kg_text = f"CSV Dataset: {document_name}"
            except:
                kg_text = f"CSV Dataset: {document_name}"
    
    # Step 3a: For CSV files, extract facts directly from DataFrame using parallel processing
    csv_facts_count = 0
    if document_type.lower() == '.csv':
        try:
            # Use parallel processing for large files, sequential for small ones
            import os
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            file_size_mb = file_size / (1024 * 1024)
            
            # Use parallel processing for files > 1MB or if explicitly requested
            use_parallel = file_size_mb > 1.0
            
            if use_parallel:
                # Determine optimal number of workers (4-8 workers, based on file size)
                num_workers = min(8, max(4, int(file_size_mb / 2)))
                print(f"🔄 Extracting facts using parallel processing ({num_workers} workers) from CSV file...")
                csv_facts_count = extract_csv_facts_directly_parallel(file_path, document_name, document_id, num_workers=num_workers)
            else:
                print(f"🔄 Extracting facts directly from CSV file (sequential, small file)...")
                csv_facts_count = extract_csv_facts_directly(file_path, document_name, document_id)
            
            print(f"✅ CSV extraction completed: {csv_facts_count} facts")
        except Exception as csv_facts_error:
            print(f"⚠️  Error in CSV fact extraction: {csv_facts_error}")
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
    
    # Step 3c: Also process text for any additional facts (complementary, not primary for CSV)
    kg_result_facts = 0
    if kg_text and len(kg_text.strip()) > 10:
        try:
            kg_result = process_with_kg_agent(kg_text, document_name, document_id)
            kg_result_facts = kg_result.get("facts_extracted", 0)
            results["kg_result"] = kg_result.get("result", "")
        except Exception as kg_error:
            print(f"⚠️  KG agent error: {kg_error}")
            import traceback
            traceback.print_exc()
    
    # Total facts = direct CSV facts + statistical facts + text extraction facts
    results["facts_extracted"] = csv_facts_count + stats_facts_count + kg_result_facts
    print(f"📊 Total facts extracted: {results['facts_extracted']} (CSV: {csv_facts_count}, Stats: {stats_facts_count}, Text: {kg_result_facts})")
    
    doc_agent.status = "completed"
    return results

def process_with_statistics_agent(file_path: str, document_name: str) -> Optional[Dict[str, Any]]:
    """Process document with Statistics Agent"""
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
            numeric_cols = [col for col in df.columns if analysis["column_types"][col] == "numeric"]
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                analysis["correlations"] = {
                    col: {
                        other_col: float(corr_matrix.loc[col, other_col])
                        for other_col in numeric_cols
                        if col != other_col and not pd.isna(corr_matrix.loc[col, other_col])
                    }
                    for col in numeric_cols
                }
            
            stats_agent.status = "active"
            print(f"✅ Statistics Agent analyzed {document_name}: {len(df)} rows, {len(df.columns)} columns")
            return analysis
            
        except ImportError:
            print("⚠️  pandas not available, skipping statistical analysis")
            stats_agent.status = "active"
            return None
        except Exception as e:
            print(f"⚠️  Statistics analysis error: {e}")
            import traceback
            traceback.print_exc()
            stats_agent.status = "active"
            return None
    
    except Exception as e:
        print(f"❌ Statistics Agent error: {e}")
        stats_agent.status = "active"
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

def extract_csv_facts_directly_parallel(file_path: str, document_name: str, document_id: str, 
                                        num_workers: int = 4, chunk_size: Optional[int] = None) -> int:
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
                calculated_chunk_size = max(50, min(200, target_data_points_per_chunk // total_cols))
            else:
                calculated_chunk_size = 100
            
            # Ensure we have enough chunks for parallelization
            # Minimum: num_workers chunks, maximum: reasonable chunk size
            min_chunks_needed = num_workers
            max_chunk_size_for_parallelization = max(50, total_rows // min_chunks_needed)
            
            # Use the smaller of: calculated size or max for parallelization
            chunk_size = min(calculated_chunk_size, max_chunk_size_for_parallelization)
            
            # Ensure minimum chunk size for efficiency
            chunk_size = max(50, chunk_size)
            
            print(f"📊 Data complexity: {data_complexity:,} data points ({total_rows:,} rows × {total_cols} cols)")
            print(f"📊 Optimal chunk size: {chunk_size} rows (targeting ~{chunk_size * total_cols:,} data points per chunk)")
        
        # Split into chunks
        chunks = []
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunks.append((start_idx, end_idx))
        
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
            
            try:
                # First pass: track all columns in this chunk
                for col in chunk_df.columns:
                    chunk_columns_processed.add(col)
                
                for idx, row in chunk_df.iterrows():
                    # Extract employee name
                    employee_name = None
                    if name_col and name_col in df.columns:
                        name_val = row[name_col]
                        if pd.notna(name_val):
                            employee_name = str(name_val).strip()
                    
                    if not employee_name:
                        continue
                    
                    # Track employee name for orchestrator routing
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
                        
                        # Create predicate - handle all columns including absences, etc.
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
                        
                        # Check if fact already exists (thread-safe)
                        with facts_lock:
                            fact_exists_check = fact_exists(employee_name, predicate, val_str)
                        
                        if not fact_exists_check:
                            # Add fact directly to graph (thread-safe)
                            with facts_lock:
                                subject_clean = employee_name.strip().replace(' ', '_')
                                predicate_clean = predicate.strip().replace(' ', '_')
                                object_clean = val_str.strip()
                                
                                subject_uri = rdflib.URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
                                predicate_uri = rdflib.URIRef(f"urn:predicate:{quote(predicate_clean, safe='')}")
                                object_literal = rdflib.Literal(object_clean)
                                
                                graph.add((subject_uri, predicate_uri, object_literal))
                                add_fact_source_document(employee_name, predicate, val_str, document_name, uploaded_at)
                                
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
                
                # Extract employee name
                employee_name = None
                if name_col and name_col in df.columns:
                    name_val = row[name_col]
                    if pd.notna(name_val):
                        employee_name = str(name_val).strip()
                
                # If no employee name, skip or use row index
                if not employee_name:
                    continue
                
                # Create facts for each column
                for col, val in row.items():
                    if pd.isna(val) or col == name_col:
                        continue
                    
                    col_lower = str(col).lower()
                    val_str = str(val).strip()
                    
                    # Create predicate based on column type
                    if 'salary' in col_lower:
                        predicate = "has salary"
                    elif 'position' in col_lower:
                        predicate = "has position"
                    elif 'department' in col_lower or 'dept' in col_lower:
                        predicate = "works in department"
                    elif 'age' in col_lower:
                        predicate = "has age"
                    else:
                        predicate = f"has {col}"
                    
                    # Check if fact already exists
                    if not fact_exists(employee_name, predicate, val_str):
                        # Add fact directly to graph
                        from knowledge import graph
                        subject_clean = employee_name.strip().replace(' ', '_')
                        predicate_clean = predicate.strip().replace(' ', '_')
                        object_clean = val_str.strip()
                        
                        subject_uri = rdflib.URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
                        predicate_uri = rdflib.URIRef(f"urn:predicate:{quote(predicate_clean, safe='')}")
                        object_literal = rdflib.Literal(object_clean)
                        
                        graph.add((subject_uri, predicate_uri, object_literal))
                        
                        # Add source document
                        add_fact_source_document(employee_name, predicate, val_str, document_name, uploaded_at)
                        
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
