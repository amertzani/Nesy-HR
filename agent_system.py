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

# ============================================================================
# GLOBAL STATE
# ============================================================================

AGENTS_FILE = "agents_store.json"

# Core agent IDs
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
            for idx, row in df_sample.head(100).iterrows():  # Increased from 50 to 100 rows
                # Create structured fact-like sentences
                record_parts = []
                for col, val in row.items():
                    if pd.notna(val):
                        # Format as "Employee has [attribute] [value]" for better extraction
                        if col.lower() in ['name', 'employee', 'id', 'employeeid']:
                            record_parts.append(f"Employee {val}")
                        else:
                            record_parts.append(f"{col} is {val}")
                
                if record_parts:
                    text_lines.append(f"Record {idx + 1}: {' | '.join(record_parts)}")
            
            # Add correlation information if statistics are available
            if statistics_result and statistics_result.get("correlations"):
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
            
            # Add summary insights
            text_lines.append("")
            text_lines.append("Dataset Insights:")
            if numeric_cols:
                text_lines.append(f"- Contains {len(numeric_cols)} numeric attributes including salary, age, tenure, and performance metrics")
            if categorical_cols:
                text_lines.append(f"- Contains {len(categorical_cols)} categorical attributes including department, role, location, and status")
            text_lines.append(f"- Dataset represents employee information and organizational data")
            
            kg_text = "\n".join(text_lines)
            print(f"📊 Extracted comprehensive text from CSV for KG agent ({len(df_sample)} rows, {len(df_sample.columns)} columns)")
        except Exception as csv_text_error:
            print(f"⚠️  Could not extract CSV text for KG: {csv_text_error}")
            import traceback
            traceback.print_exc()
            kg_text = ""
    
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
    
    try:
        kg_result = process_with_kg_agent(kg_text or "", document_name, document_id)
        results["facts_extracted"] = kg_result.get("facts_extracted", 0) + stats_facts_count
        results["kg_result"] = kg_result.get("result", "")
    except Exception as kg_error:
        print(f"⚠️  KG agent error: {kg_error}")
        import traceback
        traceback.print_exc()
    
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
        "statistics_agents": [],
        "visualization_agents": [],
        "kg_agents": [],
        "llm_agents": [],
        "document_agents": []
    }
    
    # Add core agents
    for agent_id, agent in agents_store.items():
        agent_dict = asdict(agent)
        if agent.type == "statistics":
            architecture["statistics_agents"].append(agent_dict)
        elif agent.type == "visualization":
            architecture["visualization_agents"].append(agent_dict)
        elif agent.type == "kg":
            architecture["kg_agents"].append(agent_dict)
        elif agent.type == "llm":
            architecture["llm_agents"].append(agent_dict)
    
    # Add document agents
    for agent_id, agent in document_agents.items():
        agent_dict = asdict(agent)
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
