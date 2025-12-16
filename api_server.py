"""
FastAPI Backend Server
=====================

Main API server for the Knowledge Graph system.
Provides REST endpoints for document upload, knowledge base access, chat, and insights.

Run with: uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import system modules
try:
    from knowledge import graph, load_knowledge_graph
    from documents_store import add_document, get_all_documents, delete_document
    from operational_queries import compute_operational_insights
    from strategic_queries import find_csv_file_path, load_csv_data
    
    # Simple helper functions if not available
    def save_knowledge_graph():
        import pickle
        with open("knowledge_graph.pkl", "wb") as f:
            pickle.dump(graph, f)
    
    def get_fact_source_document(subject, predicate, obj):
        # Simple implementation - return empty list
        return []
    
    def get_fact_details(subject, predicate, obj):
        return None
    
    def add_to_graph(text: str, source_document: str = "manual", agent_id: str = None):
        from urllib.parse import quote
        from rdflib import URIRef, Literal
        # Simple fact extraction
        parts = text.split()
        if len(parts) >= 3:
            subject = " ".join(parts[:-2])
            predicate = parts[-2]
            obj = parts[-1]
            s = URIRef(f"urn:{quote(subject, safe='')}")
            p = URIRef(f"urn:{quote(predicate, safe='')}")
            graph.add((s, p, Literal(obj)))
            save_knowledge_graph()
    
    KG_AVAILABLE = True
    
    # Optional imports
    try:
        from agent_system import process_document_with_agents
        AGENT_SYSTEM_AVAILABLE = True
    except ImportError:
        AGENT_SYSTEM_AVAILABLE = False
        print("⚠️  agent_system not available, document processing will be limited")
    
    try:
        from responses import respond
        RESPONSES_AVAILABLE = True
    except ImportError:
        RESPONSES_AVAILABLE = False
        print("⚠️  responses module not available, chat will use simple responses")
        
except ImportError as e:
    print(f"⚠️  Critical import error: {e}")
    KG_AVAILABLE = False
    AGENT_SYSTEM_AVAILABLE = False
    RESPONSES_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Graph API",
    description="API for knowledge graph management and HR analytics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load knowledge graph on startup
@app.on_event("startup")
async def startup_event():
    """Load knowledge graph when server starts"""
    if KG_AVAILABLE:
        try:
            load_result = load_knowledge_graph()
            if load_result:
                print(f"✅ {load_result}")
        except Exception as e:
            print(f"⚠️  Error loading knowledge graph: {e}")


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []


class FactCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    source_document: Optional[str] = "manual"


class FactDelete(BaseModel):
    subject: str
    predicate: str
    object: str


# Health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "Knowledge Graph API is running"}


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_graph_available": KG_AVAILABLE,
        "facts_count": len(graph) if KG_AVAILABLE and graph else 0
    }


# Document Upload
@app.post("/api/knowledge/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents (CSV, PDF, DOCX, TXT).
    Files are processed by the agent system and facts are extracted.
    """
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph system not available")
    
    results = []
    
    for file in files:
        try:
            # Save uploaded file to temp directory
            file_ext = Path(file.filename).suffix.lower()
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, file.filename)
            
            # Write file
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            print(f"📄 Processing uploaded file: {file.filename} ({len(content)} bytes)")
            
            # Process document with agents
            document_id = f"doc_{file.filename}"
            facts_count = 0
            
            if AGENT_SYSTEM_AVAILABLE:
                # Full multi-agent processing pipeline (if available)
                result = process_document_with_agents(
                    document_id=document_id,
                    document_name=file.filename,
                    document_type=file_ext,
                    file_path=temp_path
                )
                facts_count = result.get("facts_count", 0) if isinstance(result, dict) else 0
            else:
                # Lightweight path: do NOT generate per-cell facts when the full
                # agent system is unavailable. This avoids huge slowdowns and
                # keeps the UI responsive. Statistics and operational insights
                # are computed directly from the CSV using pandas, so they
                # remain fully functional.
                print(
                    f\"ℹ️  Skipping heavy per-cell fact extraction for {file.filename} "
                    \"(agent_system not available). Operational insights and statistics "
                    \"will still be computed from the CSV.\"
                )
                result = {\"facts_count\": 0, \"status\": \"processed_light\"}
            
            # Add to documents store with facts count
            add_document(
                name=file.filename,
                file_type=file_ext,
                file_path=temp_path,
                size=len(content)
            )
            
            # Update facts_extracted in document store
            try:
                import json
                store_file = "documents_store.json"
                if os.path.exists(store_file):
                    with open(store_file, 'r') as f:
                        data = json.load(f)
                    # Find and update the document
                    for doc in data.get("documents", []):
                        if doc.get("name") == file.filename:
                            doc["facts_extracted"] = facts_count
                            break
                    with open(store_file, 'w') as f:
                        json.dump(data, f, indent=2)
            except Exception as e:
                print(f"⚠️  Could not update facts count in store: {e}")
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "facts_extracted": facts_count,
                "message": f"Successfully processed {file.filename} - extracted {facts_count} facts"
            })
            
        except Exception as e:
            print(f"❌ Error processing {file.filename}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "success": True,
        "results": results,
        "message": f"Processed {len(results)} file(s)"
    }


# Knowledge Base - Get Facts
@app.get("/api/knowledge/facts")
async def get_facts(
    include_inferred: bool = True,
    min_confidence: float = 0.0
):
    """
    Get all facts from the knowledge graph.
    Returns structured facts with metadata.
    """
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph not available")
    
    try:
        facts_list = []
        
        for s, p, o in graph:
            # Skip metadata triples
            predicate_str = str(p)
            if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object',
                                                'has_details', 'source_document', 'uploaded_at',
                                                'is_inferred', 'confidence', 'agent_id']):
                continue
            
            # Extract fact components
            from urllib.parse import unquote
            subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
            predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
            obj = str(o)
            
            # Get metadata
            sources = get_fact_source_document(subject, predicate, obj)
            details = get_fact_details(subject, predicate, obj)
            
            fact = {
                "id": f"{hash((subject, predicate, obj))}",
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "source_documents": [str(src) for src, _ in sources] if sources else [],
                "details": details if details else None
            }
            
            facts_list.append(fact)
        
        return {
            "success": True,
            "facts": facts_list,
            "total_facts": len(facts_list),
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error getting facts: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving facts: {str(e)}")


# Knowledge Base - Create Fact
@app.post("/api/knowledge/facts")
async def create_fact(fact: FactCreate):
    """Create a new fact in the knowledge graph"""
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph not available")
    
    try:
        add_to_graph(
            text=f"{fact.subject} {fact.predicate} {fact.object}",
            source_document=fact.source_document,
            agent_id="api"
        )
        save_knowledge_graph()
        
        return {
            "success": True,
            "message": "Fact created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating fact: {str(e)}")


# Knowledge Base - Delete Fact
@app.delete("/api/knowledge/facts")
async def delete_fact(fact: FactDelete):
    """Delete a fact from the knowledge graph"""
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph not available")
    
    try:
        # Find and remove the fact
        from urllib.parse import quote
        from rdflib import URIRef
        
        s = URIRef(f"urn:{quote(fact.subject, safe='')}")
        p = URIRef(f"urn:{quote(fact.predicate, safe='')}")
        o = fact.object
        
        graph.remove((s, p, o))
        save_knowledge_graph()
        
        return {
            "success": True,
            "message": "Fact deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting fact: {str(e)}")


# Chat/Query
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Process chat messages and return responses using the knowledge graph.
    """
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph not available")
    
    try:
        if RESPONSES_AVAILABLE:
            response_text = respond(
                message=request.message,
                history=request.history
            )
        else:
            # Simple fallback response
            response_text = f"I received your message: {request.message}. The full chat system is not available, but I can help with knowledge graph queries."
        
        return {
            "success": True,
            "response": response_text,
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


# Operational Insights
@app.get("/api/insights/operational")
async def get_operational_insights():
    """
    Get operational insights computed from CSV data.
    Returns manager, department, and recruitment source analytics.
    """
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph not available")
    
    try:
        # Find CSV file
        csv_path = find_csv_file_path()
        
        # Try direct path if not found
        if not csv_path:
            csv_path = "/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv"
        
        if not csv_path or not os.path.exists(csv_path):
            return {
                "success": True,
                "data": {
                    "insights": {},
                    "message": "No CSV file available. Please upload a CSV file first."
                }
            }
        
        # Load and compute insights
        df = load_csv_data(csv_path)
        if df is None or len(df) == 0:
            return {
                "success": True,
                "data": {
                    "insights": {},
                    "message": "Could not load CSV data"
                }
            }
        
        insights = compute_operational_insights(df=df)
        
        return {
            "success": True,
            "data": {
                "insights": insights,
                "processing_status": "completed"
            }
        }
    
    except Exception as e:
        print(f"❌ Error computing operational insights: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": True,
            "data": {
                "insights": {},
                "message": f"Error computing insights: {str(e)}"
            }
        }


# Process Documents
@app.post("/api/process")
async def process_documents(request: Dict[str, Any]):
    """Process documents by their IDs"""
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph not available")
    
    document_ids = request.get("document_ids", [])
    
    results = []
    for doc_id in document_ids:
        try:
            # Find document
            documents = get_all_documents()
            doc = next((d for d in documents if d.get("id") == doc_id), None)
            
            if not doc:
                results.append({
                    "document_id": doc_id,
                    "status": "error",
                    "error": "Document not found"
                })
                continue
            
            # Process document
            if AGENT_SYSTEM_AVAILABLE:
                result = process_document_with_agents(
                    document_id=doc_id,
                    document_name=doc.get("name", ""),
                    document_type=doc.get("type", ""),
                    file_path=doc.get("file_path")
                )
            else:
                result = {"facts_count": 0, "status": "processed"}
            
            results.append({
                "document_id": doc_id,
                "status": "success",
                "facts_extracted": result.get("facts_count", 0) if isinstance(result, dict) else 0
            })
        
        except Exception as e:
            results.append({
                "document_id": doc_id,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "success": True,
        "results": results
    }


# Export Facts
@app.get("/api/export")
async def export_facts():
    """Export all facts as JSON"""
    if not KG_AVAILABLE:
        raise HTTPException(status_code=500, detail="Knowledge graph not available")
    
    try:
        facts_list = []
        
        for s, p, o in graph:
            predicate_str = str(p)
            if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object']):
                continue
            
            from urllib.parse import unquote
            subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
            predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
            obj = str(o)
            
            facts_list.append({
                "subject": subject,
                "predicate": predicate,
                "object": obj
            })
        
        return {
            "success": True,
            "facts": facts_list,
            "total_facts": len(facts_list),
            "exported_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting facts: {str(e)}")


# Documents List
@app.get("/api/documents")
async def get_documents():
    """Get list of uploaded documents"""
    try:
        documents = get_all_documents()
        return {
            "success": True,
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")


# ============================================================================
# Document Statistics & Visualizations
# ============================================================================

def _resolve_document_path(document_id: str) -> Optional[str]:
    """
    Resolve a document ID or name to a CSV file path using documents_store.json.
    Falls back to find_csv_file_path() if not found.
    """
    try:
        documents = get_all_documents()
        for doc in documents:
            if (
                str(doc.get("id")) == str(document_id)
                or str(doc.get("name")) == str(document_id)
            ):
                path = doc.get("file_path")
                if path and os.path.exists(path):
                    return path
    except Exception:
        pass

    # Fallback: try global CSV finder
    return find_csv_file_path()


def _compute_document_statistics(csv_path: str) -> Dict[str, Any]:
    """
    Compute basic statistics for a CSV document.
    Returns a dict matching what the frontend expects in statistics.tsx.
    """
    import pandas as pd

    df = load_csv_data(csv_path)
    if df is None or len(df) == 0:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "column_types": {},
            "descriptive_stats": {},
            "missing_values": {},
            "correlations": {},
            "data_quality": {},
        }

    stats: Dict[str, Any] = {}
    stats["total_rows"] = int(len(df))
    stats["total_columns"] = int(len(df.columns))

    # Determine column types
    column_types: Dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_types[col] = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            column_types[col] = "boolean"
        else:
            column_types[col] = "categorical"
    stats["column_types"] = column_types

    # Descriptive statistics per column
    descriptive_stats: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col].dropna()
        col_type = column_types[col]

        if col_type == "numeric":
            if len(series) == 0:
                continue
            descriptive_stats[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(series.count()),
            }
        else:
            # Categorical / other: value counts
            value_counts = series.value_counts().to_dict()
            # Cast keys to str for JSON
            value_counts_str = {str(k): int(v) for k, v in value_counts.items()}
            descriptive_stats[col] = {
                "value_counts": value_counts_str,
                "count": int(series.count()),
            }
    stats["descriptive_stats"] = descriptive_stats

    # Missing values
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    stats["missing_values"] = missing_values

    # Data quality: completeness and unique values
    data_quality: Dict[str, Any] = {}
    total_rows = len(df)
    for col in df.columns:
        non_missing = total_rows - missing_values[col]
        completeness = float(non_missing / total_rows) if total_rows > 0 else 0.0
        unique_values = int(df[col].nunique(dropna=True))
        data_quality[col] = {
            "completeness": completeness,
            "unique_values": unique_values,
        }
    stats["data_quality"] = data_quality

    # Correlations (numeric columns only)
    num_cols = [col for col, t in column_types.items() if t == "numeric"]
    correlations: Dict[str, Dict[str, float]] = {}
    if len(num_cols) >= 2:
        corr_df = df[num_cols].corr()
        for col1 in num_cols:
            correlations[col1] = {}
            for col2 in num_cols:
                val = corr_df.loc[col1, col2]
                correlations[col1][col2] = float(val) if pd.notna(val) else 0.0
    stats["correlations"] = correlations

    return stats


@app.get("/api/documents/{document_id}/statistics")
async def get_document_statistics(document_id: str = Path(...)):
    """
    Get statistics for a specific document (CSV).
    """
    csv_path = _resolve_document_path(document_id)
    if not csv_path or not os.path.exists(csv_path):
        return {
            "success": True,
            "data": {
                "statistics": None,
                "message": f"No CSV file found for document {document_id}",
            },
        }

    try:
        statistics = _compute_document_statistics(csv_path)
        return {
            "success": True,
            "data": {
                "statistics": statistics,
            },
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error computing statistics: {str(e)}",
        }


@app.get("/api/documents/{document_id}/visualizations")
async def get_document_visualizations(document_id: str = Path(...)):
    """
    Get simple visualization-ready data for a document.
    Currently returns an empty structure; the statistics endpoint
    already provides what the frontend needs for charts.
    """
    return {
        "success": True,
        "data": {
            "visualizations": {},
        },
    }


@app.get("/api/documents/{document_id}/statistics/export")
async def export_document_statistics(document_id: str = Path(...)):
    """
    Export statistics as JSON structure.
    The frontend will download this as a file.
    """
    csv_path = _resolve_document_path(document_id)
    if not csv_path or not os.path.exists(csv_path):
        return {
            "success": False,
            "error": f"No CSV file found for document {document_id}",
        }

    try:
        statistics = _compute_document_statistics(csv_path)
        return {
            "success": True,
            "data": statistics,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error exporting statistics: {str(e)}",
        }


@app.get("/api/documents/{document_id}/summary")
async def get_document_summary(document_id: str = Path(...)):
    """
    Get a short text summary of the document.
    For now, returns a simple summary based on rows/columns.
    """
    csv_path = _resolve_document_path(document_id)
    if not csv_path or not os.path.exists(csv_path):
        return {
            "success": True,
            "data": {
                "summary": f"No CSV file found for document {document_id}.",
            },
        }

    try:
        df = load_csv_data(csv_path)
        if df is None or len(df) == 0:
            summary = "The dataset could not be loaded or is empty."
        else:
            summary = (
                f"The dataset '{os.path.basename(csv_path)}' contains "
                f"{len(df)} rows and {len(df.columns)} columns. "
                f"Key columns include: {', '.join(list(df.columns)[:8])}."
            )
        return {
            "success": True,
            "data": {
                "summary": summary,
            },
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error generating summary: {str(e)}",
        }


# Delete Document
@app.delete("/api/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a document from the store"""
    try:
        # Try by ID first, then by name
        result = delete_document(document_id=document_id)
        if not result:
            # Try as name
            result = delete_document(name=document_id)
        
        if result:
            return {
                "success": True,
                "message": f"Document {document_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8001"))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

