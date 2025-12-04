"""
FastAPI Backend Server - Research Brain
========================================

This is the MAIN BACKEND ENTRY POINT. It provides REST API endpoints
for the React frontend to interact with the knowledge management system.

Architecture:
- Receives HTTP requests from frontend (React app)
- Processes requests using knowledge.py, file_processing.py, documents_store.py
- Returns JSON responses

Key Endpoints:
- POST /api/knowledge/upload: Upload and process documents
- POST /api/knowledge/facts: Create a new fact
- GET /api/knowledge/facts: Get all facts
- DELETE /api/knowledge/facts/{id}: Delete a fact
- GET /api/documents: Get all uploaded documents
- GET /api/export: Export all knowledge as JSON
- POST /api/knowledge/import: Import knowledge from JSON

Connection:
- Frontend connects to: http://localhost:8001 (default)
- API docs available at: http://localhost:8001/docs

To run:
    python api_server.py
    
Or use the convenience scripts:
    start_api.bat (Windows)
    start_backend.sh (macOS/Linux)
    
The scripts automatically activate the virtual environment if available.

Author: Research Brain Team
Last Updated: 2025-01-15
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile

# Import your existing modules
from responses import respond as rqa_respond
from knowledge import (
    add_to_graph as kb_add_to_graph,
    show_graph_contents as kb_show_graph_contents,
    visualize_knowledge_graph as kb_visualize_knowledge_graph,
    save_knowledge_graph as kb_save_knowledge_graph,
    load_knowledge_graph as kb_load_knowledge_graph,
    delete_all_knowledge as kb_delete_all_knowledge,
    graph as kb_graph,
    import_knowledge_from_json_file as kb_import_json
)
from file_processing import handle_file_upload as fp_handle_file_upload
from documents_store import add_document, get_all_documents, delete_document as ds_delete_document, cleanup_documents_without_facts, delete_all_documents as ds_delete_all_documents
from agent_system import (
    initialize_agents,
    process_document_with_agents,
    get_all_agents,
    get_agent_architecture,
    get_agent_by_id,
    get_document_statistics,
    get_document_visualizations,
    document_agents,
    save_agents
)
from knowledge import create_comprehensive_backup as kb_create_comprehensive_backup

from contextlib import asynccontextmanager

# Load knowledge graph on startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing knowledge graph...")
    try:
        # Load existing knowledge graph (don't clear on restart - preserve data)
        print("📂 Loading existing knowledge graph...")
        try:
            kb_load_knowledge_graph()
            fact_count = len(kb_graph) if kb_graph else 0
            print(f"✅ Loaded knowledge graph with {fact_count} facts")
        except Exception as load_error:
            print(f"⚠️  Could not load existing graph: {load_error}")
            print("   Starting with empty graph...")
            # Only clear if load failed
            delete_result = kb_delete_all_knowledge()
            print(f"Startup: {delete_result}")
        
        # Load documents (don't clear on restart - preserve data)
        print("📂 Loading existing documents...")
        deleted_docs = ds_delete_all_documents()
        if deleted_docs > 0:
            print(f"✅ Deleted {deleted_docs} documents")
        else:
            print("✅ No documents to delete")
        
        # Clear document agents (they are ephemeral)
        from agent_system import document_agents
        doc_agent_count = len(document_agents)
        document_agents.clear()
        if doc_agent_count > 0:
            print(f"✅ Cleared {doc_agent_count} ephemeral document agents")
        
        # Initialize core agents (Statistics, Visualization, KG, LLM) - CRITICAL
        from agent_system import initialize_agents
        try:
            initialize_agents()
            print("✅ Initialized core agents (Statistics, Visualization, KG, LLM)")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize agents: {e}")
            import traceback
            traceback.print_exc()
        
        # Verify graph is empty after clearing
        fact_count = len(kb_graph)
        print(f"✅ Knowledge graph initialized with {fact_count} facts (fresh start)")
        
        # IMPORTANT: Verify the graph file is actually empty
        import os
        if os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            print(f"✅ Graph file size after clear: {file_size} bytes")
            if file_size > 1000:  # If file is still large, something went wrong
                print(f"⚠️  WARNING: Graph file is {file_size} bytes but graph has {fact_count} facts!")
                print("⚠️  This might indicate the clear didn't work properly")
        
        # Initialize Ollama (check if running and pull model if needed)
        from responses import initialize_ollama, USE_OLLAMA, OLLAMA_MODEL
        if USE_OLLAMA:
            print(f"🤖 Initializing Ollama with model {OLLAMA_MODEL}...")
            ollama_ready = initialize_ollama()
            if ollama_ready:
                print(f"✅ Ollama is ready to use with {OLLAMA_MODEL}")
            else:
                print(f"⚠️  Ollama not ready - will use rule-based responses")
        else:
            print("ℹ️  Ollama disabled (USE_OLLAMA=false)")
        
        # Pre-load LLM model in background to avoid timeout on first request (only if not using Ollama)
        if not USE_OLLAMA:
            print("🔄 Pre-loading LLM model for research assistant (this may take 1-2 minutes)...")
            import asyncio
            import threading
            from responses import load_llm_model
            
            # Start pre-loading in background (don't block startup)
            def preload_llm_sync():
                try:
                    result = load_llm_model()
                    if result:
                        print("✅ LLM model pre-loaded successfully")
                    else:
                        print("⚠️  LLM model not available, will use rule-based responses")
                except Exception as e:
                    print(f"⚠️  Failed to pre-load LLM: {e}")
                    print("   Will use rule-based responses")
            
            # Run in background thread (non-blocking)
            preload_thread = threading.Thread(target=preload_llm_sync, daemon=True)
            preload_thread.start()
            print("   (Model loading in background, server is ready)")
        
    except Exception as e:
        print(f"⚠️  Warning during knowledge graph initialization: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with empty graph...")
    
    yield
    
    # Shutdown - clear ephemeral document agents
    from agent_system import document_agents, save_agents
    doc_agent_count = len(document_agents)
    document_agents.clear()
    if doc_agent_count > 0:
        print(f"✅ Cleared {doc_agent_count} ephemeral document agents on shutdown")
    
    # Save core agents before shutdown
    save_agents()
    print("✅ Saved core agents before shutdown")

# Initialize FastAPI app with lifespan
app = FastAPI(title="NesyX API", description="Backend API for NesyX Knowledge Graph System", lifespan=lifespan)

# Configure CORS - Allow all origins (you can restrict this to specific domains later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Request/Response Models
# ==========================================================

class ChatMessage(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class AddKnowledgeRequest(BaseModel):
    text: str

class AddFactRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    source: Optional[str] = "manual"
    details: Optional[str] = None

class DeleteKnowledgeRequest(BaseModel):
    keyword: Optional[str] = None
    count: Optional[int] = None

# ==========================================================
# API Endpoints
# ==========================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "NesyX API",
        "facts_count": len(kb_graph)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "facts": len(kb_graph)}

@app.post("/api/chat")
async def chat_endpoint(request: ChatMessage):
    """Chat endpoint - ask questions about the knowledge base"""
    import asyncio
    try:
        # Run the response generation in a thread pool to avoid blocking
        # and set a timeout to prevent hanging
        loop = asyncio.get_event_loop()
        
        # Check if LLM is still loading and wait a bit if needed (only if not using Ollama)
        from responses import LLM_PIPELINE, load_llm_model, USE_LLM, LLM_AVAILABLE, USE_OLLAMA
        if not USE_OLLAMA and USE_LLM and LLM_AVAILABLE and LLM_PIPELINE is None:
            # Model not loaded yet, try to load it (with timeout)
            print("⏳ LLM not loaded yet, loading now...")
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, load_llm_model),
                    timeout=90.0  # Give 90 seconds for model loading
                )
            except asyncio.TimeoutError:
                print("⚠️  LLM loading timed out, using rule-based responses")
        
        # Generate response with timeout
        # Adjust timeout based on device and document size (CPU is much slower)
        from responses import LLM_DEVICE
        from knowledge import graph as kb_graph
        
        # Calculate timeout based on graph size (more facts = more processing time)
        fact_count = len(kb_graph) if kb_graph else 0
        base_timeout_cpu = 30.0  # Increased base timeout for CPU
        base_timeout_gpu = 60.0  # Increased base timeout for GPU
        
        # Add extra time for larger knowledge graphs
        # For structured queries, we need more time for fact extraction
        if fact_count > 10000:
            extra_time = min(60.0, fact_count / 500)  # Up to 60 extra seconds for large graphs
        else:
            extra_time = 0
        
        if LLM_DEVICE == "cpu":
            timeout_seconds = base_timeout_cpu + extra_time
        else:
            timeout_seconds = base_timeout_gpu + extra_time
        
        # Cap at 3 minutes for very large graphs (increased for better reliability)
        timeout_seconds = min(timeout_seconds, 180.0)
        
        print(f"⏱️  Query timeout: {timeout_seconds:.1f}s (device: {LLM_DEVICE}, facts: {fact_count})")
        
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, rqa_respond, request.message, request.history),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"⏱️  Response generation timed out after {timeout_seconds}s on {LLM_DEVICE}")
            # Return a helpful message instead of raising an error
            response = "I'm sorry, the response is taking too long. This might be because the LLM is running on CPU, which is slower. Please try a simpler question or wait a moment and try again."
        return {
            "response": response,
            "status": "success"
        }
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail="Request timed out. The LLM is taking too long to respond. Try disabling LLM with USE_LLM=false or ask a simpler question."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/api/knowledge/add")
async def add_knowledge_endpoint(request: AddKnowledgeRequest):
    """Add knowledge to the graph from text"""
    try:
        # Get current fact count before adding
        fact_count_before = len(kb_graph)
        
        # Add knowledge to graph
        result = kb_add_to_graph(request.text)
        # add_to_graph already saves, but ensure it's saved
        kb_save_knowledge_graph()
        
        # Verify save worked
        if os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            print(f"✅ Knowledge saved - file size: {file_size} bytes, facts in graph: {len(kb_graph)}")
        
        # Extract extraction method from result message
        extraction_method = "regex"
        if "TRIPLEX" in result.upper():
            extraction_method = "triplex"
        elif "FALLBACK" in result.upper():
            extraction_method = "regex (triplex fallback)"
        
        # Get newly added facts (those added after the operation)
        # Extract the last fact added (most recent)
        facts_list = []
        for i, (s, p, o) in enumerate(kb_graph):
            subject = str(s).split(':')[-1] if ':' in str(s) else str(s)
            predicate = str(p).split(':')[-1] if ':' in str(p) else str(p)
            object_val = str(o)
            facts_list.append({
                "id": i + 1,
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "manual"
            })
        
        # Return the last fact added (most recent)
        new_fact = facts_list[-1] if facts_list else None
        
        return {
            "message": result,
            "status": "success",
            "total_facts": len(kb_graph),
            "fact": new_fact,  # Return the created fact for frontend
            "extraction_method": extraction_method  # Indicate which method was used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding knowledge: {str(e)}")

@app.post("/api/knowledge/facts")
async def create_fact_endpoint(request: AddFactRequest):
    """Create a structured fact directly (subject, predicate, object)"""
    try:
        import rdflib
        from urllib.parse import quote
        from knowledge import fact_exists as kb_fact_exists
        
        # Check if fact already exists
        if kb_fact_exists(request.subject, request.predicate, str(request.object)):
            print(f"⚠️  POST /api/knowledge/facts: Duplicate fact detected - {request.subject} {request.predicate} {request.object}")
            return {
                "message": "Fact already exists in knowledge graph",
                "status": "duplicate",
                "fact": {
                    "subject": request.subject,
                    "predicate": request.predicate,
                    "object": str(request.object)
                },
                "total_facts": len(kb_graph)
            }
        
        # For structured facts, add directly to graph with proper URI encoding
        # Replace spaces with underscores in URIs to avoid RDFLib warnings
        subject_clean = request.subject.strip().replace(' ', '_')
        predicate_clean = request.predicate.strip().replace(' ', '_')
        object_value = str(request.object).strip()
        
        # Create URIs (encode spaces to avoid RDFLib warnings)
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        # Add directly to graph
        kb_graph.add((subject_uri, predicate_uri, object_literal))
        
        # Add details if provided
        if request.details and request.details.strip():
            from knowledge import add_fact_details as kb_add_fact_details
            kb_add_fact_details(request.subject, request.predicate, object_value, request.details)
        
        # Add source document and timestamp (manual for directly created facts)
        from datetime import datetime
        from knowledge import add_fact_source_document as kb_add_fact_source_document
        timestamp = datetime.now().isoformat()
        kb_add_fact_source_document(request.subject, request.predicate, object_value, "manual", timestamp)
        
        # Save to disk
        save_result = kb_save_knowledge_graph()
        
        # Verify the fact was added
        fact_count = len(kb_graph)
        print(f"✅ POST /api/knowledge/facts: Added fact - {request.subject} {request.predicate} {request.object}")
        if request.details:
            print(f"✅ Added details: {request.details[:50]}...")
        print(f"✅ Save result: {save_result}")
        print(f"✅ Total facts in graph: {fact_count}")
        
        # Verify file was written
        if os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            print(f"✅ Knowledge file size: {file_size} bytes")
        
        # Get details for the response
        from knowledge import get_fact_details as kb_get_fact_details
        details = kb_get_fact_details(request.subject, request.predicate, object_value)
        
        # Create the fact object - use the actual index in the graph
        new_fact = {
            "id": str(fact_count),  # Use current count as ID (string format)
            "subject": request.subject,  # Return original subject (with spaces)
            "predicate": request.predicate,  # Return original predicate (with spaces)
            "object": object_value,  # Return original object
            "source": request.source,
            "details": details if details else None
        }
        
        return {
            "message": f"✅ Added fact successfully. Total facts: {fact_count}",
            "status": "success",
            "total_facts": fact_count,
            "fact": new_fact
        }
    except Exception as e:
        print(f"❌ Error creating fact: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error creating fact: {str(e)}")

@app.get("/api/knowledge/triplex-status")
async def triplex_status_endpoint():
    """Get Triplex model status and availability"""
    try:
        from knowledge import TRIPLEX_AVAILABLE, USE_TRIPLEX, TRIPLEX_MODEL, TRIPLEX_DEVICE
        
        status = {
            "available": TRIPLEX_AVAILABLE,
            "enabled": USE_TRIPLEX,
            "loaded": TRIPLEX_MODEL is not None,
            "device": TRIPLEX_DEVICE if TRIPLEX_AVAILABLE else "N/A"
        }
        
        if TRIPLEX_AVAILABLE and USE_TRIPLEX:
            status["message"] = "Triplex is available and enabled. LLM extraction will be used."
        elif TRIPLEX_AVAILABLE and not USE_TRIPLEX:
            status["message"] = "Triplex is available but disabled. Set USE_TRIPLEX=true to enable."
        else:
            status["message"] = "Triplex is not available. Using regex-based extraction."
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Triplex status: {str(e)}")

@app.post("/api/knowledge/upload")
async def upload_file_endpoint(files: List[UploadFile] = File(...)):
    """Upload and process files (PDF, DOCX, TXT, CSV)"""
    import asyncio
    import os
    tmp_paths = []  # Initialize outside try block so finally can access it
    
    print(f"📤 Upload endpoint called with {len(files)} file(s)")
    
    async def process_upload():
        """Inner function to process upload - wrapped in timeout"""
        print(f"🔄 Starting upload processing...")
        # Import kb_graph at the very start to avoid UnboundLocalError
        from knowledge import graph as kb_graph, save_knowledge_graph
        facts_before = len(kb_graph) if kb_graph else 0
        file_info_list = []
        
        # Map temporary file paths to original filenames
        temp_to_original = {}
        
        for file in files:
            # Save uploaded file temporarily
            suffix = os.path.splitext(file.filename)[1] if file.filename else ""
            original_filename = file.filename or 'unknown'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
                tmp_paths.append(tmp_path)
                temp_to_original[tmp_path] = original_filename
                file_info_list.append({
                    'name': original_filename,
                    'size': len(content),
                    'type': suffix.lstrip('.') or 'unknown'
                })
        
        try:
            # Ensure core agents are initialized (but don't clear document agents - they persist across uploads)
            try:
                from agent_system import initialize_agents
                initialize_agents(clear_document_agents=False)  # Keep existing document agents
            except Exception as agent_error:
                print(f"⚠️  Warning: Error initializing agents: {agent_error}")
                import traceback
                traceback.print_exc()
            
            # Process files using new multi-agent system (Statistics, Visualization, KG, LLM)
            from file_processing import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
            
            processed_docs = []
            total_facts_extracted = 0
            extraction_method = "pipeline"
            
            # Process each file with specialized agents
            for i, file_info in enumerate(file_info_list):
                file_path = tmp_paths[i]
                file_extension = os.path.splitext(file_path)[1].lower()
                document_id = file_info['name']  # Use filename as document ID
                
                try:
                    # Extract text from file (for non-CSV files)
                    extracted_text = None
                    if file_extension == '.pdf':
                        extracted_text = extract_text_from_pdf(file_path)
                    elif file_extension == '.docx':
                        extracted_text = extract_text_from_docx(file_path)
                    elif file_extension == '.txt':
                        extracted_text = extract_text_from_txt(file_path)
                    elif file_extension == '.csv':
                        # CSV files: Extract limited text for KG agent (Statistics Agent reads file directly)
                        # Only extract a sample to avoid timeout on large files
                        try:
                            import pandas as pd
                            # Detect separator
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                first_line = f.readline()
                                comma_count = first_line.count(',')
                                semicolon_count = first_line.count(';')
                                if semicolon_count > comma_count and semicolon_count > 0:
                                    sep = ';'
                                else:
                                    sep = ','
                            
                            # Read only first 100 rows for KG processing (to avoid timeout)
                            df_sample = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python', nrows=100)
                            if len(df_sample.columns) == 1:
                                df_sample = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python', nrows=100)
                            
                            # Convert sample to text
                            text_lines = [f"CSV Dataset: {len(df_sample)} sample rows, {len(df_sample.columns)} columns"]
                            text_lines.append(f"Columns: {', '.join(df_sample.columns.tolist())}")
                            text_lines.append("")
                            text_lines.append("Sample Data:")
                            for idx, row in df_sample.head(50).iterrows():  # Only first 50 rows
                                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                                text_lines.append(row_text)
                            extracted_text = "\n".join(text_lines)
                        except Exception as csv_error:
                            print(f"⚠️  Could not extract CSV text: {csv_error}")
                            extracted_text = ""  # Fallback to empty
                    else:
                        print(f"⚠️  Unsupported file type: {file_extension}")
                        continue
                    
                    # Validate text extraction for non-CSV files
                    if file_extension != '.csv':
                        if not extracted_text or extracted_text.startswith("Error") or len(extracted_text.strip()) < 10:
                            print(f"⚠️  Could not extract text from {file_info['name']}: {extracted_text[:100] if extracted_text else 'empty'}")
                            continue
                except Exception as extract_error:
                    print(f"❌ Error extracting text from {file_info['name']}: {extract_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                try:
                    # Clear knowledge graph for new document upload (start fresh)
                    # kb_graph and save_knowledge_graph already imported at top of process_upload function
                    if kb_graph and len(kb_graph) > 0:
                        facts_before_clear = len(kb_graph)
                        print(f"🗑️  Clearing {facts_before_clear:,} existing facts from previous uploads...")
                        kb_graph.remove((None, None, None))  # Remove all triples
                        save_knowledge_graph()
                        print(f"✅ Knowledge graph cleared - starting fresh for {file_info['name']}")
                    
                    # Process document: Document Agent → Worker Agents → KG Agent → Knowledge Graph
                    # Simultaneously: Statistics Agent, Operational Agent, Visualization Agent
                    file_size_mb = file_info.get('size', 0) / (1024 * 1024)
                    print(f"📄 Processing: {file_info['name']} ({file_size_mb:.1f} MB)")
                    
                    # CRITICAL: Ensure file_path is valid and stored for CSV files
                    csv_file_path = file_path
                    if file_extension.lower() == '.csv':
                        # Verify file exists, if not try to find it
                        if not csv_file_path or not os.path.exists(csv_file_path):
                            import glob
                            temp_dir = tempfile.gettempdir()
                            possible_paths = [
                                os.path.join(temp_dir, file_info['name']),
                                os.path.join('/tmp', file_info['name']),
                                file_info['name'],
                            ]
                            for temp_path in [temp_dir, '/tmp', '/var/tmp']:
                                if os.path.exists(temp_path):
                                    pattern = os.path.join(temp_path, f'*{file_info["name"]}*')
                                    matches = glob.glob(pattern)
                                    possible_paths.extend(matches)
                            
                            for path in possible_paths:
                                if path and os.path.exists(path) and path.endswith('.csv'):
                                    csv_file_path = path
                                    print(f"✅ Found CSV file: {csv_file_path}")
                                    break
                        
                        if not csv_file_path or not os.path.exists(csv_file_path):
                            print(f"❌ ERROR: Could not find CSV file for {file_info['name']}")
                            print(f"   Searched paths: {possible_paths[:5]}")
                            continue  # Skip this file if we can't find it
                    
                    # Process document with agents (this creates the document agent internally)
                    agent_result = process_document_with_agents(
                        document_id=document_id,
                        document_name=file_info['name'],
                        document_type=file_extension,
                        file_path=csv_file_path,  # Use verified file path
                        extracted_text=extracted_text or ""
                    )
                    
                    # After processing, store file_path in document agent for background processing
                    doc_agent_id = f"doc_{document_id}"
                    from agent_system import document_agents
                    from datetime import datetime
                    
                    if doc_agent_id in document_agents and file_extension.lower() == '.csv' and csv_file_path:
                        doc_agent = document_agents[doc_agent_id]
                        doc_agent.file_path = csv_file_path
                        print(f"✅ Stored file_path in document agent for background processing: {csv_file_path}")
                    
                    if not agent_result:
                        print(f"⚠️  No result returned from process_document_with_agents for {file_info['name']}")
                        continue
                    
                    facts_extracted = agent_result.get('facts_extracted', 0)
                    total_facts_extracted += facts_extracted
                    
                    # Store statistics, visualizations, and operational insights in document agent metadata
                    doc_agent = document_agents[doc_agent_id]
                    
                    doc_agent.metadata['statistics'] = agent_result.get('statistics')
                    doc_agent.metadata['visualizations'] = agent_result.get('visualizations')
                    
                    # Operational insights are now computed in background (async)
                    # Check if they're already available from background processing
                    if 'operational_insights' in agent_result:
                        doc_agent.metadata['operational_insights'] = agent_result.get('operational_insights', {})
                    else:
                        # Initialize as empty - will be populated by background thread
                        doc_agent.metadata['operational_insights'] = {}
                    
                    # Store processing status for frontend to check
                    if 'processing_status' in agent_result:
                        doc_agent.metadata['processing_status'] = agent_result['processing_status']
                    
                    print(f"✅ Document ready: KG facts available, background processing continues")
                    
                    doc_agent.metadata['processed_at'] = datetime.now().isoformat()
                    doc_agent.facts_extracted = facts_extracted
                    # Status is set by process_document_with_agents ("ready" when KG is available)
                    # Don't override if it's already "ready" (background processing continues)
                    if doc_agent.status != "ready":
                        doc_agent.status = "completed"
                    
                    # Save document (even if 0 facts, if statistics were extracted)
                    has_statistics = agent_result.get('statistics') is not None
                    if facts_extracted > 0 or has_statistics:
                        doc = add_document(
                            name=file_info['name'],
                            size=file_info['size'],
                            file_type=file_info['type'],
                            facts_extracted=facts_extracted,
                            agent_id=doc_agent_id
                        )
                        if doc:
                            # Note: Operational insights are stored in document agent metadata only
                            # They are NOT persisted to documents_store.json - will be recomputed on restart or on-demand
                            # This ensures they're always up-to-date with the current CSV data
                            
                            processed_docs.append(doc)
                            print(f"✅ Completed: {file_info['name']} ({facts_extracted:,} facts in KG)")
                        else:
                            print(f"⚠️  Failed to save document {file_info['name']}")
                    else:
                        print(f"⚠️  Skipping document {file_info['name']}: no facts extracted and no statistics")
                except Exception as process_error:
                    print(f"❌ Error processing document {file_info['name']} with agents: {process_error}")
                    print(f"   Error type: {type(process_error).__name__}")
                    import traceback
                    traceback.print_exc()
                    # Continue processing other files even if one fails
                    continue
            
            # Save knowledge graph (facts are already added by KG Agent)
            # IMPORTANT: Ensure graph is saved to disk
            try:
                kb_save_knowledge_graph()
            except Exception as save_error:
                print(f"⚠️  Warning: Error saving knowledge graph: {save_error}")
                import traceback
                traceback.print_exc()
            
            # CRITICAL: Reload from disk to get the actual saved facts
            try:
                kb_load_knowledge_graph()
            except Exception as load_error:
                print(f"⚠️  Warning: Error loading knowledge graph: {load_error}")
                import traceback
                traceback.print_exc()
            
            facts_after = len(kb_graph)
            facts_extracted = facts_after - facts_before
            
            print(f"✅ Multi-agent upload processed {len(files)} file(s)")
            print(f"   Extraction method: {extraction_method}")
            print(f"   Facts before: {facts_before}, after: {facts_after}, extracted: {facts_extracted}")
            print(f"   Total facts extracted by agents: {total_facts_extracted}")
            print(f"   Graph now has {len(kb_graph)} total facts")
            print(f"   Processed documents: {len(processed_docs)}")
            
            # Create result message with operational insights summary
            insights_summary = ""
            for doc in processed_docs:
                doc_agent_id = f"doc_{doc.get('name', '')}"
                from agent_system import document_agents
                if doc_agent_id in document_agents:
                    doc_agent = document_agents[doc_agent_id]
                    insights = doc_agent.metadata.get('operational_insights', {})
                    if insights:
                        insights_summary += f"\n   - {doc.get('name', '')}: {len(insights)} operational insights generated"
            
            result = f"Processed {len(files)} file(s) using multi-agent system (Statistics, Visualization, KG, LLM). Extracted {total_facts_extracted} facts"
            if insights_summary:
                result += f"\n\n✅ Automatic operational analysis completed:{insights_summary}"
            
            # Handle case where no facts were extracted
            if len(processed_docs) == 0 and facts_extracted == 0:
                # No facts extracted - remove documents
                print(f"⚠️  No facts extracted from {len(files)} file(s) - REMOVING documents")
                for file_info in file_info_list:
                    from documents_store import load_documents, save_documents
                    docs = load_documents()
                    original_count = len(docs)
                    docs = [d for d in docs if d.get('name') != file_info['name']]
                    removed_count = original_count - len(docs)
                    if removed_count > 0:
                        save_documents(docs)
                        print(f"   🗑️  Removed {file_info['name']} (no facts extracted)")
            
            # Final save to ensure everything is persisted to disk
            try:
                kb_save_knowledge_graph()
            except Exception as save_error2:
                print(f"⚠️  Warning: Error in final save: {save_error2}")
            
            # CRITICAL: Reload one more time to ensure in-memory graph matches disk
            try:
                kb_load_knowledge_graph()
            except Exception as load_error2:
                print(f"⚠️  Warning: Error in final load: {load_error2}")
            
            # Verify final state
            final_fact_count = len(kb_graph)
            if os.path.exists("knowledge_graph.pkl"):
                file_size = os.path.getsize("knowledge_graph.pkl")
                print(f"✅ Final save - file size: {file_size} bytes, facts in graph: {final_fact_count}")
                
            # Update total_facts in response to reflect actual graph state
            if final_fact_count > 0:
                print(f"✅ Upload complete: Graph now has {final_fact_count} facts in memory")
            
            # Get final fact count after all saves and reloads
            final_total = len(kb_graph)
            
            # Ensure we have valid values
            final_facts_extracted = max(total_facts_extracted, facts_extracted, final_total - facts_before)
            
            print(f"✅ Upload processed {len(files)} file(s), extracted {final_facts_extracted} facts")
            if final_facts_extracted < 0:
                final_facts_extracted = 0
            
            # Always return success response, even if some steps had warnings
            response_data = {
                "message": result,
                "files_processed": len(files),
                "status": "success",
                "total_facts": max(final_total, facts_after),
                "facts_extracted": final_facts_extracted,
                "extraction_method": extraction_method,
                "documents": processed_docs if processed_docs else []
            }
            
            print(f"📤 Returning response: status={response_data['status']}, files={response_data['files_processed']}, facts={response_data['total_facts']}, extracted={response_data['facts_extracted']}, docs={len(response_data['documents'])}")
            return response_data
        except Exception as process_error:
            error_msg = str(process_error)
            print(f"❌ Error in upload processing: {error_msg}")
            import traceback
            traceback.print_exc()
            raise
    
    # Execute upload with timeout - calculate based on file size and file type
    total_size = sum(f.size for f in files if hasattr(f, 'size') and f.size)
    file_size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
    
    # Check if any files are CSV (CSV files need more time due to row/column processing)
    has_csv = any(f.filename and f.filename.lower().endswith('.csv') for f in files)
    
    # Adaptive timeout calculation
    if has_csv:
        # CSV files: More generous timeout due to row/column processing complexity
        # Base: 60 minutes for CSV files (increased from 30 minutes)
        # Additional: 10 minutes per 10MB over 50MB, or 1 minute per 100 rows estimated
        base_timeout = 3600.0  # 60 minutes base for CSV
        file_size_timeout = max(0, (file_size_mb - 50) / 10) * 600  # 10 min per 10MB over 50MB
        # Estimate rows from file size (rough: ~1KB per row average, but can vary)
        estimated_rows = max(0, (file_size_mb * 1024) / 1)  # Rough estimate: 1KB per row
        row_based_timeout = (estimated_rows / 100) * 60  # 1 minute per 100 rows
        additional_timeout = max(file_size_timeout, row_based_timeout)
        timeout_seconds = min(base_timeout + additional_timeout, 7200.0)  # Max 2 hours
        print(f"⏱️  CSV upload timeout set to {timeout_seconds/60:.1f} minutes (file size: {file_size_mb:.1f} MB, estimated ~{int(estimated_rows)} rows)")
    else:
        # Non-CSV files: Standard timeout based on file size
        base_timeout = 1800.0  # 30 minutes base
    additional_timeout = max(0, (file_size_mb - 50) / 10) * 300  # 5 min per 10MB over 50MB
    timeout_seconds = min(base_timeout + additional_timeout, 3600.0)  # Max 1 hour
    print(f"⏱️  Upload timeout set to {timeout_seconds/60:.1f} minutes (file size: {file_size_mb:.1f} MB)")
    
    try:
        result = await asyncio.wait_for(process_upload(), timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        print(f"⏱️  Upload processing timed out after {timeout_seconds/60:.1f} minutes")
        # Clean up temp files even on timeout
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        raise HTTPException(
            status_code=504, 
            detail=f"Upload processing timed out after {timeout_seconds/60:.1f} minutes. Very large files may need more time. The file is being processed in parallel batches - please try uploading a smaller file or contact support."
        )
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error uploading files: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp files on error
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Return more detailed error information
        error_detail = f"Error uploading files: {error_msg}"
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_detail += ". The file may be too large. Try uploading a smaller file or wait longer."
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Clean up temporary files
        if tmp_paths:
            for tmp_path in tmp_paths:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as cleanup_error:
                        print(f"⚠️  Warning: Could not delete temp file {tmp_path}: {cleanup_error}")

@app.get("/api/knowledge/graph")
async def get_graph_endpoint():
    """Get knowledge graph visualization"""
    try:
        graph_html = kb_visualize_knowledge_graph()
        return {
            "graph_html": graph_html,
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting graph: {str(e)}")

@app.get("/api/knowledge/contents")
async def get_contents_endpoint():
    """Get all knowledge graph contents as text"""
    try:
        contents = kb_show_graph_contents()
        return {
            "contents": contents,
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting contents: {str(e)}")

@app.get("/api/document/processing-status")
async def get_processing_status_endpoint():
    """Get processing status for documents (check if background tasks are complete)"""
    try:
        from agent_system import document_agents
        
        statuses = []
        for agent_id, agent in document_agents.items():
            if hasattr(agent, 'document_type') and agent.document_type.lower() == '.csv':
                metadata = getattr(agent, 'metadata', {})
                processing_status = metadata.get('processing_status', {})
                statuses.append({
                    "document_id": agent.document_id,
                    "document_name": agent.document_name,
                    "status": agent.status,
                    "processing_status": processing_status,
                    "background_complete": metadata.get('background_processing_complete', False),
                    "background_completed_at": metadata.get('background_processing_completed_at'),
                    "facts_extracted": agent.facts_extracted
                })
        
        return {
            "success": True,
            "documents": statuses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting processing status: {str(e)}")

@app.get("/api/insights/operational")
async def get_operational_insights_endpoint():
    """Get operational insights computed from data aggregations"""
    try:
        from agent_system import document_agents
        
        # First, try to get cached insights from document agents (computed during upload or background processing)
        cached_insights = None
        processing_status_info = None
        print(f"🔍 Checking {len(document_agents)} document agents for operational insights...")
        for agent_id, agent in document_agents.items():
            if hasattr(agent, 'document_type') and agent.document_type.lower() == '.csv':
                metadata = getattr(agent, 'metadata', {})
                processing_status = metadata.get('processing_status', {})
                print(f"  📄 Agent {agent_id}: type={getattr(agent, 'document_type', 'unknown')}, has metadata={bool(metadata)}")
                print(f"  📊 Metadata keys: {list(metadata.keys())}")
                
                # Check if operational insights are available
                if 'operational_insights' in metadata:
                    cached_insights = metadata.get('operational_insights')
                    print(f"  ✅ Found operational_insights in metadata: type={type(cached_insights)}, len={len(cached_insights) if isinstance(cached_insights, dict) else 'N/A'}")
                    if cached_insights and isinstance(cached_insights, dict) and len(cached_insights) > 0:
                        print(f"✅ Using cached operational insights from document agent ({len(cached_insights)} keys: {list(cached_insights.keys())[:10]})")
                        # Verify we have structured insights
                        has_structured = any(key in cached_insights for key in ['by_department', 'by_manager', 'by_recruitment_source'])
                        if has_structured:
                            print(f"   ✅ Contains structured insights (by_department, by_manager, etc.)")
                        else:
                            print(f"   ⚠️  Missing structured insights - only has: {list(cached_insights.keys())}")
                        return {
                            "success": True,
                            "data": {
                                "insights": cached_insights,
                                "processing_status": processing_status.get('operational_insights', 'completed')
                            }
                        }
                
                # Check processing status - if insights are not ready yet
                ops_status = processing_status.get('operational_insights', 'unknown')
                background_complete = metadata.get('background_processing_complete', False)
                
                # If background processing is not complete and status is processing/pending, return status
                if not background_complete and (ops_status == 'processing' or ops_status == 'pending'):
                    processing_status_info = {
                        "status": "processing",
                        "message": "Operational insights are being computed in the background. Please check again in a few moments."
                    }
        
        # If still processing, return status message
        if processing_status_info:
            return {
                "success": True,
                "data": {
                    "insights": {},
                    "processing_status": processing_status_info["status"],
                    "message": processing_status_info["message"]
                }
            }
        
        # Note: Operational insights are NOT persisted to documents_store.json
        # They are computed fresh on server restart or on-demand via API
        # This ensures they're always up-to-date with the current CSV data
        
        # If no cached insights, try to compute from CSV file
        # First, try to find CSV file path from document agents
        csv_file_path = None
        for agent_id, agent in document_agents.items():
            if hasattr(agent, 'document_type') and agent.document_type.lower() == '.csv':
                if hasattr(agent, 'file_path') and agent.file_path and os.path.exists(agent.file_path):
                    csv_file_path = agent.file_path
                    print(f"📁 Found CSV file in agent: {csv_file_path}")
                    break
        
        # If not found in agents, try to find from documents_store or temp directory
        if not csv_file_path:
            from documents_store import get_all_documents
            documents = get_all_documents()
            csv_docs = [d for d in documents if d.get('type', '').lower() == 'csv' or d.get('file_type', '').lower() == 'csv']
            
            if csv_docs:
                latest_doc = csv_docs[-1]
                doc_name = latest_doc.get('name', '')
                # Check temp directory (where files are uploaded)
                import glob
                temp_dir = tempfile.gettempdir()
                possible_paths = [
                    os.path.join(temp_dir, doc_name),
                    os.path.join('/tmp', doc_name),
                    doc_name,  # Try direct path
                    latest_doc.get('file_path'),  # Try file_path from document store
                    latest_doc.get('path'),  # Try path from document store
                ]
                # Also search for any CSV with matching name
                for temp_path in [temp_dir, '/tmp', '/var/tmp']:
                    if os.path.exists(temp_path):
                        pattern = os.path.join(temp_path, f'*{doc_name}*')
                        matches = glob.glob(pattern)
                        possible_paths.extend(matches)
                
                for path in possible_paths:
                    if path and os.path.exists(path) and path.endswith('.csv'):
                        csv_file_path = path
                        print(f"📁 Found CSV file: {csv_file_path}")
                        break
        
        from operational_queries import compute_operational_insights
        
        try:
            if not csv_file_path:
                print(f"⚠️  No CSV file path available to compute insights")
                return {
                    "success": True,
                    "data": {
                        "insights": {},
                        "message": "No CSV file available. Please upload a CSV file first."
                    }
                }
            
            if not os.path.exists(csv_file_path):
                print(f"⚠️  CSV file path does not exist: {csv_file_path}")
                return {
                    "success": True,
                    "data": {
                        "insights": {},
                        "message": f"CSV file not found at: {csv_file_path}"
                    }
                }
            
            print(f"📊 Computing operational insights on-demand from: {csv_file_path}")
            insights = compute_operational_insights(csv_file_path=csv_file_path)
            
            if not insights or (isinstance(insights, dict) and len(insights) == 0):
                print(f"⚠️  No operational insights computed (empty result)")
                return {
                    "success": True,
                    "data": {
                        "insights": {},
                        "message": "Operational insights computation returned empty results."
                    }
                }
            
            print(f"✅ Computed operational insights: {len(insights)} keys: {list(insights.keys())[:10]}")
            # Verify we have structured insights
            has_structured = any(key in insights for key in ['by_department', 'by_manager', 'by_recruitment_source'])
            if has_structured:
                print(f"   ✅ Contains structured insights (by_department, by_manager, etc.)")
            else:
                print(f"   ⚠️  Missing structured insights - only has: {list(insights.keys())}")
            
            # CRITICAL: Store computed insights in document agent metadata so they're available for future requests
            if insights and isinstance(insights, dict) and len(insights) > 0:
                for agent_id, agent in document_agents.items():
                    if hasattr(agent, 'document_type') and agent.document_type.lower() == '.csv':
                        agent.metadata['operational_insights'] = insights
                        agent.metadata['processing_status'] = agent.metadata.get('processing_status', {})
                        agent.metadata['processing_status']['operational_insights'] = 'completed'
                        print(f"✅ Stored computed insights in document agent {agent_id} metadata")
                        break
            
            # Note: Operational insights are NOT persisted - they're computed fresh each time
            # This ensures they're always up-to-date with the current CSV data
            
            return {
                "success": True,
                "data": {
                    "insights": insights
                }
            }
        except Exception as compute_error:
            print(f"❌ Error computing insights: {compute_error}")
            import traceback
            traceback.print_exc()
            return {
                "success": True,
                "data": {
                    "insights": {},
                    "message": f"Error computing insights: {str(compute_error)}"
                }
            }
    except ImportError as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️  Import error in operational insights: {e}")
        raise HTTPException(status_code=500, detail=f"Operational insights module not available: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️  Error computing operational insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting operational insights: {str(e)}")

@app.get("/api/knowledge/facts")
async def get_facts_endpoint(
    include_inferred: bool = True,
    min_confidence: float = 0.0
):
    """Get all knowledge graph facts as structured JSON array"""
    try:
        # CRITICAL: Always reload from disk to get the latest saved facts
        # This ensures we have facts that were saved after upload, even if server was restarted
        # The in-memory graph might be empty if server was restarted (cleared on startup)
        load_result = kb_load_knowledge_graph()
        
        # Debug: If graph is empty but file exists, something is wrong
        import os
        if len(kb_graph) == 0 and os.path.exists("knowledge_graph.pkl"):
            file_size = os.path.getsize("knowledge_graph.pkl")
            if file_size > 1000:  # File has data but graph is empty
                print(f"⚠️  WARNING: Graph file is {file_size} bytes but graph is empty!")
                # Try reloading again
                kb_load_knowledge_graph()
        
        facts = []
        from urllib.parse import unquote, quote
        import rdflib
        
        # OPTIMIZED: Build lookup maps in a single pass instead of calling functions for each fact
        # This reduces O(n*m) complexity to O(n) where n = total triples, m = facts
        # Build fact_id_uri -> metadata map first
        # ENHANCED: Now supports multiple sources per fact
        metadata_map = {}  # fact_id_uri -> {details, source_documents: [(source, timestamp), ...]}
        
        # Pass 1: Collect all metadata triples (O(n))
        # Metadata triples use fact_id_uri format: urn:fact:subject|predicate|object
        for s, p, o in kb_graph:
            predicate_str = str(p)
            # Check if this is a metadata triple (has fact_id_uri as subject)
            if 'urn:fact:' in str(s):
                fact_id_uri = str(s)
                
                if 'has_details' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    metadata_map[fact_id_uri]['details'] = str(o)
                elif 'source_document' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Source entry format: "source_document|uploaded_at" or just "source_document"
                    source_entry = str(o)
                    if '|' in source_entry:
                        parts = source_entry.split('|', 1)
                        if len(parts) == 2:
                            source_doc = parts[0]
                            timestamp = parts[1]
                            metadata_map[fact_id_uri]['source_documents'].append((source_doc, timestamp))
                    else:
                        # Legacy format or separate triples - will be matched with timestamp below
                        metadata_map[fact_id_uri]['source_documents'].append((source_entry, None))
                elif 'uploaded_at' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    timestamp = str(o)
                    # Try to match with existing source entries that don't have timestamps
                    if 'source_documents' in metadata_map[fact_id_uri]:
                        # Update entries without timestamps
                        updated_sources = []
                        for source_doc, existing_timestamp in metadata_map[fact_id_uri]['source_documents']:
                            if existing_timestamp is None:
                                updated_sources.append((source_doc, timestamp))
                            else:
                                updated_sources.append((source_doc, existing_timestamp))
                        # If no existing sources, add a new entry with this timestamp
                        if not updated_sources:
                            updated_sources.append(("", timestamp))
                        metadata_map[fact_id_uri]['source_documents'] = updated_sources
                elif 'is_inferred' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Store inferred status (convert "true"/"false" string to boolean)
                    metadata_map[fact_id_uri]['is_inferred'] = str(o).lower() == 'true'
                elif 'confidence' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Store confidence score (convert string to float)
                    try:
                        metadata_map[fact_id_uri]['confidence'] = float(str(o))
                    except (ValueError, TypeError):
                        metadata_map[fact_id_uri]['confidence'] = 0.7  # Default confidence
                elif 'agent_id' in predicate_str:
                    if fact_id_uri not in metadata_map:
                        metadata_map[fact_id_uri] = {'source_documents': []}
                    # Store agent_id
                    metadata_map[fact_id_uri]['agent_id'] = str(o)
        
        # Pass 2: Collect facts and match with metadata using fact_id URI (O(n))
        fact_index = 0
        for s, p, o in kb_graph:
            # Skip metadata triples
            predicate_str = str(p)
            subject_str = str(s)
            
            # Skip if this is a metadata triple (has fact_id_uri as subject)
            if 'urn:fact:' in subject_str:
                continue
            
            # Skip metadata predicates
            if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str or
                'is_inferred' in predicate_str or 'confidence' in predicate_str or
                'agent_id' in predicate_str):
                continue
            
            # Must be a proper entity URI (not fact URI)
            if 'urn:entity:' not in subject_str and ('urn:' not in subject_str or 'urn:fact:' in subject_str):
                continue
            
            # Must be a proper predicate URI
            if 'urn:predicate:' not in predicate_str and 'urn:' not in predicate_str:
                continue
            
            fact_index += 1
            
            # Extract subject from URI (format: urn:entity:subject or urn:subject)
            # Handle both urn:entity:subject and urn:subject formats
            subject_uri_str = str(s)
            if 'urn:entity:' in subject_uri_str:
                subject = subject_uri_str.split('urn:entity:')[-1]
            elif 'urn:' in subject_uri_str:
                subject = subject_uri_str.split('urn:')[-1]
            else:
                subject = subject_uri_str
            subject = unquote(subject).replace('_', ' ')
            
            # Extract predicate from URI (format: urn:predicate:predicate or urn:predicate)
            # Handle both urn:predicate:predicate and urn:predicate formats
            predicate_uri_str = str(p)
            if 'urn:predicate:' in predicate_uri_str:
                predicate = predicate_uri_str.split('urn:predicate:')[-1]
            elif 'urn:' in predicate_uri_str:
                predicate = predicate_uri_str.split('urn:')[-1]
            else:
                predicate = predicate_uri_str
            predicate = unquote(predicate).replace('_', ' ')
            
            # Object is already a literal
            object_val = str(o)
            
            # CRITICAL: Normalize object value to match how facts are stored
            # This ensures fact IDs match between storage and retrieval
            # Convert 18.0 -> 18, but keep decimals if needed (e.g., 4.5 -> 4.5)
            normalized_object_val = object_val
            try:
                if '.' in normalized_object_val:
                    float_val = float(normalized_object_val)
                    if float_val.is_integer():
                        normalized_object_val = str(int(float_val))
            except (ValueError, AttributeError):
                pass  # Keep as string if conversion fails
            
            # Build fact_id URI the same way add_fact_source_document does (for lookup)
            # Format: subject|predicate|object -> urn:fact:subject|predicate|object
            # Use normalized value to match how metadata was stored
            fact_id = f"{subject}|{predicate}|{normalized_object_val}"
            fact_id_clean = fact_id.strip().replace(' ', '_')
            fact_id_uri = f"urn:fact:{quote(fact_id_clean, safe='')}"
            
            # Get metadata from lookup map (O(1) lookup)
            metadata = metadata_map.get(fact_id_uri, {})
            
            # Debug: Log if metadata not found (for troubleshooting)
            if not metadata and len(metadata_map) > 0:
                # Try to find a matching fact_id_uri (case-insensitive, partial match)
                matching_uri = None
                for stored_uri in metadata_map.keys():
                    if fact_id_uri.lower() in stored_uri.lower() or stored_uri.lower() in fact_id_uri.lower():
                        matching_uri = stored_uri
                        break
                if matching_uri:
                    # Mismatch handled by fallback lookup - no need to log
                    metadata = metadata_map.get(matching_uri, {})
            
            # Get all sources for this fact
            source_documents = metadata.get('source_documents', [])
            
            # Format sources: if multiple, use the first one for backward compatibility
            # but also include all sources in a new field
            primary_source = source_documents[0][0] if source_documents and source_documents[0][0] else None
            primary_timestamp = source_documents[0][1] if source_documents and source_documents[0][1] else None
            
            # Format all sources as a list
            all_sources = []
            for source_doc, timestamp in source_documents:
                if source_doc:  # Only include entries with actual source documents
                    all_sources.append({
                        "document": source_doc,
                        "uploadedAt": timestamp if timestamp else None
                    })
            
            # Determine type from is_inferred status
            # Handle both boolean and string "true"/"false" values
            is_inferred_val = metadata.get('is_inferred', False)
            if isinstance(is_inferred_val, str):
                is_inferred = is_inferred_val.lower() == 'true'
            else:
                is_inferred = bool(is_inferred_val)
            fact_type = "inferred" if is_inferred else "original"
            
            # Get confidence score
            confidence = metadata.get('confidence', 0.7)  # Default confidence if not found
            
            # Get agent_id
            agent_id = metadata.get('agent_id', None)
            
            # Apply filters
            if not include_inferred and is_inferred:
                continue  # Skip inferred facts if filter is enabled
            if confidence < min_confidence:
                continue  # Skip facts below confidence threshold
            
            facts.append({
                "id": str(fact_index),
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph",
                "details": metadata.get('details') if metadata.get('details') else None,
                "sourceDocument": primary_source,  # Backward compatibility: first source
                "uploadedAt": primary_timestamp,  # Backward compatibility: first timestamp
                "sourceDocuments": all_sources if all_sources else None,  # New: all sources
                "isInferred": is_inferred,  # Backward compatibility: marks if fact is inferred (boolean)
                "type": fact_type,  # Primary field: type of fact ("original" or "inferred")
                "confidence": confidence,  # New: confidence score (0.0 to 1.0)
                "agentId": agent_id  # New: ID of worker agent that extracted this fact
            })
        
        print(f"✅ GET /api/knowledge/facts: Returning {len(facts)} facts")
        if len(facts) > 0:
            print(f"   Sample fact: {facts[0]}")
        else:
            print("   ⚠️  No facts in graph!")
            # Debug: Check if file exists
            if os.path.exists("knowledge_graph.pkl"):
                file_size = os.path.getsize("knowledge_graph.pkl")
                print(f"   📁 knowledge_graph.pkl exists ({file_size} bytes) but graph is empty!")
                # If file exists but no facts, try to see what's in the graph
                all_triples = list(kb_graph)
                print(f"   📊 Total triples in graph: {len(all_triples)}")
                if len(all_triples) > 0:
                    print(f"   📊 Sample triple: {all_triples[0]}")
        
        # CRITICAL: Return facts in the format the frontend expects
        # Frontend expects: { success: true, data: { facts: [...] } }
        # But FastAPI returns directly, so we need to ensure the response has the right structure
        response = {
            "facts": facts,
            "total_facts": len(facts),  # Use len(facts) not len(kb_graph) since we filter metadata
            "status": "success"
        }
        return response
    except Exception as e:
        print(f"❌ Error getting facts: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting facts: {str(e)}")

@app.get("/api/documents")
async def get_documents_endpoint(include_all: bool = False):
    """Get all uploaded documents
    
    Args:
        include_all: If True, return all documents. If False (default), only return documents that contributed facts.
    """
    try:
        # FIRST: ALWAYS clean up documents without facts before returning
        # This ensures documents with 0 facts are PERMANENTLY removed
        cleanup_documents_without_facts()
        
        all_documents = get_all_documents()
        
        # DOUBLE CHECK: Filter out any documents with facts_extracted = 0
        # This is a safety net in case cleanup didn't catch everything
        all_documents = [doc for doc in all_documents if doc.get('facts_extracted', 0) > 0]
        
        # Filter: ONLY return documents that have contributed facts (facts_extracted > 0)
        # This ensures we NEVER show documents without facts
        if not include_all:
            documents = all_documents  # Already filtered above
            print(f"✅ GET /api/documents: Returning {len(documents)} documents with facts")
        else:
            documents = all_documents
            print(f"✅ GET /api/documents: Returning {len(documents)} documents (all)")
        
        # Debug: Log agent_id for each document
        for doc in documents:
            agent_id = doc.get('agent_id')
            print(f"📄 Document '{doc.get('name')}': agent_id={agent_id}, facts={doc.get('facts_extracted', 0)}")
        
        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_all_documents": len(documents),  # Both are the same now (filtered)
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.delete("/api/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a document by ID"""
    try:
        success = ds_delete_document(document_id)
        if success:
            return {
                "message": "Document deleted successfully",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/api/knowledge/delete")
async def delete_knowledge_endpoint(request: DeleteKnowledgeRequest):
    """Delete knowledge from the graph"""
    try:
        if request.keyword:
            result = kb_delete_all_knowledge()  # You may want to implement keyword-based deletion
        elif request.count:
            from knowledge import delete_recent_knowledge
            result = delete_recent_knowledge(request.count)
        else:
            result = kb_delete_all_knowledge()
        
        kb_save_knowledge_graph()
        return {
            "message": result,
            "status": "success",
            "total_facts": len(kb_graph)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting knowledge: {str(e)}")

@app.delete("/api/knowledge/facts/{fact_id}")
async def delete_fact_endpoint(fact_id: str):
    """Delete a specific fact by ID (subject, predicate, object)"""
    try:
        import rdflib
        from urllib.parse import quote
        from knowledge import fact_exists as kb_fact_exists
        
        # Parse fact_id - it should be in format "subject|predicate|object"
        # Or we can accept it as a JSON string
        try:
            import json
            from urllib.parse import unquote
            # Decode URL encoding first
            decoded_id = unquote(fact_id)
            fact_data = json.loads(decoded_id)
            subject = fact_data.get('subject')
            predicate = fact_data.get('predicate')
            object_val = fact_data.get('object')
        except json.JSONDecodeError:
            # Try parsing as pipe-separated
            parts = fact_id.split('|')
            if len(parts) == 3:
                subject, predicate, object_val = parts
            else:
                # Try to find fact by searching all facts
                # This is a fallback - ideally fact_id should be structured
                raise HTTPException(status_code=400, detail="Invalid fact ID format. Expected JSON or 'subject|predicate|object'")
        
        if not subject or not predicate or object_val is None:
            raise HTTPException(status_code=400, detail="Missing subject, predicate, or object")
        
        # Check if fact exists
        if not kb_fact_exists(subject, predicate, str(object_val)):
            raise HTTPException(status_code=404, detail="Fact not found in knowledge graph")
        
        # Create URI-encoded triple to match how it's stored
        subject_clean = str(subject).strip().replace(' ', '_')
        predicate_clean = str(predicate).strip().replace(' ', '_')
        object_value = str(object_val).strip()
        
        subject_uri = rdflib.URIRef(f"urn:{quote(subject_clean, safe='')}")
        predicate_uri = rdflib.URIRef(f"urn:{quote(predicate_clean, safe='')}")
        object_literal = rdflib.Literal(object_value)
        
        # Remove details first (if any)
        from knowledge import remove_fact_details as kb_remove_fact_details
        kb_remove_fact_details(subject, predicate, object_value)
        
        # Remove from graph
        if (subject_uri, predicate_uri, object_literal) in kb_graph:
            kb_graph.remove((subject_uri, predicate_uri, object_literal))
            kb_save_knowledge_graph()
            
            print(f"✅ DELETE /api/knowledge/facts/{fact_id}: Deleted fact - {subject} {predicate} {object_val}")
            print(f"✅ Graph now has {len(kb_graph)} facts")
            
            return {
                "message": "Fact deleted successfully",
                "status": "success",
                "deleted_fact": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_val
                },
                "total_facts": len(kb_graph)
            }
        else:
            raise HTTPException(status_code=404, detail="Fact not found in knowledge graph")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting fact: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting fact: {str(e)}")

@app.get("/api/export")
async def export_knowledge_endpoint(
    include_inferred: bool = True,
    min_confidence: float = 0.0
):
    """Export all knowledge graph facts as JSON"""
    try:
        from datetime import datetime
        from urllib.parse import unquote
        
        # Reload graph to ensure we have latest facts
        kb_load_knowledge_graph()
        
        # Import get_fact_details function
        from knowledge import get_fact_details as kb_get_fact_details
        
        # Extract facts from graph
        facts = []
        # Import get_fact_source_document function
        from knowledge import get_fact_source_document as kb_get_fact_source_document
        
        for i, (s, p, o) in enumerate(kb_graph):
            # Skip metadata triples (those with special predicates for details, source document, timestamp)
            predicate_str = str(p)
            # Skip if this is a metadata triple (has fact_id_uri as subject or metadata predicate)
            if ('urn:fact:' in str(s) or  # Metadata triple (fact_id_uri as subject)
                'fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str or
                'is_inferred' in predicate_str or 'agent_id' in predicate_str or
                'confidence' in predicate_str):  # Confidence is metadata, not a fact predicate
                continue
            
            # Extract subject from URI - must be urn:entity: format
            subject_uri_str = str(s)
            if 'urn:entity:' in subject_uri_str:
                subject = subject_uri_str.split('urn:entity:')[-1]
                subject = unquote(subject).replace('_', ' ')
            else:
                # Skip if not a proper entity URI (must be urn:entity:)
                continue
            
            # Extract predicate from URI - must be urn:predicate: format
            predicate_uri_str = str(p)
            if 'urn:predicate:' in predicate_uri_str:
                predicate = predicate_uri_str.split('urn:predicate:')[-1]
                predicate = unquote(predicate).replace('_', ' ')
            else:
                # Skip if not a proper predicate URI (must be urn:predicate:)
                continue
            
            # Object is already a literal, just get the string value
            object_val = str(o)
            
            # Get details for this fact
            details = kb_get_fact_details(subject, predicate, object_val)
            
            # Get all source documents and timestamps
            all_sources = kb_get_fact_source_document(subject, predicate, object_val)
            
            # Get inferred status
            from knowledge import get_fact_is_inferred as kb_get_fact_is_inferred
            is_inferred_val = kb_get_fact_is_inferred(subject, predicate, object_val)
            # Handle None (not set) as False (original)
            is_inferred = bool(is_inferred_val) if is_inferred_val is not None else False
            
            # Get confidence score
            from knowledge import get_fact_confidence as kb_get_fact_confidence
            confidence = kb_get_fact_confidence(subject, predicate, object_val)
            
            # Get agent_id
            from knowledge import get_fact_agent_id as kb_get_fact_agent_id
            agent_id = kb_get_fact_agent_id(subject, predicate, object_val)
            
            # Determine type from is_inferred status
            fact_type = "inferred" if is_inferred else "original"
            
            # Apply filters
            if not include_inferred and is_inferred:
                continue  # Skip inferred facts if filter is enabled
            if confidence < min_confidence:
                continue  # Skip facts below confidence threshold
            
            # Format sources
            primary_source = all_sources[0][0] if all_sources and all_sources[0][0] else None
            primary_timestamp = all_sources[0][1] if all_sources and all_sources[0][1] else None
            
            # Format all sources as a list
            source_docs_list = []
            for source_doc, timestamp in all_sources:
                if source_doc:
                    source_docs_list.append({
                        "document": source_doc,
                        "uploadedAt": timestamp if timestamp else None
                    })
            
            facts.append({
                "id": str(i + 1),
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": "knowledge_graph",
                "details": details if details else None,
                "sourceDocument": primary_source,  # Backward compatibility
                "uploadedAt": primary_timestamp,  # Backward compatibility
                "sourceDocuments": source_docs_list if source_docs_list else None,  # All sources
                "isInferred": is_inferred,  # Backward compatibility: marks if fact is inferred (boolean)
                "type": fact_type,  # Primary field: type of fact ("original" or "inferred")
                "confidence": confidence,  # New: confidence score (0.0 to 1.0)
                "agentId": agent_id  # New: ID of worker agent that extracted this fact
            })
        
        # Create export response with metadata
        export_data = {
            "facts": facts,
            "metadata": {
                "version": "1.2.3",
                "totalFacts": len(facts),
                "lastUpdated": datetime.now().isoformat()
            }
        }
        
        print(f"✅ GET /api/export: Exporting {len(facts)} facts")
        return export_data
    except Exception as e:
        print(f"❌ Error exporting knowledge: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exporting knowledge: {str(e)}")

@app.post("/api/knowledge/import")
async def import_json_endpoint(file: UploadFile = File(...)):
    """Import knowledge from JSON file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            facts_before = len(kb_graph)
            result = kb_import_json(tmp_path)
            kb_save_knowledge_graph()
            facts_after = len(kb_graph)
            facts_added = facts_after - facts_before
            
            # Parse the result message to extract added/skipped counts
            import re
            added_match = re.search(r'Imported (\d+)', result)
            skipped_match = re.search(r'skipped (\d+)', result)
            added_count = int(added_match.group(1)) if added_match else facts_added
            skipped_count = int(skipped_match.group(1)) if skipped_match else 0
            
            print(f"✅ POST /api/knowledge/import: Added {added_count} facts, skipped {skipped_count} duplicates")
            
            return {
                "message": result,
                "status": "success",
                "total_facts": len(kb_graph),
                "facts_added": added_count,
                "facts_skipped": skipped_count
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing JSON: {str(e)}")

@app.get("/api/knowledge/stats")
async def get_stats_endpoint():
    """Get knowledge graph statistics"""
    try:
        return {
            "total_facts": len(kb_graph),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/api/knowledge/save")
async def save_knowledge_endpoint():
    """Manually trigger knowledge graph save"""
    try:
        result = kb_save_knowledge_graph()
        return {
            "message": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving knowledge: {str(e)}")

# ============================================================================
# AGENT SYSTEM ENDPOINTS
# ============================================================================

@app.get("/api/agents")
async def get_agents_endpoint():
    """Get all agents (orchestrators, LLM, workers)"""
    try:
        agents = get_all_agents()
        return {
            "agents": agents,
            "total_agents": len(agents),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agents: {str(e)}")

@app.get("/api/agents/architecture")
async def get_agent_architecture_endpoint():
    """Get agent architecture for visualization"""
    try:
        # Ensure core agents are initialized
        from agent_system import initialize_agents, document_agents, agents_store
        initialize_agents(clear_document_agents=False)  # Preserve document agents
        
        # Debug: Log document agents
        print(f"🔍 Architecture endpoint: Found {len(document_agents)} document agents")
        for doc_id, doc_agent in document_agents.items():
            print(f"   - {doc_id}: {doc_agent.document_name} (status: {doc_agent.status})")
        
        architecture = get_agent_architecture()
        
        # Debug: Log architecture response
        print(f"📊 Architecture: {len(architecture.get('orchestrator_agents', []))} orchestrator, {len(architecture.get('statistics_agents', []))} stats, {len(architecture.get('visualization_agents', []))} viz, {len(architecture.get('kg_agents', []))} KG, {len(architecture.get('llm_agents', []))} LLM, {len(architecture.get('document_agents', []))} documents")
        
        return {
            "architecture": architecture,
            "status": "success"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in get_agent_architecture_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting agent architecture: {str(e)}")


@app.get("/api/documents/{document_id}/statistics")
async def get_document_statistics_endpoint(document_id: str):
    """Get statistics for a specific document"""
    try:
        from agent_system import get_document_statistics
        statistics = get_document_statistics(document_id)
        
        if statistics is None:
            raise HTTPException(status_code=404, detail=f"Statistics not found for document: {document_id}")
        
        return {
            "statistics": statistics,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.get("/api/documents/{document_id}/visualizations")
async def get_document_visualizations_endpoint(document_id: str):
    """Get visualizations for a specific document"""
    try:
        from agent_system import get_document_visualizations
        visualizations = get_document_visualizations(document_id)
        
        if visualizations is None:
            raise HTTPException(status_code=404, detail=f"Visualizations not found for document: {document_id}")
        
        return {
            "visualizations": visualizations,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error getting visualizations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting visualizations: {str(e)}")

@app.get("/api/documents/{document_id}/statistics/export")
async def export_document_statistics_endpoint(document_id: str):
    """Export statistics for a specific document as JSON"""
    try:
        from agent_system import get_document_statistics, get_document_visualizations
        from datetime import datetime
        
        statistics = get_document_statistics(document_id)
        visualizations = get_document_visualizations(document_id)
        
        if statistics is None:
            raise HTTPException(status_code=404, detail=f"Statistics not found for document: {document_id}")
        
        # Create comprehensive export data
        export_data = {
            "document_id": document_id,
            "exported_at": datetime.now().isoformat(),
            "statistics": statistics,
            "visualizations": visualizations,
            "metadata": {
                "version": "1.0.0",
                "export_type": "statistics",
                "total_rows": statistics.get("total_rows", 0),
                "total_columns": statistics.get("total_columns", 0),
            }
        }
        
        return export_data
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error exporting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting statistics: {str(e)}")

@app.get("/api/documents/{document_id}/summary")
async def get_document_summary_endpoint(document_id: str):
    """Get a summary of a specific document using the LLM agent"""
    try:
        from agent_system import summarize_document
        from documents_store import load_documents
        
        # Get document name
        documents = load_documents()
        document = next((d for d in documents if d.get('id') == document_id or d.get('name') == document_id), None)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        document_name = document.get('name', document_id)
        summary = summarize_document(document_id, document_name)
        
        if summary is None:
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        
        return {
            "document_id": document_id,
            "document_name": document_name,
            "summary": summary,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error getting document summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document summary: {str(e)}")

@app.get("/api/agents/{agent_id}")
async def get_agent_endpoint(agent_id: str):
    """Get a specific agent by ID"""
    try:
        agent = get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        return {
            "agent": agent,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8001 (matches frontend default)
    port = int(os.getenv("API_PORT", 8001))
    host = os.getenv("API_HOST", "0.0.0.0")  # Bind to all interfaces for external access
    
    # Check if the requested port is available, kill existing processes if needed
    import socket
    import subprocess
    
    def is_port_available(check_port):
        """Check if a port is available by trying to bind to it"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', check_port))
            sock.close()
            return True
        except OSError:
            return False
    
    def kill_processes_on_port(check_port):
        """Kill processes using the specified port"""
        try:
            # Try to find processes using the port (works on macOS/Linux)
            result = subprocess.run(
                ['lsof', '-ti', f':{check_port}'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"⚠️  Port {check_port} is in use by process(es): {', '.join(pids)}")
                print(f"🔧 Killing existing process(es)...")
                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], check=False, timeout=1)
                    except:
                        pass
                import time
                time.sleep(1)  # Wait for processes to terminate
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # lsof might not be available or command failed
            pass
        return False
    
    # Try to free the requested port first
    if not is_port_available(port):
        kill_processes_on_port(port)
        # Check again after killing
        if not is_port_available(port):
            print(f"⚠️  Port {port} is still busy, trying alternatives...")
        for attempt_port in [8001, 8002, 8003, 8004]:
            if attempt_port != port and is_port_available(attempt_port):
                port = attempt_port
                print(f"✅ Using port {port} instead")
                break
        else:
            print(f"⚠️  Warning: Could not find available port, using {port} anyway (may fail if busy)")
    
    print(f"Starting NesyX API server on http://{host}:{port}")
    print(f"API documentation available at http://localhost:{port}/docs")
    print(f"Frontend should connect to: http://localhost:{port}")
    
    uvicorn.run(app, host=host, port=port)

