# Multi-Agent Knowledge Graph Architecture

## Overview
A multi-agent system for processing documents and building a knowledge graph, with intelligent research assistant capabilities.

---

## 🏗️ Architecture Components

### 1. **Core Agents** (Always Active)

#### **Task Allocator** (`task_allocator`)
- **Role**: Orchestrator for document allocation
- **Responsibility**: Assigns documents to worker agents
- **Behavior**: Creates a new worker agent for each uploaded document
- **Location**: `agent_system.py` → `allocate_document_to_worker()`

#### **Knowledge Graph Combiner** (`kg_combiner`)
- **Role**: Orchestrator for knowledge graph combination
- **Responsibility**: Combines facts from all worker agents into the main graph
- **Status**: Currently all workers add directly to main graph (placeholder for future distributed processing)
- **Location**: `agent_system.py` → `combine_knowledge_graphs()`

#### **LLM Agent** (`llm_agent`)
- **Role**: LLM-powered research assistant
- **Responsibility**: 
  - Generates intelligent responses to user queries
  - Retrieves relevant facts from knowledge graph
  - Provides context-aware insights
- **Location**: `responses.py` → `respond()`

---

### 2. **Worker Agents** (Created Dynamically)

#### **Worker Agent** (`worker_{document_id}`)
- **Created**: One per uploaded document
- **Lifecycle**: 
  - Created when document is uploaded
  - Processes document and extracts facts
  - Cleared on server shutdown (ephemeral)
- **Features**:
  - Has its own separate knowledge graph (`separate_graph`)
  - Adds facts to both its graph AND main graph
  - Tracks facts extracted count
  - Stores document metadata
- **Location**: `agent_system.py` → `WorkerAgent` class, `process_document_with_worker()`

---

## 📊 Data Flow

### Document Upload Flow

```
1. User uploads file → api_server.py
   ↓
2. Extract text from file (PDF/DOCX/TXT/CSV)
   → file_processing.py
   ↓
3. Task Allocator assigns document to worker
   → allocate_document_to_worker()
   → Creates new WorkerAgent
   ↓
4. Worker processes document
   → process_document_with_worker()
   → Extracts text → knowledge extraction pipeline
   → Adds facts to main graph (with agent_id metadata)
   → Copies facts to worker's separate_graph
   ↓
5. Save document metadata
   → add_document() with agent_id
   → documents_store.json
   ↓
6. Return response to frontend
   → Includes document with agent_id
```

### Knowledge Extraction Pipeline

```
Text → kg_pipeline.py
  ↓
1. Preprocessing (sentences, POS tags)
  ↓
2. Named Entity Recognition (NER)
  ↓
3. Coreference Resolution
  ↓
4. Relation Extraction
  ↓
5. Entity Linking
  ↓
6. Knowledge Graph Construction (RDFLib)
  ↓
7. Post-processing / Reasoning (inference)
  ↓
Facts stored in knowledge_graph.pkl
```

---

## 🗄️ Data Storage

### **Knowledge Graph** (`knowledge_graph.pkl`)
- **Format**: RDFLib Graph (pickle)
- **Content**: All facts (triples: subject, predicate, object)
- **Metadata**: Stored as separate triples
  - `source_document`: Which document the fact came from
  - `agent_id`: Which worker agent extracted it
  - `confidence`: Confidence score
  - `is_inferred`: Whether fact was inferred
  - `has_details`: Additional details
- **Location**: `knowledge.py` → `graph` variable

### **Documents Store** (`documents_store.json`)
- **Format**: JSON
- **Content**: Document metadata
  - `name`, `size`, `type`
  - `facts_extracted`: Number of facts
  - `agent_id`: Worker agent that processed it
  - `uploaded_at`: Timestamp
- **Location**: `documents_store.py`

### **Agents Store** (`agents_store.json`)
- **Format**: JSON
- **Content**: Core agents only (orchestrators + LLM agent)
- **Note**: Worker agents are NOT persisted (ephemeral)
- **Location**: `agent_system.py`


---

## 🤖 Research Assistant (LLM)

### **LLM Options** (Priority Order)
1. **Ollama** (Default for Mac)
   - Model: `gemma2:2b` (lightweight, good quality)
   - URL: `http://localhost:11434`
   - Environment: `USE_OLLAMA=true`

2. **OpenAI API** (Optional)
   - Model: `gpt-4o-mini`
   - Requires: `USE_OPENAI=true`, `OPENAI_API_KEY`
   
3. **Local HuggingFace** (Fallback)
   - Model: `Qwen/Qwen2.5-0.5B-Instruct`
   - Requires: `transformers` library

### **Response Generation**
- Retrieves relevant facts from knowledge graph
- Uses LLM to generate intelligent, insight-focused responses
- Filters out agent ownership info from responses
- Focuses on actionable insights, not raw data dumps
- **Location**: `responses.py` → `respond()`

---

## 🔄 Key Processes


### **Fact Extraction**
- Uses knowledge extraction pipeline (`kg_pipeline.py`)
- Extracts triples: (subject, predicate, object)
- Adds metadata: source document, agent_id, confidence
- Stores in RDFLib graph
- **Location**: `knowledge.py` → `add_to_graph()`

### **Agent Lifecycle**
- **Startup**: Core agents initialized, worker agents cleared
- **Document Upload**: New worker agent created per document
- **Shutdown**: Worker agents cleared, core agents saved
- **Location**: `api_server.py` → `lifespan()` function

---

## 🌐 API Endpoints

### **Document Management**
- `POST /api/knowledge/upload` - Upload and process files
- `GET /api/documents` - Get all documents
- `DELETE /api/documents/{id}` - Delete document

### **Knowledge Base**
- `GET /api/knowledge/facts` - Get all facts
- `POST /api/knowledge/facts` - Create fact manually
- `DELETE /api/knowledge/facts/{id}` - Delete fact
- `GET /api/export` - Export knowledge base as JSON

### **Agents**
- `GET /api/agents/architecture` - Get agent architecture
- `GET /api/agents/{agent_id}` - Get specific agent

### **Research Assistant**
- `POST /api/chat` - Ask questions, get intelligent responses

---

## 📁 File Structure

```
/Users/s20/Enesy-Dev/
├── api_server.py          # Main FastAPI backend
├── agent_system.py        # Multi-agent system logic
├── knowledge.py           # Knowledge graph management
├── kg_pipeline.py         # Knowledge extraction pipeline
├── file_processing.py     # Document text extraction
├── documents_store.py     # Document metadata management
├── responses.py           # LLM/research assistant
├── csv_analysis.py        # CSV analysis (legacy, not used)
├── csv_to_facts.py        # CSV to facts (legacy, not used)
└── RandDKnowledgeGraph/   # React frontend
    └── client/
        └── src/
            ├── pages/
            │   └── agents.tsx      # Agent architecture visualization
            └── components/
                ├── DocumentList.tsx      # Shows documents with agent_id
                └── KnowledgeBaseTable.tsx # Shows facts with agentId
```

---

## 🔍 Key Features

### **Multi-Agent Processing**
- Each document gets its own worker agent
- Agents track their own facts
- Main graph combines all facts
- Agent ownership tracked via `agent_id` metadata


### **Intelligent Responses**
- LLM-powered research assistant
- Context-aware fact retrieval
- Insight-focused, not data dumps
- Filters agent metadata from responses

### **Ephemeral Workers**
- Worker agents created per document
- Cleared on server restart
- Core agents persist across restarts

---

## 🐛 Recent Fixes

1. **Document Agent Assignment**: Fixed fallback case where documents might be saved without `agent_id`
2. **CSV Processing**: Reverted to simple text extraction (no special analysis)
3. **LLM Integration**: Ollama support for Mac, OpenAI API option
4. **Error Handling**: Robust error handling throughout pipeline

---

## 🚀 Running the System

### **Backend**
```bash
python3 api_server.py
# Runs on http://localhost:8002
```

### **Frontend**
```bash
cd RandDKnowledgeGraph/client
npm run dev
# Runs on http://localhost:5173
```

### **Environment Variables**
- `USE_OLLAMA=true` - Enable Ollama (default)
- `OLLAMA_MODEL=gemma2:2b` - Ollama model
- `USE_OPENAI=false` - Enable OpenAI API
- `OPENAI_API_KEY=...` - OpenAI API key
- `API_PORT=8002` - Backend port
- `PORT=5173` - Frontend port

---

## 📝 Summary

**Architecture Type**: Multi-Agent System with Orchestrators

**Core Concept**: 
- 3 core agents (Task Allocator, KG Combiner, LLM Agent)
- Dynamic worker agents (one per document)
- Shared knowledge graph with agent ownership tracking
- LLM-powered research assistant

**Data Flow**: Document → Worker Agent → Knowledge Graph → Research Assistant

**Key Innovation**: Each document processed by dedicated worker agent, enabling traceability and future distributed processing.

