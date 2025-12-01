# Research Brain - Knowledge Management System

Multi-agent knowledge graph system for extracting, storing, and visualizing facts from documents.

## Quick Start

```bash
# Start backend (port 8002)
./start_backend.sh

# Start frontend (port 5173)
./start_frontend.sh

# Or start both
./start_all.sh
```

## Architecture

### Backend (Python/FastAPI)
- **`api_server.py`** - Main FastAPI server, REST API endpoints
- **`agent_system.py`** - Multi-agent orchestration (Task Allocator, KG Combiner, LLM Agent, Workers)
- **`knowledge.py`** - Knowledge graph management (RDFLib), fact extraction, storage
- **`file_processing.py`** - Document text extraction (PDF, DOCX, TXT, CSV)
- **`documents_store.py`** - Document metadata management
- **`kg_pipeline.py`** - Knowledge graph processing pipeline
- **`responses.py`** - Chat/query response generation

### Frontend (React/TypeScript)
- **`RandDKnowledgeGraph/client/src/`** - React app
  - **`pages/`** - Main pages (upload, knowledge-base, graph, agents, chat)
  - **`components/`** - UI components (tables, visualizations, dialogs)
  - **`lib/api-client.ts`** - API client for backend communication

## Core Functionalities

### 1. Document Upload & Processing
- **Location**: `api_server.py` → `/api/knowledge/upload`
- **Flow**: Upload → Extract text → Allocate to worker agent → Extract facts → Store in KG
- **Agents**: Task Allocator assigns documents to Worker Agents

### 2. Knowledge Graph Management
- **Location**: `knowledge.py`
- **Features**: Fact extraction (Triplex LLM + regex), entity normalization, confidence scoring
- **Storage**: RDFLib graph → `knowledge_graph.pkl`

### 3. Multi-Agent System
- **Location**: `agent_system.py`
- **Agents**:
  - **Task Allocator**: Assigns documents to workers
  - **KG Combiner**: Combines worker graphs
  - **LLM Agent**: Generates intelligent responses to queries
  - **Worker Agents**: One per document, separate knowledge graphs


### 5. Knowledge Base View
- **Location**: Frontend `pages/knowledge-base.tsx`, Backend `/api/knowledge/facts`
- **Features**: Table view, filtering (inferred/original, confidence), agent ownership

### 6. Graph Visualization
- **Location**: Frontend `pages/graph.tsx`, `components/KnowledgeGraphVisualization.tsx`
- **Features**: Interactive graph, node/edge editing, force-directed layout

### 7. Agent Architecture
- **Location**: Frontend `pages/agents.tsx`, Backend `/api/agents/architecture`
- **Features**: Visualize orchestrators, LLM agent, worker agents

### 8. Chat/Query Interface
- **Location**: Frontend `pages/chat.tsx`, Backend `responses.py`
- **Features**: Natural language queries, context retrieval from KG

### 9. Import/Export
- **Location**: Frontend `pages/import-export.tsx`, Backend `/api/export`
- **Features**: Export facts as JSON, import support

## Data Flow

1. **Upload** → `file_processing.py` extracts text
2. **Allocation** → `agent_system.py` creates worker agent
3. **Extraction** → `knowledge.py` extracts facts (with agent_id)
4. **Storage** → Facts stored in RDFLib graph + worker's separate graph
5. **Storage** → Facts stored in knowledge graph
6. **Combination** → KG Combiner aggregates worker graphs
7. **Visualization** → Frontend displays facts/graph

## Key Files

- **Backend Entry**: `api_server.py`
- **Agent Logic**: `agent_system.py`
- **KG Core**: `knowledge.py`
- **Frontend Entry**: `RandDKnowledgeGraph/client/src/main.tsx`
- **API Client**: `RandDKnowledgeGraph/client/src/lib/api-client.ts`

## Configuration

- Backend port: `8002` (env: `API_PORT`)
- Frontend port: `5173` (env: `PORT`)
- Agents storage: `agents_store.json`
- KG storage: `knowledge_graph.pkl`