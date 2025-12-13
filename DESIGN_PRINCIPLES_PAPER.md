# Design Principles for Human-Centric AI-Powered Information Systems

## Abstract

This section describes the design principles that guide our multi-agent knowledge graph system for HR decision support. These principles align with requirements for transparent, ethical, and human-centered AI systems in information access contexts, addressing concerns about explainability, traceability, fairness, accountability, and AI governance raised in recent Information Systems research (Mitra, 2025; Stray, 2024; Bernard, 2025).

---

## 1. Explainability: Making AI Reasoning Transparent

### Principle Statement
Every system response must be explainable through explicit evidence facts that users can inspect, verify, and understand. The system avoids black-box decision-making by making all reasoning steps visible and traceable.

### Rationale
Explainability is fundamental to building trust in AI systems, especially in sensitive domains like HR management where decisions affect people's careers and livelihoods. Research in Information Systems has emphasized the need for explainable AI systems that enable users to understand how conclusions were reached (Mitra, 2025). Our system implements explainability through a multi-layered approach: (1) evidence-based responses that include supporting facts from the knowledge graph, (2) provenance chains that enable users to trace information from responses back to source documents, and (3) routing transparency that shows which agents were selected and why.

### System Implementation

**Evidence Assembly Component** (`query_processor.py`, lines 776-821): The `build_evidence_context()` function assembles evidence facts from knowledge graph retrieval, formats them as Subject-Predicate-Object triples with source attribution, and includes them in every response. This ensures that users can see exactly which facts support each system conclusion.

**Response Generation Component** (`responses.py`, lines 1000-1024): The `respond()` function ensures all responses include evidence sections that cite specific facts from the knowledge graph. The LLM is explicitly prompted to cite facts and include source documents in its responses, preventing hallucination and ensuring grounding in the knowledge graph.

**Knowledge Base Display Component** (Frontend `KnowledgeBaseTable.tsx`): Provides a comprehensive table view where users can inspect all facts, filter by source, confidence, and agent, and see complete provenance information. This enables users to verify system reasoning by examining the underlying facts.

### Example
When a user asks "What is John Smith's salary?", the system returns:
```
Answer: John Smith has a salary of $75,000.

Evidence from Knowledge Graph:
1. John Smith → has_salary → $75,000 [Source: employees.csv]
```

This example demonstrates how explainability is achieved through explicit evidence display, enabling users to verify the system's reasoning.

---

## 2. Traceability: Complete Provenance Tracking

### Principle Statement
Every fact in the system maintains complete provenance information, enabling users to trace information from source documents through extraction to query responses. This ensures accountability and enables verification of system outputs.

### Rationale
Traceability is essential for accountability in AI systems. In HR contexts, decisions must be auditable and verifiable. Research has emphasized the importance of provenance tracking in AI systems (Storey, 2025). Our system implements traceability through: (1) source attribution where every fact stores its source document(s) and upload timestamp(s), (2) agent ownership where each fact tracks which agent extracted it, enabling accountability for extraction quality, (3) temporal tracking through upload timestamps that enable users to understand when information was added and identify potential data staleness, and (4) multi-source support where facts can have multiple sources, allowing the system to track when the same fact appears in different documents.

### System Implementation

**Fact Metadata Storage Component** (`knowledge.py`, lines 2660-2705): The `add_fact_source_document()` function stores source document and timestamp metadata for each fact in the RDF knowledge graph. It uses RDF triples with special predicates (`urn:source_document`, `urn:uploaded_at`) to link facts to their sources. For example, a fact about "John Smith → has_salary → $75,000" is linked to metadata triples: `(fact_id, urn:source_document, "employees.csv")`, `(fact_id, urn:uploaded_at, "2024-01-15T10:30:00")`, and `(fact_id, urn:agent_id, "worker_001")`.

**Provenance Retrieval Component** (`knowledge.py`, lines 2948-3015): The `get_fact_source_document()` function retrieves all source documents and timestamps for a given fact, supporting multiple sources per fact. It returns a list of tuples `[(source_document, uploaded_at), ...]` for all sources, enabling complete provenance reconstruction.

**API Endpoint for Fact Inspection** (`api_server.py`, lines 1142-1379): The `/api/knowledge/facts` endpoint exposes facts with complete metadata (source documents, timestamps, agent IDs, confidence scores) via REST API. It supports filtering by `min_confidence` and `include_inferred` to enable quality control and provenance-based filtering.

### Example
Each fact in the knowledge graph maintains a complete provenance chain:
- **Source Document**: `employees.csv`
- **Upload Timestamp**: `2024-01-15T10:30:00`
- **Extracting Agent**: `worker_001`
- **Confidence Score**: `0.95`

This provenance information enables users to verify the origin and quality of every piece of information in the system.

---

## 3. Interpretability: Understanding System Decisions

### Principle Statement
Users must be able to understand not just what the system decided, but why it made those decisions. This includes query routing choices, agent selection, and fact retrieval strategies.

### Rationale
Interpretability goes beyond explainability by making the system's decision-making process itself understandable. Research in human-computer interaction has emphasized the importance of interpretable AI systems (Liu, 2024). Our system implements interpretability through: (1) transparent query routing where the orchestrator's routing decisions are logged with explicit reasoning, (2) agent selection visibility where users can see which agents processed their queries and why, and (3) query type detection where the system explicitly identifies query types (structured, operational, statistical, conversational) and explains the routing strategy.

### System Implementation

**Orchestrator Routing Logic Component** (`orchestrator.py`, lines 56-135): The `find_agents_for_query()` function determines which agents should handle a query and returns routing information with explicit reasoning. It returns a dictionary with `strategy`, `target_agents`, and `reason` fields that explain the routing decision. For example, employee-specific queries are routed to specific document workers that processed that employee, statistical queries are routed to the statistics agent, and operational queries are routed to the operational query agent.

**Query Type Detection Component** (`query_processor.py`, lines 19-171): The `detect_query_type()` function analyzes user queries to identify query type and extract relevant parameters. It outputs a structured dictionary with `query_type`, `operation`, `attribute`, and `entity_name` fields, enabling users to see how their query was interpreted and routed.

**Routing Information in Responses** (`orchestrator.py`, lines 138-238): The `orchestrate_query()` function returns routing information alongside query responses, enabling users to understand which agents were involved. Routing info includes strategy, target agents, and reasoning, which can be displayed in the UI for transparency.

### Example
When a user asks "What is the salary of John Smith?", the system returns routing information:
```json
{
  "strategy": "specific_agents",
  "target_agents": ["worker_001"],
  "reason": "Found employee 'John Smith' in 1 agent(s)"
}
```

This routing information enables users to understand how their query was processed and which components of the system were involved.

---

## 4. Fairness: Source-Aware Bias Detection

### Principle Statement
The system tracks data sources to enable bias analysis and ensure fair treatment across different groups. While the system does not automatically detect bias, it provides the infrastructure necessary for users to perform bias audits.

### Rationale
Fairness in AI systems requires the ability to audit outputs for potential biases. In HR contexts, this is critical for ensuring equitable treatment. Research has emphasized the importance of fairness and bias detection in AI systems (Bernard, 2025). Our system supports fairness through: (1) source transparency where complete tracking of which documents contributed which facts enables users to identify potential source-based biases, (2) confidence scoring where facts have confidence scores that can indicate extraction quality, which may correlate with data quality issues, and (3) multi-source tracking where when the same fact appears in multiple sources, the system tracks all sources, enabling users to identify consensus vs. single-source claims.

### System Implementation

**Source Document Tracking Component** (`knowledge.py`): Every fact stores its source document(s), enabling users to filter facts by source and identify potential source-based biases. For example, HR professionals can filter facts by source document to see if certain documents (e.g., performance reviews vs. recruitment data) contribute disproportionately to certain types of insights.

**Confidence Score System Component** (`knowledge.py`, lines 2861-2946): The `add_fact_confidence()` and `get_fact_confidence()` functions store and retrieve confidence scores (0.0 to 1.0) for each fact, indicating extraction quality. Confidence scores are stored as RDF triples with predicate `urn:confidence`. Low confidence scores may indicate extraction uncertainty or data quality issues that could affect fairness.

**Fact Filtering by Quality Component** (`api_server.py`, line 1145): The `/api/knowledge/facts` endpoint supports a `min_confidence` parameter that allows users to filter facts by confidence threshold, enabling quality-based filtering. Users can query only high-confidence facts (confidence > 0.8) to ensure responses are based on reliable data.

### Example
HR professionals can perform bias audits by:
1. Filtering facts by source document to identify potential source-based biases
2. Examining confidence scores to identify low-quality extractions that may affect fairness
3. Comparing facts across multiple sources to identify consensus vs. single-source claims

This infrastructure enables users to perform comprehensive bias analysis and ensure fair treatment across different groups.

---

## 5. Accountability: Agent Ownership and Responsibility

### Principle Statement
Every system action is attributable to a specific agent, enabling accountability for extraction quality, processing decisions, and system behavior. This supports governance and quality control.

### Rationale
Accountability is essential for AI governance. Research has emphasized the importance of accountability in AI systems (Stray, 2024). Our multi-agent architecture enables accountability by: (1) agent ownership where each fact tracks which agent extracted it (`agent_id` metadata), (2) agent status tracking where agents have status fields (`active`, `processing`, `error`) that enable monitoring, and (3) specialized responsibility where each agent has well-defined responsibilities, making it clear which agent is responsible for which types of processing.

### System Implementation

**Agent ID Tracking in Facts Component** (`knowledge.py`): When facts are added to the knowledge graph, the system stores which agent extracted them. Agent IDs are stored as RDF triples with predicate `urn:agent_id`. For example, a fact extracted by worker agent `worker_001` will have metadata: `(fact_id, urn:agent_id, "worker_001")`.

**Agent System Architecture Component** (`agent_system.py`, lines 1-90): Defines eight specialized agents with clear responsibilities: (1) Knowledge Graph Agent (central storage manager), (2) Document Agents (coordinate document processing), (3) Worker Agents (extract facts from document chunks), (4) Statistics Agent (computes correlations and distributions), (5) Operational Query Agent (generates operational insights), (6) LLM Agent (generates natural language responses), (7) Orchestrator Agent (routes queries to appropriate agents), and (8) Visualization Agent (creates charts and graphs). Clear responsibility boundaries enable users to understand which agent is responsible for which system outputs.

**Agent Status Monitoring Component** (`agent_system.py`): Agents have status fields that track their current state. Status can be `active`, `processing`, or `error`, enabling system monitoring and debugging.

### Example
A fact extracted by worker agent `worker_001` has complete accountability information:
- **Agent ID**: `worker_001`
- **Agent Type**: `document_worker`
- **Agent Status**: `active`
- **Fact**: `John Smith → has_salary → $75,000`

This accountability information enables users to identify which agent is responsible for each system output and enables quality control and governance.

---

## 6. AI Governance: Transparent Multi-Agent Orchestration

### Principle Statement
The system implements governance through transparent orchestration, where all agent interactions, query routing decisions, and processing steps are logged and auditable. This enables oversight and control over AI system behavior.

### Rationale
AI governance requires mechanisms for oversight, control, and auditability. Research has emphasized the importance of AI governance in Information Systems (Mitra, 2025). Our system implements governance through: (1) centralized orchestration where the orchestrator agent acts as a central coordinator, making all routing decisions transparent and auditable, (2) explicit routing strategies where routing decisions use explicit strategies (`all_agents`, `specific_agents`, `statistics_agent`, `operational_agent`) that are logged and explainable, (3) agent communication patterns where all agent-to-agent communication flows through the knowledge graph, creating an audit trail, and (4) query logging where routing information is returned with every query, enabling audit trails.

### System Implementation

**Orchestrator Central Coordination Component** (`orchestrator.py`, lines 138-238): The `orchestrate_query()` function acts as a central coordinator that routes queries to appropriate agents and returns routing information. All query routing decisions flow through a single point, enabling centralized governance and auditing. The function returns routing information including strategy, target agents, and reasoning.

**Routing Strategy Enumeration Component** (`orchestrator.py`, lines 56-135): The `find_agents_for_query()` function uses explicit routing strategies that are logged and explainable. Strategies include: `all_agents` (query all document workers for global queries), `specific_agents` (query specific agents that processed a particular employee), `statistics_agent` (route to statistics agent for correlation/distribution queries), and `operational_agent` (route to operational query agent for groupby/aggregation queries). Explicit strategies enable policy enforcement and auditing.

**Knowledge Graph as Audit Trail Component** (`knowledge.py`): All facts are stored in the knowledge graph with complete metadata, creating a permanent audit trail. Every system action (fact extraction, processing, query) leaves a trace in the knowledge graph, enabling comprehensive auditing.

### Example
All query routing decisions flow through the orchestrator, which returns routing information:
```json
{
  "strategy": "operational_agent",
  "target_agents": ["operational_query_agent"],
  "reason": "Routed to operational agent based on keyword detection",
  "query_info": {
    "query_type": "operational",
    "operation": "average",
    "attribute": "salary"
  }
}
```

This routing information enables centralized governance and auditing of all system decisions.

---

## 7. Human-Centricity: Designed for HR Professionals

### Principle Statement
The entire system is designed around the needs and workflows of HR professionals, with intuitive interfaces, conversational interactions, and visual exploration tools that make system capabilities accessible and understandable.

### Rationale
Human-centricity ensures that AI systems serve human needs rather than requiring humans to adapt to system constraints. Research in human-computer interaction has emphasized the importance of human-centric design in AI systems (Luo, 2024). Our system implements human-centricity through: (1) conversational interface where natural language queries enable HR professionals to interact with the system using their domain language, (2) visual knowledge exploration where interactive graph visualization enables users to explore entity relationships visually, (3) evidence visibility where every answer shows supporting facts, enabling users to verify and understand system reasoning, and (4) intuitive navigation where clear page structure (chat, knowledge base, graph, statistics) matches HR professional workflows.

### System Implementation

**Conversational Chat Interface Component** (Frontend `ChatInterface.tsx`): Provides a natural language interface where HR professionals can ask questions in conversational language. Features include message history, loading states, evidence display in responses, and suggested queries. This enables HR professionals to interact with the system using their natural language without learning query syntax.

**Interactive Knowledge Graph Visualization Component** (Frontend `KnowledgeGraphVisualization.tsx`): Force-directed graph layout showing entity relationships visually. Features include node/edge editing, zoom and pan, and node selection and details. This enables visual exploration of relationships, making complex data structures understandable.

**Knowledge Base Table View Component** (Frontend `KnowledgeBaseTable.tsx`): Comprehensive table view of all facts with filtering capabilities. Filters include source, confidence, agent, and inferred/original status. This enables HR professionals to inspect and verify the underlying data.

**Statistics Dashboard Component** (Frontend statistics page): Displays correlations, distributions, and descriptive statistics extracted from the data. This provides insights into data patterns without requiring statistical expertise.

### Example
HR professionals can ask "Which department has the highest turnover?" in natural language and receive answers with supporting evidence:
```
Answer: The Sales department has the highest turnover rate at 15.3%.

Evidence from Knowledge Graph:
1. Sales → has_turnover_rate → 15.3% [Source: hr_analytics.csv]
2. Marketing → has_turnover_rate → 8.7% [Source: hr_analytics.csv]
3. Engineering → has_turnover_rate → 5.2% [Source: hr_analytics.csv]
```

This human-centric design enables HR professionals to use the system effectively without technical training.

---

## 8. Transparency: No Black-Box Decisions

### Principle Statement
The system explicitly avoids black-box decision-making by making all reasoning steps, data sources, and processing decisions visible and inspectable. Users can always understand how the system reached its conclusions.

### Rationale
Transparency is the foundation of trust in AI systems. Research has emphasized the importance of transparency in AI systems (White, 2025). Our system implements transparency through: (1) complete source attribution where every fact and response includes source document attribution, (2) evidence display where all responses include supporting evidence facts, (3) routing visibility where query routing decisions are logged and explainable, and (4) knowledge graph inspection where users can inspect the entire knowledge graph to understand system state.

### System Implementation

**Response Format with Evidence Component** (`query_processor.py`, lines 776-821): The `build_evidence_context()` function formats evidence facts with source attribution for display in responses. This ensures users can see exactly which facts support each response.

**Knowledge Graph as Inspectable State Component** (`knowledge.py`): The entire knowledge graph is accessible via API, enabling users to inspect all stored facts. This enables users to verify system state and understand what information the system has access to.

**API Endpoints for Inspection Component** (`api_server.py`): Multiple endpoints enable programmatic inspection of system state: `/api/knowledge/facts` (get all facts with metadata), `/api/knowledge/graph` (get graph visualization data), and `/api/agents` (get agent status and information).

### Example
Users can inspect the entire knowledge graph via API to verify system state:
```json
{
  "facts": [
    {
      "subject": "John Smith",
      "predicate": "has_salary",
      "object": "$75,000",
      "sourceDocument": "employees.csv",
      "uploadedAt": "2024-01-15T10:30:00",
      "agentId": "worker_001",
      "confidence": 0.95
    }
  ]
}
```

This transparency enables users to verify system state and understand how responses were generated.

---

## Summary: Design Principles in Practice

These eight design principles work together to create a system that is:

1. **Explainable**: Every response includes evidence facts that users can inspect
2. **Traceable**: Complete provenance tracking from source documents to responses
3. **Interpretable**: Routing decisions and system behavior are understandable
4. **Fair**: Source tracking enables bias analysis and quality control
5. **Accountable**: Agent ownership enables responsibility attribution
6. **Governable**: Transparent orchestration enables oversight and control
7. **Human-Centric**: Designed for HR professional workflows and needs
8. **Transparent**: No black-box decisions; all reasoning is visible

These principles align with the requirements for responsible AI systems in information access contexts, as outlined in the special issue on "Human-Centric and Generative AI for Information Systems." The system demonstrates how generative AI can be designed for transparency, ethical responsibility, and effective information retrieval while maintaining human values and ethical considerations alongside algorithmic performance.

The implementation of these principles is not merely theoretical but is embedded throughout the system architecture, from the knowledge graph storage layer through the multi-agent orchestration layer to the user interface layer. Each principle is supported by specific system components with concrete implementations, enabling users to verify, audit, and understand system behavior at every level.

