# Operational & Strategic Query Routing & Functionality

## Overview
This document explains how operational and strategic queries are detected, routed, and processed in the system.

---

## 1. Query Detection Flow

### Entry Point: `responses.py` → `respond()`
When a user asks a question via the LLM chat interface:
1. `respond()` is called with the user's message
2. It first tries **orchestrator-based query processing** (structured/strategic queries)
3. If that fails, falls back to **LLM with context retrieval** (general queries)

### Detection: `query_processor.py` → `detect_query_type()`
- Checks if query matches **strategic/operational patterns** first
- Calls `strategic_queries.py` → `detect_strategic_query()` to identify:
  - **S1, S2** → Strategic queries (3-variable combinations)
  - **O1, O2, O3, O4** → Operational queries (2-variable combinations)
- If no strategic match, checks for **structured queries** (max/min/filter)
- Otherwise, returns `query_type: "general"`

### Detection Patterns: `strategic_queries.py` → `detect_strategic_query()`

#### Strategic Queries (S1, S2):
- **S1**: Performance–Engagement–Status combinations
  - Patterns: "early-warning risk clusters", "high performance low engagement termination"
  - Variables: `["PerformanceScore", "EngagementSurvey", "EmploymentStatus"]`
  - Subtypes: S1.1 (early-warning clusters), S1.2 (active high performers at risk)

- **S2**: Recruitment–Performance–Retention
  - Patterns: "rank recruitment channels", "underperforming recruitment sources"
  - Variables: `["RecruitmentSource", "PerformanceScore", "EmploymentStatus"]`
  - Subtypes: S2.1 (strategic ranking), S2.2 (underperforming sources)

#### Operational Queries (O1, O2, O3, O4):
- **O1**: Performance × Department
  - Patterns: "departmental performance monitoring", "tracking low-performance concentrations"
  - Variables: `["PerformanceScore", "Department"]`
  - Subtypes: O1.1 (departmental monitoring), O1.2 (low-performance tracking)

- **O2**: Absences × EmploymentStatus
  - Patterns: "absence patterns by employment status", "absences distribution termination"
  - Variables: `["Absences", "EmploymentStatus"]`
  - Subtype: O2.1 (absence patterns)

- **O3**: Engagement × Manager
  - Patterns: "team-level engagement monitoring", "manager engagement distribution"
  - Variables: `["EngagementSurvey", "ManagerName"]`
  - Subtype: O3.1 (team-level monitoring)
  - **Note**: Patterns are specific to avoid matching basic "who is the manager" queries

- **O4**: Average per Group with Min/Max
  - Patterns: "average performance per manager and return min", "average absences per department return max"
  - Variables: Dynamic (metric + group_by from query)
  - Subtype: O4.1 (average aggregation)

---

## 2. Routing Flow

### Orchestrator: `orchestrator.py` → `orchestrate_query()`

When a strategic/operational query is detected:

1. **Orchestrator receives query_info** with:
   - `query_type: "strategic"`
   - `strategic_type: "S1" | "S2" | "O1" | "O2" | "O3" | "O4"`
   - `subtype: "S1.1" | "O1.1"`, etc.
   - `variables: [...]` (column names)

2. **Routing Decision**:
   ```python
   if strategic_type in ["S1", "S2"]:
       → Route to Strategic Query Agent
   elif strategic_type in ["O1", "O2", "O3", "O4"]:
       → Route to Operational Query Agent
   ```

3. **Agent Processing**:
   - Sets agent status to `"processing"`
   - Calls agent's main processing function
   - Sets agent status to `"active"`
   - Returns `(answer, evidence_facts, routing_info)`

---

## 3. Processing Flow

### Strategic Query Agent: `strategic_query_agent.py`

**Main Function**: `process_strategic_query_with_agent(query_info, question)`

**Steps**:
1. **Reconstruct DataFrame from KG**:
   - Calls `reconstruct_dataframe_from_facts()`
   - Iterates through all KG facts
   - Builds DataFrame with employees as rows, attributes as columns
   - Handles metadata skipping for performance

2. **Normalize Column Names**:
   - Uses `normalize_column_name(df, target_name)` to find actual column
   - Handles variations: "PerformanceScore" vs "PerfScoreID"
   - Uses fuzzy matching and aliases

3. **Route to Analysis Function**:
   ```python
   if strategic_type == "S1":
       if subtype == "S1.1":
           → _analyze_s1_1(df, columns, query_info)
       elif subtype == "S1.2":
           → _analyze_s1_2(df, columns, query_info)
   elif strategic_type == "S2":
       if subtype == "S2.1":
           → _analyze_s2_1(df, columns, query_info)
       elif subtype == "S2.2":
           → _analyze_s2_2(df, columns, query_info)
   ```

4. **Analysis Functions** (in `strategic_queries.py`):
   - **`_analyze_s1_1`**: Early-warning risk clusters
     - Filters: high performance + low engagement + high termination
     - Groups by department, role, tenure
     - Returns clusters with statistics
   
   - **`_analyze_s1_2`**: Active high performers at risk
     - Filters: active employees + high performance + declining engagement
     - Groups by department, manager, role
     - Flags retention risk clusters
   
   - **`_analyze_s2_1`**: Strategic ranking of recruitment channels
     - Groups by RecruitmentSource
     - Computes joint distribution: performance × employment status
     - Ranks channels by high performance + sustained active status
   
   - **`_analyze_s2_2`**: Underperforming recruitment sources
     - Identifies sources with low performance + early turnover
     - Provides quantitative evidence and recommendations

5. **Store Insights in KG**:
   - Adds facts like: `"Strategic Insights" → "has_early_warning_cluster" → "Department: Sales, Role: Technician"`
   - Makes insights accessible to LLM via `retrieve_context()`

6. **Return Results**:
   - Formatted answer (Markdown)
   - Evidence facts (for traceability)
   - Routing info (for UI display)

---

### Operational Query Agent: `operational_query_agent.py`

**Main Function**: `process_operational_query_with_agent(query_info, question)`

**Steps**:
1. **Reconstruct DataFrame from KG** (same as Strategic Agent)

2. **Normalize Column Names** (same as Strategic Agent)

3. **Route to Analysis Function**:
   ```python
   if strategic_type == "O1":
       if subtype == "O1.1":
           → process_o1_1(df, columns, query_info)
       elif subtype == "O1.2":
           → process_o1_2(df, columns, query_info)
   elif strategic_type == "O2":
       → process_o2_1(df, columns, query_info)
   elif strategic_type == "O3":
       → process_o3_1(df, columns, query_info)
   elif strategic_type == "O4":
       → process_o4_1(df, columns, query_info)
   ```

4. **Analysis Functions** (in `strategic_queries.py`):
   - **`process_o1_1`**: Departmental performance monitoring
     - Groups by Department
     - Computes performance score distribution
     - Highlights departments with deteriorating scores
   
   - **`process_o1_2`**: Tracking low-performance concentrations
     - Groups by Department
     - Computes proportion of "Needs Improvement" / "PIP" employees
     - Triggers alerts when proportions exceed thresholds
   
   - **`process_o2_1`**: Absence patterns by employment status
     - Groups by EmploymentStatus (Active vs Terminated)
     - Computes average absences per group
     - Identifies patterns associated with termination
   
   - **`process_o3_1`**: Team-level engagement monitoring
     - Groups by ManagerName
     - Computes average engagement survey scores per manager
     - Identifies managers with low/declining engagement
   
   - **`process_o4_1`**: Average per group with min/max
     - Groups by specified column (manager, department, etc.)
     - Computes average of metric per group
     - Returns min/max group based on aggregate_op

5. **Store Insights in KG** (same as Strategic Agent)

6. **Return Results** (same as Strategic Agent)

---

## 4. Key Files & Functions

### Detection & Routing:
- **`query_processor.py`**: `detect_query_type()` - First-level detection
- **`strategic_queries.py`**: `detect_strategic_query()` - Strategic/operational pattern matching
- **`orchestrator.py`**: `orchestrate_query()` - Routes to appropriate agent

### Processing:
- **`strategic_query_agent.py`**: 
  - `process_strategic_query_with_agent()` - Main entry point
  - `reconstruct_dataframe_from_facts()` - KG → DataFrame
  - `normalize_column_name()` - Column name matching

- **`operational_query_agent.py`**:
  - `process_operational_query_with_agent()` - Main entry point
  - Reuses `reconstruct_dataframe_from_facts()` and `normalize_column_name()` from strategic agent

### Analysis:
- **`strategic_queries.py`**: All analysis functions
  - Strategic: `_analyze_s1_1()`, `_analyze_s1_2()`, `_analyze_s2_1()`, `_analyze_s2_2()`
  - Operational: `process_o1_1()`, `process_o1_2()`, `process_o2_1()`, `process_o3_1()`, `process_o4_1()`

### Storage:
- **`knowledge.py`**: 
  - `add_to_graph()` - Stores facts and insights
  - `retrieve_context()` - Retrieves relevant facts for LLM (now limited to prevent timeout)

---

## 5. Example Flow: O4 Query

**User Query**: "get the average performance score per manager and bring me the min"

1. **Detection** (`query_processor.py`):
   - `detect_query_type()` → calls `detect_strategic_query()`
   - Matches O4 pattern: `r'(?:average|avg|mean).*?(?:performance|score).*?(?:per|by).*?(?:manager).*?(?:and|then|return).*?(?:min)'`
   - Returns: `{query_type: "strategic", strategic_type: "O4", subtype: "O4.1", metric: "performance score", group_by: "manager", aggregate_op: "min"}`

2. **Routing** (`orchestrator.py`):
   - `orchestrate_query()` receives query_info
   - Detects `strategic_type == "O4"`
   - Routes to Operational Query Agent

3. **Processing** (`operational_query_agent.py`):
   - `process_operational_query_with_agent()` called
   - Reconstructs DataFrame from KG
   - Normalizes columns: "performance score" → "PerformanceScore" or "PerfScoreID"
   - Normalizes columns: "manager" → "ManagerName"
   - Calls `process_o4_1(df, columns, query_info)`

4. **Analysis** (`strategic_queries.py`):
   - `process_o4_1()`:
     - Groups by ManagerName
     - Computes average PerformanceScore per manager
     - Finds minimum average value
     - Returns manager with minimum average
     - Formats as Markdown table

5. **Storage**:
   - Stores insight: `"Average PerformanceScore per ManagerName" → "has_min_average_value" → "45.23"`
   - Stores insight: `"Average PerformanceScore per ManagerName" → "has_min_average_group" → "John Smith"`

6. **Response**:
   - Returns formatted answer with table
   - LLM can later access these insights via `retrieve_context()`

---

## 6. LLM Access to Insights

When a user asks a general question that relates to insights:
1. `retrieve_context()` is called with the question
2. It searches KG for relevant facts
3. **Insights are boosted** (score += 10) if they match keywords like "operational", "strategic", "analysis"
4. Top N facts (limited to prevent timeout) are returned as context
5. LLM uses this context to answer the question

**Example**: User asks "which managers have low engagement?"
- `retrieve_context()` finds facts from O3.1 analysis
- Returns: `"Strategic Insights → has_low_engagement_manager → John Smith (avg: 2.3)"`
- LLM uses this to provide answer

---

## 7. Performance Optimizations

### For Large Documents (1000+ rows):
1. **Context Limiting** (`knowledge.py`):
   - `retrieve_context()` now limits results based on graph size:
     - > 5000 facts: limit to 100 most relevant
     - > 2000 facts: limit to 150 most relevant
     - Otherwise: limit to 200 most relevant
   - Prevents LLM timeout

2. **DataFrame Reconstruction**:
   - Skips metadata triples early (optimized pattern matching)
   - Progress logging for large graphs
   - Processes facts in batches

3. **Column Normalization**:
   - Uses fuzzy matching and aliases
   - Handles variations: "PerformanceScore" vs "PerfScoreID"

---

## Summary

**Detection**: `query_processor.py` → `strategic_queries.py` → Pattern matching  
**Routing**: `orchestrator.py` → Routes S1/S2 to Strategic Agent, O1-O4 to Operational Agent  
**Processing**: Agent reconstructs DataFrame from KG → Normalizes columns → Calls analysis function  
**Analysis**: `strategic_queries.py` → Performs pandas operations → Returns formatted results  
**Storage**: Insights stored in KG → Accessible to LLM via `retrieve_context()`  
**Optimization**: Context limiting prevents LLM timeout for large documents

