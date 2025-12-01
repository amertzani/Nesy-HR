# Fact Extraction Logic - Complete Explanation

## Overview

The fact extraction system converts CSV data into a knowledge graph (RDF triples) using parallel processing. Each row becomes multiple facts about an employee, stored as `(subject, predicate, object)` triples.

---

## High-Level Flow

```
CSV File Upload
    ↓
Read CSV into DataFrame (pandas)
    ↓
Split into Chunks (parallel processing)
    ↓
For Each Chunk (Worker Agent):
    ├─ For Each Row:
    │   ├─ Extract Employee Name (subject)
    │   ├─ For Each Column:
    │   │   ├─ Map Column → Predicate
    │   │   ├─ Normalize Value (object)
    │   │   └─ Create Triple: (employee, predicate, value)
    │   ├─ Check for Duplicates (fact_exists)
    │   └─ Add to Knowledge Graph
    └─ Track Progress
    ↓
Save Knowledge Graph to Disk
```

---

## Step-by-Step Process

### 1. **CSV Reading & Preparation**

**File**: `agent_system.py`  
**Function**: `extract_csv_facts_directly_parallel_from_df()` (line 932)

```python
# Read CSV into pandas DataFrame
df = pd.read_csv(file_path, sep=';', encoding='utf-8')

# Find employee name column (e.g., "Employee_Name")
name_col = find_column_with_keywords(['name', 'employee', 'empname'])
```

**Key Points**:
- Automatically detects separator (`;`, `,`, or `\t`)
- Identifies the employee name column for use as the subject

---

### 2. **Adaptive Chunking for Parallel Processing**

**File**: `agent_system.py`  
**Function**: `extract_csv_facts_directly_parallel_from_df()`  
**Lines**: 976-1006

```python
# Calculate optimal chunk size based on data complexity
data_complexity = total_rows × total_cols
target_data_points_per_chunk = 15,000

# Example:
# 1000 rows × 30 columns = 30,000 data points
# Chunk size = 15,000 ÷ 30 = 500 rows per chunk
# Number of chunks = 1000 ÷ 500 = 2 chunks
```

**Why Chunking?**
- Enables parallel processing (multiple workers)
- Balances memory usage
- Optimizes processing speed

**Chunk Size Calculation**:
- **Small files** (< 5,000 data points): 2 workers, ~25-100 rows/chunk
- **Medium files** (5,000-50,000): 4-6 workers, ~20-75 rows/chunk
- **Large files** (> 50,000): 8-12 workers, ~50-150 rows/chunk

---

### 3. **Worker Agent Processing (Per Chunk)**

**File**: `agent_system.py`  
**Function**: `process_chunk()` (nested function inside `extract_csv_facts_directly_parallel_from_df()`)  
**Lines**: 1046-1400

Each chunk is processed by a **Worker Agent** that:

1. **Extracts Employee Name** (Subject):
   ```python
   employee_name = row["Employee_Name"]  # e.g., "Brill, Donna"
   
   # Normalize: Remove quotes
   if employee_name.startswith('"'):
       employee_name = employee_name[1:-1].strip()
   # Result: "Brill, Donna" (no quotes)
   ```

2. **Processes Each Column** (Creates Facts):
   ```python
   for col, val in row.items():
       # Skip NaN values and name column
       if pd.isna(val) or col == name_col:
           continue
       
       # Map column name → predicate
       predicate = map_column_to_predicate(col)
       
       # Normalize value
       normalized_val = normalize_value(val)
       
       # Create fact: (employee, predicate, value)
       fact = (employee_name, predicate, normalized_val)
   ```

---

### 4. **Column → Predicate Mapping**

**File**: `agent_system.py`  
**Function**: `process_chunk()` (inside the column processing loop)  
**Lines**: 1177-1232

The system maps CSV column names to standardized predicates:

| CSV Column | Predicate | Example |
|------------|-----------|---------|
| `Salary` | `"has salary"` | `(Brill, Donna, has salary, 53492)` |
| `ManagerName` | `"has manager name"` | `(Brill, Donna, has manager name, David Stanley)` |
| `ManagerID` | `"has manager id"` | `(Brill, Donna, has manager id, 14)` |
| `Position` | `"has position"` | `(Brill, Donna, has position, Production Technician I)` |
| `PositionID` | `"has position id"` | `(Brill, Donna, has position id, 19)` |
| `Department` | `"works in department"` | `(Brill, Donna, works in department, Production)` |
| `Absences` | `"has absences"` | `(Brill, Donna, has absences, 6)` |
| `PerformanceScore` | `"has performance score"` | `(Brill, Donna, has performance score, Fully Meets)` |

**Mapping Logic**:
```python
if 'salary' in col_lower:
    predicate = "has salary"
elif 'manager' in col_lower and 'name' in col_lower:
    predicate = "has manager name"
elif 'manager' in col_lower and 'id' in col_lower:
    predicate = "has manager id"
elif 'position' in col_lower and 'id' not in col_lower:
    predicate = "has position"
# ... more mappings ...
else:
    predicate = f"has {col}"  # Default: use column name
```

**Special Cases**:
- **Manager columns**: NEVER skipped (even if empty)
- **PositionID vs Position**: Different predicates to avoid duplicates
- **ID columns**: Get specific predicates (`has position id`, `has manager id`)

---

### 5. **Value Normalization**

**File**: `agent_system.py`  
**Function**: `process_chunk()` (inside fact addition logic)  
**Lines**: 1261-1271 (parallel processing)  
**Also in**: `extract_csv_facts_directly()` (line ~1968) and `extract_csv_facts_directly_parallel()` (line ~1757)

Values are normalized for consistency:

```python
# Example: ManagerID = 14.0 (float from CSV)
val_str = "14.0"

# Normalize: Convert float to int if appropriate
if '.' in val_str:
    float_val = float(val_str)  # 14.0
    if float_val.is_integer():   # True
        normalized_val = str(int(float_val))  # "14"
```

**Normalization Rules**:
- `14.0` → `"14"` (float to int)
- `3.5` → `"3.5"` (keeps decimals)
- `"David Stanley"` → `"David Stanley"` (strings unchanged)
- Empty strings: Skipped (except manager columns)

**Why Normalize?**
- Ensures `fact_exists()` checks work correctly
- Prevents duplicates: `14.0` vs `14` would be different facts
- Consistent storage and lookup

---

### 6. **Deduplication (fact_exists Check)**

**File**: `agent_system.py`  
**Function**: `process_chunk()` (batch duplicate checking)  
**Lines**: 1238-1247

Before adding a fact, the system checks if it already exists:

```python
# Batch check all facts for this row
facts_to_add = []
facts_to_skip = []

for emp_name, pred, val in row_facts_to_check:
    if fact_exists(emp_name, pred, val):
        facts_to_skip.append((emp_name, pred, val))
    else:
        facts_to_add.append((emp_name, pred, val))
```

**How `fact_exists()` Works**:

**File**: `knowledge.py`  
**Function**: `fact_exists()`  
**Lines**: 1293-1318
```python
def fact_exists(subject, predicate, object_val):
    # Normalize inputs (case-insensitive)
    subject_norm = normalize_entity(subject.lower())
    predicate_norm = predicate.lower()
    object_norm = normalize_entity(object_val.lower())
    
    # Fast O(1) lookup using in-memory set
    return (subject_norm, predicate_norm, object_norm) in _fact_lookup_set
```

**In-Memory Index** (`_fact_lookup_set`):
- Stores normalized `(subject, predicate, object)` tuples
- Provides O(1) lookup (much faster than iterating graph)
- Automatically updated when facts are added

---

### 7. **Adding Facts to Knowledge Graph**

**File**: `agent_system.py`  
**Function**: `process_chunk()` (batch fact addition)  
**Lines**: 1249-1312

**Knowledge Graph Storage**:
- **File**: `knowledge.py`
- **Graph Object**: Global `graph` (RDFLib Graph instance)
- **Storage File**: `knowledge_graph.pkl` (pickled RDF graph)

Facts are added as RDF triples:

```python
# For each new fact
subject_clean = "Brill_Donna"  # Replace spaces with underscores
predicate_clean = "has_manager_name"
object_clean = "David Stanley"

# Create RDF URIs
subject_uri = URIRef("urn:entity:Brill_Donna")
predicate_uri = URIRef("urn:predicate:has_manager_name")
object_literal = Literal("David Stanley")

# Add to graph
graph.add((subject_uri, predicate_uri, object_literal))

# Update in-memory index
_fact_lookup_set.add(("brill donna", "has manager name", "david stanley"))
```

**RDF Structure**:
- **Subject**: `urn:entity:Brill_Donna` (employee name)
- **Predicate**: `urn:predicate:has_manager_name` (relationship type)
- **Object**: `"David Stanley"` (value, as literal)

**Batch Processing**:
- All facts for a row are collected first
- Then added in a single batch (reduces lock contention)
- Source document tracking added separately

---

### 8. **Source Document Tracking**

**File**: `agent_system.py`  
**Function**: `process_chunk()` (source tracking)  
**Lines**: 1306-1312

**Source Tracking Function**:
- **File**: `knowledge.py`
- **Function**: `add_fact_source_document()`

Each fact is linked to its source document:

```python
add_fact_source_document(
    employee_name="Brill, Donna",
    predicate="has manager name",
    object_val="David Stanley",
    source_document="HRDataset_v14.csv",
    uploaded_at="2024-01-15T10:30:00"
)
```

**Why Track Sources?**
- Enables traceability (know where facts came from)
- Supports multi-document knowledge graphs
- Helps with fact verification

---

## Example: Complete Fact Extraction

### Input CSV Row:
```csv
Employee_Name,ManagerName,ManagerID,Salary,Position,Department
"Brill, Donna",David Stanley,14.0,53492,Production Technician I,Production
```

### Processing Steps:

1. **Extract Employee Name**:
   - Raw: `"Brill, Donna"`
   - Normalized: `Brill, Donna` (quotes removed)

2. **Process Each Column**:

   **Column: ManagerName**
   - Value: `"David Stanley"`
   - Predicate: `"has manager name"` (from mapping)
   - Normalized: `"David Stanley"` (no change)
   - Fact: `(Brill, Donna, has manager name, David Stanley)`

   **Column: ManagerID**
   - Value: `14.0` (float)
   - Predicate: `"has manager id"` (from mapping)
   - Normalized: `"14"` (14.0 → 14)
   - Fact: `(Brill, Donna, has manager id, 14)`

   **Column: Salary**
   - Value: `53492`
   - Predicate: `"has salary"` (from mapping)
   - Normalized: `"53492"` (no change)
   - Fact: `(Brill, Donna, has salary, 53492)`

   **Column: Position**
   - Value: `"Production Technician I"`
   - Predicate: `"has position"` (from mapping)
   - Normalized: `"Production Technician I"` (no change)
   - Fact: `(Brill, Donna, has position, Production Technician I)`

   **Column: Department**
   - Value: `"Production"`
   - Predicate: `"works in department"` (from mapping)
   - Normalized: `"Production"` (no change)
   - Fact: `(Brill, Donna, works in department, Production)`

3. **Check for Duplicates**:
   - For each fact, check `fact_exists()`
   - Skip if already exists
   - Add if new

4. **Add to Knowledge Graph**:
   - All new facts added as RDF triples
   - In-memory index updated
   - Source document tracked

### Result:
5 facts added to knowledge graph:
```
(Brill, Donna, has manager name, David Stanley)
(Brill, Donna, has manager id, 14)
(Brill, Donna, has salary, 53492)
(Brill, Donna, has position, Production Technician I)
(Brill, Donna, works in department, Production)
```

---

## Key Optimizations

### 1. **Parallel Processing**
- Multiple worker agents process chunks simultaneously
- Thread-safe locks prevent race conditions
- Shared knowledge graph with synchronized access

### 2. **Batch Operations**
- Collect all facts for a row first
- Check duplicates in one batch
- Add facts in one batch (reduces lock contention)

### 3. **In-Memory Index**
- `_fact_lookup_set` provides O(1) duplicate checking
- Much faster than iterating entire graph
- Automatically maintained

### 4. **Adaptive Chunking**
- Chunk size based on data complexity
- More workers for larger files
- Balanced load across workers

### 5. **Value Normalization**
- Consistent storage (14.0 → 14)
- Prevents duplicate facts
- Reliable fact lookup

---

## Error Handling & Logging

### Progress Logging:
```
🔄 Worker 0: Processing rows 0-99 (100 rows expected)
   📊 Worker 0: Processing row 0 (1/100 rows processed)
   📝 Adding manager fact: Brill, Donna → has manager name → David Stanley
   📝 Adding manager fact: Brill, Donna → has manager id → 14
```

### Verification:
- Chunk coverage verification (ensures no rows missed)
- Row processing tracking (logs skipped rows)
- Final verification (compares processed vs expected rows)

---

## Summary

The fact extraction system:
1. ✅ Reads CSV into DataFrame (`agent_system.py` → `extract_csv_facts_directly_parallel_from_df()`)
2. ✅ Splits into chunks for parallel processing (`agent_system.py` → lines 1007-1025)
3. ✅ Maps columns to standardized predicates (`agent_system.py` → `process_chunk()` → lines 1177-1232)
4. ✅ Normalizes values (14.0 → 14) (`agent_system.py` → multiple locations)
5. ✅ Checks for duplicates (O(1) lookup) (`knowledge.py` → `fact_exists()`)
6. ✅ Adds facts as RDF triples (`agent_system.py` → `process_chunk()` → lines 1249-1312)
7. ✅ Tracks source documents (`knowledge.py` → `add_fact_source_document()`)
8. ✅ Maintains in-memory index (`knowledge.py` → `_fact_lookup_set`)

**Result**: A structured knowledge graph where each employee's information is stored as searchable facts, enabling powerful querying and analysis.

---

## File Reference Summary

| Process | File | Function/Component | Lines |
|---------|------|-------------------|-------|
| CSV Upload Trigger | `api_server.py` | `/api/knowledge/upload` endpoint | ~300-400 |
| Main Extraction Entry | `agent_system.py` | `extract_csv_facts_directly_parallel_from_df()` | 932-1400 |
| Chunking Logic | `agent_system.py` | `extract_csv_facts_directly_parallel_from_df()` | 976-1006 |
| Worker Processing | `agent_system.py` | `process_chunk()` (nested function) | 1046-1400 |
| Column Mapping | `agent_system.py` | `process_chunk()` → column loop | 1177-1232 |
| Value Normalization | `agent_system.py` | `process_chunk()` → fact addition | 1261-1271 |
| Duplicate Checking | `knowledge.py` | `fact_exists()` | 1293-1318 |
| Graph Storage | `knowledge.py` | Global `graph` (RDFLib) | Throughout |
| In-Memory Index | `knowledge.py` | `_fact_lookup_set` | ~50, ~250 |
| Source Tracking | `knowledge.py` | `add_fact_source_document()` | ~1800 |
| Graph Persistence | `knowledge.py` | `save_knowledge_graph()` | ~80-113 |

