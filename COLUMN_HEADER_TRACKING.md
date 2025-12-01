# CSV Column Header Tracking - How Headers Are Maintained

## Overview

This document explains how the system maintains and tracks CSV column headers during fact extraction, ensuring that the relationship between CSV columns and extracted facts is preserved.

---

## Current Implementation

### 1. **Column Header Detection & Storage**

**File**: `agent_system.py`  
**Function**: `extract_csv_facts_directly_parallel_from_df()`  
**Lines**: 966-974, 1027-1042

```python
# Step 1: Read CSV - pandas automatically reads headers
df = pd.read_csv(file_path, sep=';', encoding='utf-8')
# df.columns contains: ['Employee_Name', 'ManagerName', 'ManagerID', 'Salary', ...]

# Step 2: Store ALL columns in parent document agent
parent_agent.columns_processed = list(df.columns)
# Result: ['Employee_Name', 'ManagerName', 'ManagerID', 'Salary', 'Position', ...]
```

**Key Points**:
- ✅ **All column headers are captured** when CSV is read into DataFrame
- ✅ **Column list is stored** in `DocumentAgent.columns_processed`
- ✅ **Available for querying** via agent metadata

---

### 2. **Column Tracking During Processing**

**File**: `agent_system.py`  
**Function**: `process_chunk()`  
**Lines**: 1161-1162, 1146-1149

```python
# During fact extraction, each column is tracked
for col, val in row.items():
    # Ensure column is tracked
    chunk_columns_processed.add(col)  # Tracks columns in this chunk
    
    # Also track in parent document agent
    if col not in parent_agent.columns_processed:
        parent_agent.columns_processed.append(col)
```

**What This Ensures**:
- ✅ Every column processed is recorded
- ✅ Parent agent knows ALL columns that exist in the CSV
- ✅ Can query which columns are available for a document

---

### 3. **Column → Predicate Mapping**

**File**: `agent_system.py`  
**Function**: `process_chunk()`  
**Lines**: 1177-1232

The system maps CSV column names to standardized predicates:

#### **Known Columns (Standardized Predicates)**:
```python
# ManagerName → "has manager name"
# ManagerID → "has manager id"
# Salary → "has salary"
# Position → "has position"
# Department → "works in department"
```

**Example**:
- CSV Column: `ManagerName`
- Predicate: `"has manager name"`
- Fact: `(Brill, Donna, has manager name, David Stanley)`

#### **Unknown Columns (Preserve Column Name)**:
```python
# For columns not in the mapping, use column name directly
else:
    predicate = f"has {col}"  # Preserves original column name
```

**Example**:
- CSV Column: `CustomField`
- Predicate: `"has CustomField"`
- Fact: `(Brill, Donna, has CustomField, some_value)`

**Key Point**: Unknown columns preserve their original name in the predicate!

---

### 4. **Column Information in Agent Metadata**

**File**: `agent_system.py`  
**Class**: `DocumentAgent`  
**Field**: `columns_processed: List[str]`

```python
@dataclass
class DocumentAgent(Agent):
    columns_processed: List[str] = field(default_factory=list)
    # Stores: ['Employee_Name', 'ManagerName', 'ManagerID', 'Salary', ...]
```

**How to Access**:
```python
# Get all columns for a document
agent = document_agents[document_id]
available_columns = agent.columns_processed
# Result: ['Employee_Name', 'ManagerName', 'ManagerID', 'Salary', ...]
```

---

## What Is Preserved

### ✅ **Preserved Information**:

1. **Complete Column List**:
   - Stored in `DocumentAgent.columns_processed`
   - Available for each document/agent
   - Can query: "What columns are in this CSV?"

2. **Column Names in Predicates** (for unknown columns):
   - Unknown columns: `predicate = f"has {col}"`
   - Example: `"has CustomField"` preserves `CustomField`

3. **Column Tracking**:
   - Each column is tracked during processing
   - Parent agent maintains complete list
   - Worker agents track their processed columns

---

## What Is NOT Directly Preserved

### ⚠️ **Limitation**:

**Original Column Name for Standardized Predicates**:
- Known columns lose direct link to original name
- Example: `ManagerName` → `"has manager name"` (original name not in predicate)
- Cannot directly tell that `"has manager name"` came from `ManagerName` column

**Why This Happens**:
- Standardized predicates ensure consistency across different CSV formats
- `ManagerName`, `Manager_Name`, `manager_name` all map to `"has manager name"`
- This is intentional for data normalization

---

## How to Reconstruct Column Information

### **Method 1: Reverse Predicate Mapping**

**File**: `strategic_query_agent.py`  
**Function**: `normalize_column_name()` (lines ~100-150)

The system can reverse-map predicates to column names:

```python
# Known mappings
predicate_to_column = {
    "has manager name": ["ManagerName", "Manager_Name", "manager_name"],
    "has manager id": ["ManagerID", "Manager_ID", "manager_id"],
    "has salary": ["Salary", "salary"],
    # ...
}

# For unknown columns, predicate IS the column name
# "has CustomField" → column is "CustomField"
```

### **Method 2: Query Agent Metadata**

```python
# Get columns for a document
agent = document_agents[document_id]
columns = agent.columns_processed

# Check if a column exists
if "ManagerName" in columns:
    print("ManagerName column exists in CSV")
```

### **Method 3: Check Predicate Pattern**

```python
# For standardized predicates, check known patterns
if predicate == "has manager name":
    # Likely came from ManagerName, Manager_Name, or similar
    possible_columns = ["ManagerName", "Manager_Name", "manager_name"]
    
# For unknown columns, predicate contains column name
if predicate.startswith("has ") and predicate not in known_predicates:
    column_name = predicate.replace("has ", "")
    # This is the original column name!
```

---

## Example: Complete Column Tracking

### **Input CSV**:
```csv
Employee_Name,ManagerName,ManagerID,Salary,CustomField
"Brill, Donna",David Stanley,14.0,53492,ABC123
```

### **Processing**:

1. **Column Detection**:
   ```python
   df.columns = ['Employee_Name', 'ManagerName', 'ManagerID', 'Salary', 'CustomField']
   ```

2. **Storage in Agent**:
   ```python
   parent_agent.columns_processed = [
       'Employee_Name', 'ManagerName', 'ManagerID', 'Salary', 'CustomField'
   ]
   ```

3. **Fact Extraction**:
   - `ManagerName` → Predicate: `"has manager name"` (standardized)
   - `ManagerID` → Predicate: `"has manager id"` (standardized)
   - `Salary` → Predicate: `"has salary"` (standardized)
   - `CustomField` → Predicate: `"has CustomField"` (preserves name)

4. **Resulting Facts**:
   ```
   (Brill, Donna, has manager name, David Stanley)
   (Brill, Donna, has manager id, 14)
   (Brill, Donna, has salary, 53492)
   (Brill, Donna, has CustomField, ABC123)  ← Column name preserved!
   ```

---

## Current Capabilities

### ✅ **What You Can Do**:

1. **Query Available Columns**:
   ```python
   agent = document_agents[document_id]
   columns = agent.columns_processed
   # Know exactly which columns exist in the CSV
   ```

2. **Check Column Existence**:
   ```python
   if "ManagerName" in agent.columns_processed:
       # ManagerName column exists
   ```

3. **Reconstruct DataFrame**:
   - `strategic_query_agent.py` → `reconstruct_dataframe_from_facts()`
   - Uses predicate-to-column mapping
   - Rebuilds DataFrame structure from facts

4. **Identify Unknown Columns**:
   - Predicates like `"has CustomField"` directly show column name
   - No mapping needed for unknown columns

---

## Potential Improvements

### **Enhancement 1: Store Original Column Name as Fact Metadata**

**Current**: Predicate only (standardized)
**Proposed**: Add column name to fact details

```python
# When creating fact
add_fact_details(
    subject="Brill, Donna",
    predicate="has manager name",
    object="David Stanley",
    details=f"Source column: ManagerName"  # Add original column name
)
```

**Benefits**:
- Direct traceability: fact → original column
- No reverse mapping needed
- Preserves CSV structure information

### **Enhancement 2: Column Metadata in Document Agent**

**Current**: Just column names list
**Proposed**: Column metadata dictionary

```python
agent.column_metadata = {
    "ManagerName": {
        "type": "string",
        "predicate": "has manager name",
        "sample_values": ["David Stanley", "Ketsia Liebig"]
    },
    "Salary": {
        "type": "numeric",
        "predicate": "has salary",
        "sample_values": [53492, 64919]
    }
}
```

**Benefits**:
- Rich metadata about each column
- Type information
- Sample values for validation

---

## Summary

### **Current State**:

✅ **What Works**:
- All column headers are captured and stored
- Column list available in `DocumentAgent.columns_processed`
- Unknown columns preserve name in predicate
- Can query which columns exist

⚠️ **Limitation**:
- Standardized predicates lose direct link to original column name
- Requires reverse mapping to reconstruct

### **How to Use**:

1. **Get Available Columns**:
   ```python
   agent = document_agents[document_id]
   columns = agent.columns_processed
   ```

2. **Check Column Existence**:
   ```python
   if "ManagerName" in agent.columns_processed:
       # Column exists
   ```

3. **Reconstruct Column from Predicate**:
   - Unknown columns: `predicate.replace("has ", "")` = column name
   - Known columns: Use reverse mapping or check `columns_processed`

---

## File Locations

| Component | File | Function/Class | Lines |
|-----------|------|----------------|-------|
| Column Detection | `agent_system.py` | `extract_csv_facts_directly_parallel_from_df()` | 966-974 |
| Column Storage | `agent_system.py` | `DocumentAgent.columns_processed` | 65, 1033 |
| Column Tracking | `agent_system.py` | `process_chunk()` | 1161-1162 |
| Predicate Mapping | `agent_system.py` | `process_chunk()` | 1177-1232 |
| Reverse Mapping | `strategic_query_agent.py` | `normalize_column_name()` | ~100-150 |

