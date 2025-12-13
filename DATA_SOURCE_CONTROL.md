# Data Source Control Guide

This document explains where data comes from and how to control it.

## 📊 Data Sources

### 1. Ground Truth Computation

**Source**: CSV file (default: `/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv`)

**How it works**:
- Ground truth is computed **directly from the CSV** using pandas
- Uses `groupby()`, `crosstab()`, and correlation operations
- Stored in `test_scenarios.json`

**Control it**:
```bash
# Use default CSV
python evaluation_test_scenarios.py

# Use a different CSV file
python evaluation_test_scenarios.py /path/to/your/dataset.csv
```

**What it computes**:
- For each scenario, calculates statistics directly from CSV
- Example: `df.groupby('Department')['Salary'].mean()` for salary by department
- This is the "correct" answer that system responses should match

### 2. Terminal Query Tool Responses

**Source**: Knowledge Graph (`knowledge_graph.pkl`)

**How it works**:
- The terminal tool (`answer_query_terminal.py`) loads facts from `knowledge_graph.pkl`
- This file contains facts extracted when you uploaded the CSV through the web interface
- Includes both raw employee facts AND operational insights

**Where the KG comes from**:
1. You upload CSV through web interface (`/api/knowledge/upload`)
2. System processes CSV and extracts facts
3. Operational insights are computed and stored as facts
4. Everything is saved to `knowledge_graph.pkl`

**Control it**:
- The KG file is at: `knowledge_graph.pkl` (in project root)
- To use a different KG: Copy/rename the file, or specify path (would need code modification)
- To repopulate: Upload CSV again through web interface

**Current KG status**:
- File: `knowledge_graph.pkl` (4.8MB)
- Facts: 27,569 total
- Operational insights: 319 facts
- Source: Processed from `HRDataset_v14.csv`

### 3. System Responses (Web Interface)

**Source**: Same knowledge graph (`knowledge_graph.pkl`) + LLM

**How it works**:
- Web interface loads same KG file
- LLM uses facts from KG to generate responses
- If LLM fails, you can use direct KG queries (workaround)

## 🔄 Data Flow

```
CSV File (HRDataset_v14.csv)
    ↓
[Upload via Web Interface]
    ↓
[System Processing]
    ├─→ Extract employee facts → knowledge_graph.pkl
    ├─→ Compute operational insights → knowledge_graph.pkl
    └─→ Compute statistics → knowledge_graph.pkl
    ↓
knowledge_graph.pkl (27,569 facts)
    ↓
[Terminal Tools Read From KG]
    ├─→ answer_query_terminal.py
    ├─→ query_kg_direct.py
    └─→ extract_operational_data.py
```

```
CSV File (HRDataset_v14.csv)
    ↓
[Ground Truth Computation]
    ├─→ Direct pandas operations
    └─→ test_scenarios.json
```

## 🎯 Controlling Data Sources

### Option 1: Use Different CSV for Ground Truth

```bash
# Compute ground truth from a different CSV
python evaluation_test_scenarios.py /path/to/different/dataset.csv
```

This will:
- Load the new CSV
- Compute ground truth statistics
- Save to `test_scenarios.json`
- **Note**: This doesn't change the knowledge graph

### Option 2: Repopulate Knowledge Graph

To change what the terminal tools use:

1. **Upload new CSV through web interface**:
   - Start backend: `./start_backend.sh`
   - Open web interface
   - Upload new CSV file
   - Wait for processing

2. **Or process CSV programmatically**:
   ```python
   from agent_system import process_document_with_agents
   process_document_with_agents(
       document_id="test",
       document_name="new_dataset.csv",
       document_type=".csv",
       file_path="/path/to/new_dataset.csv"
   )
   ```

### Option 3: Use Both Sources for Comparison

You can:
1. Compute ground truth from CSV A
2. Populate KG from CSV B
3. Compare answers to see differences

## 📍 Current Configuration

**Ground Truth Source**:
- File: `/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv`
- Computed: Direct pandas operations
- Stored: `test_scenarios.json`

**Terminal Tool Source**:
- File: `knowledge_graph.pkl` (in project root)
- Contains: Facts from processed `HRDataset_v14.csv`
- Facts: 27,569 total (including 319 operational insights)

**Both use the same dataset** (HRDataset_v14.csv), but:
- Ground truth: Computed fresh from CSV each time
- Terminal tool: Uses pre-computed facts stored in KG

## 🔍 Verify Data Sources

### Check Ground Truth Source

```bash
# Check test_scenarios.json
cat test_scenarios.json | grep -A 5 '"path"'
```

Shows which CSV was used to compute ground truth.

### Check Knowledge Graph Source

```bash
# Check KG file
ls -lh knowledge_graph.pkl

# See what's in the KG
python query_kg_direct.py --list-operational
```

### Check if They Match

The ground truth and KG should be from the same CSV file. To verify:

1. **Ground truth**: Check `test_scenarios.json` → `dataset.path`
2. **KG**: Check when CSV was uploaded (from documents_store.json or web interface)

## 💡 Recommendations

1. **For Evaluation**: Use the same CSV for both ground truth and KG
2. **For Testing**: You can use different CSVs to test system robustness
3. **For Comparison**: Compute ground truth from CSV, then compare with KG answers

## 🛠️ Making Changes

### To Change Ground Truth CSV:

```bash
python evaluation_test_scenarios.py /new/path/to/csv.csv
```

### To Change KG Data:

1. Upload new CSV through web interface, OR
2. Modify code to load a different KG file (advanced)

### To Use Both:

Keep them separate - ground truth is always computed fresh from CSV, while KG is persistent storage.

