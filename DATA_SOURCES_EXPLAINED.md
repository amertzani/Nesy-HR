# Data Sources Explained

## 📊 Current Configuration

### Ground Truth (test_scenarios.json)

**Source CSV**: `/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv`
- **Rows**: 311
- **Columns**: 36
- **Computed**: Direct pandas operations (groupby, crosstab, correlation)
- **Stored in**: `test_scenarios.json`

**How it's computed**:
```python
# Example: Absences by EmploymentStatus
df.groupby('EmploymentStatus')['Absences'].agg(['mean', 'median', 'std', 'count'])
```

### Terminal Query Tool (answer_query_terminal.py)

**Source**: `knowledge_graph.pkl` (in project root)
- **File size**: 4.8MB
- **Total facts**: 27,569
- **Operational insights**: 319 facts
- **Original data**: Same CSV (`HRDataset_v14.csv`) processed through web interface

**How it works**:
1. You uploaded CSV through web interface
2. System extracted facts and computed operational insights
3. Everything saved to `knowledge_graph.pkl`
4. Terminal tool loads this file and searches it

## 🔄 Data Flow

```
HRDataset_v14.csv (311 rows, 36 columns)
    │
    ├─→ [Ground Truth Computation]
    │   └─→ Direct pandas → test_scenarios.json
    │
    └─→ [Web Interface Upload]
        └─→ System Processing
            └─→ knowledge_graph.pkl (27,569 facts)
                └─→ Terminal Tools Read From Here
```

## ✅ They Use the Same Data!

Both ground truth and terminal tool responses come from **the same CSV file**:
- **Ground truth**: Computed fresh from CSV using pandas
- **Terminal tool**: Uses facts extracted from the same CSV (stored in KG)

## 🎛️ How to Control

### 1. Change Ground Truth CSV

```bash
# Use a different CSV for ground truth
python evaluation_test_scenarios.py /path/to/different.csv
```

This will:
- Compute new ground truth from the new CSV
- Save to `test_scenarios.json`
- **Does NOT change the knowledge graph**

### 2. Change Knowledge Graph Data

The terminal tool uses `knowledge_graph.pkl`. To change it:

**Option A: Upload new CSV through web interface**
1. Start backend: `./start_backend.sh`
2. Open web interface
3. Upload new CSV
4. System will process and update `knowledge_graph.pkl`

**Option B: Use a different KG file** (requires code modification)
- Currently hardcoded to `knowledge_graph.pkl`
- Could be modified to accept a file path argument

### 3. Verify They Match

```bash
# Check ground truth source
python -c "import json; print(json.load(open('test_scenarios.json'))['dataset']['path'])"

# Check KG facts count
python -c "from knowledge import load_knowledge_graph, graph; load_knowledge_graph(); print(f'KG has {len(graph)} facts')"
```

## 🔍 Current Status

✅ **Ground Truth**: From `HRDataset_v14.csv` (311 rows, 36 columns)  
✅ **Terminal Tool**: From `knowledge_graph.pkl` (27,569 facts from same CSV)  
✅ **They Match**: Both use the same source data

## 💡 Important Notes

1. **Ground truth is computed fresh** from CSV each time you run `evaluation_test_scenarios.py`
2. **KG is persistent** - once populated, it stays until you upload a new file
3. **They should match** if both use the same CSV file
4. **You can use different CSVs** to test system with different datasets

## 🧪 Testing with Different Data

If you want to test with a different dataset:

1. **For ground truth**:
   ```bash
   python evaluation_test_scenarios.py /path/to/new_dataset.csv
   ```

2. **For terminal tool**:
   - Upload new CSV through web interface
   - Or manually process and save to KG

3. **Compare results**:
   - Ground truth from CSV A
   - Terminal answers from CSV B
   - See how system handles different data

## 📝 Summary

- **Ground Truth**: Always computed from CSV (configurable via command line)
- **Terminal Tool**: Uses knowledge graph file (populated from uploaded CSV)
- **Both currently use**: `HRDataset_v14.csv`
- **You can control**: CSV path for ground truth, KG by uploading different CSV

