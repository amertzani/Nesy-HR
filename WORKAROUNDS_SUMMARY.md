# Workarounds for LLM Query Issues

This document summarizes all the workarounds available to extract operational insights and answer evaluation queries **without relying on the LLM**.

## 🎯 Problem

The LLM may not work well for complex queries, but the operational insights and statistics **already exist** in:
1. Knowledge Graph (as facts)
2. Operational Insights API endpoint
3. Pre-computed data structures

## ✅ Solutions Available

### Solution 1: Direct Knowledge Graph Query (Recommended)

**Tool**: `query_kg_direct.py`

**What it does**: Searches the knowledge graph directly by keywords, bypassing LLM completely.

**Usage**:
```bash
# List all operational insights
python query_kg_direct.py --list-operational

# Search for specific queries
python query_kg_direct.py "average engagement manager"
python query_kg_direct.py "salary department" --source operational_insights
```

**Advantages**:
- ✅ Fast (no LLM processing)
- ✅ Reliable (direct fact retrieval)
- ✅ Works even when LLM fails
- ✅ Shows exact values from operational insights

**Example Output**:
```
Query: average engagement manager
Found 15 matching facts

[operational_insights] (15 facts)
1. Manager Alex Sweetwater → has average team engagement → 4.08
2. Manager Amy Dunn → has average team engagement → 3.92
...
```

### Solution 2: Structured Data Extraction

**Tool**: `extract_operational_data.py`

**What it does**: Extracts operational insights and formats them as tables.

**Usage**:
```bash
# Get manager insights as table
python extract_operational_data.py --type manager

# Get department insights as table
python extract_operational_data.py --type department

# Get both
python extract_operational_data.py --type all
```

**Advantages**:
- ✅ Formatted tables (ready for paper/analysis)
- ✅ Direct extraction (no LLM)
- ✅ Easy to compare with ground truth

**Example Output**:
```
MANAGER INSIGHTS
================================================================================
Name | Avg Engagement | Avg Performance | Avg Salary | Team Size
--------------------------------------------------------------------------------
Alex Sweetwater | 4.08 | 3.50 | 95000.00 | 9
Amy Dunn | 3.92 | 4.20 | 75000.00 | 21
...
```

### Solution 3: API Endpoint (If Server Running)

**Endpoint**: `GET /api/insights/operational`

**What it does**: Returns structured operational insights directly from the API.

**Usage**:
```bash
# If server is running
curl http://localhost:8000/api/insights/operational

# Or use Python
import requests
response = requests.get("http://localhost:8000/api/insights/operational")
data = response.json()
```

**Advantages**:
- ✅ Structured JSON format
- ✅ All insights in one response
- ✅ Easy to process programmatically

### Solution 4: Direct Python Access

**What it does**: Import system modules and access operational insights directly.

**Usage**:
```python
from operational_queries import compute_operational_insights
from strategic_queries import load_csv_data

# Load data
df = load_csv_data("/path/to/HRDataset_v14.csv")

# Get insights
insights = compute_operational_insights(df=df)

# Access structured data
manager_insights = insights.get('by_manager', [])
dept_insights = insights.get('by_department', [])
```

**Advantages**:
- ✅ Full programmatic control
- ✅ Can compute on-the-fly
- ✅ Access to all data structures

## 📊 For Your Evaluation Scenarios

### Scenario O1: Performance Score by Department

**Direct Query**:
```bash
python query_kg_direct.py "performance department" --source operational_insights
```

**Structured Table**:
```bash
python extract_operational_data.py --type department
```

**Compare with Ground Truth**: Values from `test_scenarios.json` show expected department performance metrics.

### Scenario O2: Absences by Employment Status

**Direct Query**:
```bash
python query_kg_direct.py "absence employment status"
```

**Note**: This might need to query employee facts directly, not just operational insights.

### Scenario O3: Engagement by Manager

**Direct Query**:
```bash
python query_kg_direct.py "engagement manager" --source operational_insights
```

**Structured Table**:
```bash
python extract_operational_data.py --type manager
```

**Compare**: Ground truth shows manager engagement averages (e.g., Alex Sweetwater: 4.08, Amy Dunn: 3.92)

### Scenario O4: Salary by Department

**Direct Query**:
```bash
python query_kg_direct.py "salary department" --source operational_insights
```

**Structured Table**:
```bash
python extract_operational_data.py --type department
```

**Compare**: Ground truth shows IT/IS: $97K, Admin: $72K, Executive: $250K

## 🔄 Recommended Workflow

1. **Check Available Data**:
   ```bash
   python query_kg_direct.py --list-operational
   ```

2. **Extract Answers for Each Scenario**:
   ```bash
   # For each scenario, run appropriate query
   python query_kg_direct.py "<scenario keywords>" --source operational_insights
   ```

3. **Get Structured Data**:
   ```bash
   python extract_operational_data.py --type all
   ```

4. **Compare with Ground Truth**:
   - Load ground truth from `test_scenarios.json`
   - Compare extracted values with expected values
   - Calculate accuracy metrics

5. **Format for Paper**:
   - Use extracted tables directly
   - Show comparison with ground truth
   - Document the workaround approach

## 💡 Additional Ideas

### Idea 1: Automated Comparison Script

Create a script that:
- Loads test scenarios
- Extracts answers using direct KG queries
- Compares with ground truth
- Generates accuracy report

### Idea 2: Hybrid Approach

- Use direct KG queries for operational insights (k=2)
- Use LLM for strategic queries (k≥3) when needed
- Document which method was used for each query

### Idea 3: Pre-computed Answer Cache

- Extract all operational insights once
- Store as JSON
- Use for evaluation without re-querying

## 📝 Summary

**You have 4 workarounds** to extract operational insights without LLM:

1. ✅ **Direct KG Query** (`query_kg_direct.py`) - Best for keyword searches
2. ✅ **Structured Extraction** (`extract_operational_data.py`) - Best for tables
3. ✅ **API Endpoint** - Best for programmatic access
4. ✅ **Direct Python** - Best for custom processing

**All of these bypass the LLM completely** and give you direct access to the operational insights that are already computed and stored in your system!

## 🚀 Next Steps

1. **Test the tools** (once KG is populated):
   ```bash
   python query_kg_direct.py --list-operational
   ```

2. **Extract answers for evaluation scenarios**

3. **Compare with ground truth** to calculate accuracy

4. **Document in paper** how direct KG queries provide reliable answers even when LLM fails

