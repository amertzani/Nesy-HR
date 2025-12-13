# Quick Test Guide - Direct KG Queries

## ✅ It's Working!

Your knowledge graph has **27,569 facts** including **319 operational insights**. Here's how to use it:

## Quick Commands

### 1. List All Operational Insights
```bash
python query_kg_direct.py --list-operational
```

### 2. Search by Keywords

**Engagement by Manager:**
```bash
python query_kg_direct.py "engagement manager" --source operational_insights
```

**Salary by Department:**
```bash
python query_kg_direct.py "salary department" --source operational_insights
```

**Performance by Department:**
```bash
python query_kg_direct.py "performance department" --source operational_insights
```

**Absences:**
```bash
python query_kg_direct.py "absence" --source operational_insights
```

### 3. Get Structured Tables

```bash
# Manager insights
python extract_operational_data.py --type manager

# Department insights  
python extract_operational_data.py --type department

# Both
python extract_operational_data.py --type all
```

## Example Results

When you run `python query_kg_direct.py "engagement manager"`, you'll see:

```
Found 15 matching facts

Manager Brannon Miller → has → average team engagement score of 4
Manager Alex Sweetwater's team → has → average engagement survey value of 3
Manager Board of Directors → has → average engagement survey value of 5
Manager Janet King's team → has → average engagement survey value of 4
...
```

## Tips

1. **Use simpler keywords**: "engagement manager" works better than "average engagement by manager"
2. **Filter by source**: Use `--source operational_insights` to get only pre-computed insights
3. **Increase limit**: Use `--limit 50` or `--limit 100` for more results
4. **Check what's available**: Run `--list-operational` first to see all available insights

## For Your Evaluation Scenarios

### Scenario O1: Performance by Department
```bash
python query_kg_direct.py "performance department" --source operational_insights
```

### Scenario O2: Absences by Employment Status
```bash
python query_kg_direct.py "absence employment" --limit 30
```

### Scenario O3: Engagement by Manager
```bash
python query_kg_direct.py "engagement manager" --source operational_insights
```

### Scenario O4: Salary by Department
```bash
python query_kg_direct.py "salary department" --source operational_insights
```

## What You Can Do Now

1. ✅ **Extract answers** directly from KG for each scenario
2. ✅ **Compare with ground truth** from `test_scenarios.json`
3. ✅ **Calculate accuracy** by matching extracted values
4. ✅ **Generate tables** for your paper using the structured data

The tools are working! You can now extract operational insights without relying on the LLM.

