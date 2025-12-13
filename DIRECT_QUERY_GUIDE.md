# Direct Knowledge Graph Query Guide

This guide shows you how to extract operational insights and statistics directly from the knowledge graph, **bypassing the LLM completely**.

## Why Use Direct Queries?

- ✅ **Faster**: No LLM processing time
- ✅ **More Reliable**: Direct access to stored facts
- ✅ **Accurate**: Gets exact values from operational insights
- ✅ **Works Even When LLM Fails**: Perfect workaround for LLM issues

## Quick Start

### 1. List Available Operational Insights

```bash
python query_kg_direct.py --list-operational
```

This shows all operational insights stored in the knowledge graph, organized by:
- By Manager (engagement, performance, salary, etc.)
- By Department (salary, performance, absences, etc.)
- By Recruitment Source (performance, retention, etc.)

### 2. Search by Keywords

```bash
# Search for engagement-related facts
python query_kg_direct.py "average engagement manager"

# Search for salary by department
python query_kg_direct.py "salary department"

# Search only in operational insights
python query_kg_direct.py "performance" --source operational_insights

# Get more results
python query_kg_direct.py "manager team" --limit 100
```

### 3. Extract Structured Data as Tables

```bash
# Get manager insights as a table
python extract_operational_data.py --type manager

# Get department insights as a table
python extract_operational_data.py --type department

# Get both
python extract_operational_data.py --type all
```

## Example Queries for Evaluation Scenarios

### Scenario O1: Performance Score by Department

```bash
python query_kg_direct.py "performance department" --source operational_insights
```

### Scenario O2: Absences by Employment Status

```bash
python query_kg_direct.py "absence employment status"
```

### Scenario O3: Engagement by Manager

```bash
python query_kg_direct.py "engagement manager" --source operational_insights
# Or get structured table:
python extract_operational_data.py --type manager
```

### Scenario O4: Salary by Department

```bash
python query_kg_direct.py "salary department" --source operational_insights
# Or get structured table:
python extract_operational_data.py --type department
```

## How It Works

1. **Direct KG Access**: Queries the knowledge graph directly (no LLM)
2. **Keyword Matching**: Searches facts by keywords in subject, predicate, or object
3. **Source Filtering**: Can filter by source document (e.g., "operational_insights")
4. **Structured Extraction**: Extracts and formats data as tables

## Output Examples

### Keyword Search Output

```
================================================================================
Query: average engagement manager
Found 15 matching facts
================================================================================

[operational_insights] (15 facts)
--------------------------------------------------------------------------------
1. Manager Alex Sweetwater → has average team engagement → 4.08
    [Source: operational_insights]
2. Manager Amy Dunn → has average team engagement → 3.92
    [Source: operational_insights]
...
```

### Structured Table Output

```
================================================================================
MANAGER INSIGHTS
================================================================================

Name | Avg Engagement | Avg Performance | Avg Salary | Avg Absences | Team Size
--------------------------------------------------------------------------------
Alex Sweetwater | 4.08 | 3.50 | 95000.00 | 5.00 | 9
Amy Dunn | 3.92 | 4.20 | 75000.00 | 8.00 | 21
...
```

## Tips

1. **Use Specific Keywords**: More specific = better results
   - ✅ Good: "average engagement manager"
   - ❌ Less specific: "engagement"

2. **Filter by Source**: Use `--source operational_insights` to get only pre-computed insights

3. **Check What's Available**: Run `--list-operational` first to see what's in the KG

4. **Combine with Evaluation**: Use these results to compare with ground truth from your scenarios

## Integration with Evaluation

You can use these tools to:

1. **Extract Answers**: Get system responses directly from KG
2. **Compare with Ground Truth**: Match against values in `test_scenarios.json`
3. **Generate Metrics**: Calculate accuracy by comparing extracted values with expected values
4. **Bypass LLM Issues**: Get reliable answers even when LLM fails

## Example Workflow

```bash
# 1. Check what operational insights are available
python query_kg_direct.py --list-operational

# 2. Query for specific scenario
python query_kg_direct.py "engagement manager" --source operational_insights

# 3. Get structured data
python extract_operational_data.py --type manager

# 4. Compare with ground truth from test_scenarios.json
# (You can write a script to automate this comparison)
```

## Troubleshooting

### "Knowledge graph is empty"
- Make sure you've uploaded and processed the CSV file
- Check that operational insights were computed: `./start_backend.sh` should show "Operational insights computed"

### "No matching facts found"
- Try broader keywords
- Check available insights: `python query_kg_direct.py --list-operational`
- Make sure operational insights were stored (check backend logs)

### "Could not import knowledge graph modules"
- Make sure you're in the correct directory
- The knowledge graph should be initialized when the system starts

## Next Steps

1. **Automate Comparison**: Create a script that:
   - Extracts answers from KG using these tools
   - Compares with ground truth from scenarios
   - Calculates accuracy metrics

2. **Format for Paper**: Use extracted data to create tables/figures for your evaluation section

3. **Combine with LLM Results**: Compare direct KG queries vs LLM responses to show the workaround's effectiveness

