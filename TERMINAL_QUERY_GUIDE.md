# Terminal Query Testing Guide

## ✅ How Ground Truth Was Computed

The ground truth in `test_scenarios.json` is computed **directly from the CSV dataset** using pandas:

1. **For numeric × categorical** (e.g., Absences × EmploymentStatus):
   - Uses `groupby().agg(['mean', 'median', 'std', 'count'])`
   - Example: Active employees have mean=9.83 absences

2. **For categorical × categorical** (e.g., PerformanceScore × Department):
   - Uses `pd.crosstab()` - cross-tabulation table
   - Shows distribution counts (e.g., IT/IS: 6 Exceeds, 42 Fully Meets, etc.)

3. **For numeric × numeric**:
   - Uses correlation coefficient

See `GROUND_TRUTH_EXPLANATION.md` for detailed examples.

## 🖥️ Terminal Query Tool

You can now answer queries directly in the terminal!

### Basic Usage

```bash
# Answer any query
python answer_query_terminal.py "Which department has the highest average performance score?"

# See detailed information
python answer_query_terminal.py "What is the average engagement by manager?" --verbose
```

### Example Queries

**Performance Questions:**
```bash
python answer_query_terminal.py "Which department has the highest average performance score?"
python answer_query_terminal.py "What is the performance distribution by department?"
```

**Engagement Questions:**
```bash
python answer_query_terminal.py "Which manager has the highest team engagement?"
python answer_query_terminal.py "What is the average engagement by manager?"
```

**Salary Questions:**
```bash
python answer_query_terminal.py "Which department has the highest average salary?"
python answer_query_terminal.py "What is the salary distribution by department?"
```

**Absence Questions:**
```bash
python answer_query_terminal.py "How do absences differ by employment status?"
python answer_query_terminal.py "Which department has the most absences?"
```

### How It Works

1. **Parses your query** to understand:
   - What metric (performance, engagement, salary, etc.)
   - What group (department, manager, etc.)
   - What operation (highest, lowest, average, distribution)

2. **Searches knowledge graph** for matching facts

3. **Extracts values** and computes answers

4. **Returns formatted answer**

### Example Output

```bash
$ python answer_query_terminal.py "Which department has the highest average salary?"

📂 Loading knowledge graph...
✅ 📂 Loaded 27569 facts from storage
📊 Knowledge graph: 27569 facts

🔍 Query: Which department has the highest average salary?

================================================================================
ANSWER
================================================================================
IT/IS has the highest average salary of 94382.00
```

## 🔍 Other Tools Available

### 1. Direct Keyword Search

```bash
# Search for specific keywords
python query_kg_direct.py "engagement manager" --source operational_insights

# List all operational insights
python query_kg_direct.py --list-operational
```

### 2. Structured Data Extraction

```bash
# Get manager insights as table
python extract_operational_data.py --type manager

# Get department insights as table
python extract_operational_data.py --type department
```

## 📊 Testing Your Evaluation Scenarios

### Scenario O1: Performance by Department

**Query:**
```bash
python answer_query_terminal.py "Which department has the highest average performance score?"
```

**Compare with ground truth**: From `test_scenarios.json`, you can compute which department has highest average by converting PerformanceScore to numeric and calculating mean.

### Scenario O2: Absences by Employment Status

**Query:**
```bash
python answer_query_terminal.py "How do absences differ by employment status?"
```

**Compare with ground truth**: 
- Active: mean=9.83 (from test_scenarios.json)
- Terminated for Cause: mean=11.56
- Voluntarily Terminated: mean=10.95

### Scenario O3: Engagement by Manager

**Query:**
```bash
python answer_query_terminal.py "Which manager has the highest team engagement?"
python answer_query_terminal.py "What is the average engagement by manager?"
```

**Compare with ground truth**: Check manager engagement averages from the KG vs. expected values.

### Scenario O4: Salary by Department

**Query:**
```bash
python answer_query_terminal.py "Which department has the highest average salary?"
```

**Compare with ground truth**:
- IT/IS: $97,065 (from test_scenarios.json)
- Admin Offices: $71,792
- Executive Office: $250,000

## 💡 Tips

1. **Use natural language**: The tool parses queries like "Which department has the highest..."
2. **Be specific**: Include both metric and group (e.g., "performance department" not just "performance")
3. **Check verbose mode**: Use `--verbose` to see which facts were used
4. **Compare with ground truth**: Use values from `test_scenarios.json` to verify accuracy

## 🎯 Next Steps

1. **Test all evaluation queries** using `answer_query_terminal.py`
2. **Extract answers** for each scenario
3. **Compare with ground truth** from `test_scenarios.json`
4. **Calculate accuracy metrics** (exact match, tolerance-based)
5. **Generate evaluation tables** for your paper

All tools bypass the LLM and work directly with the knowledge graph!

