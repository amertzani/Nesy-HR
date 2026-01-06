# How to Test Evaluation Queries

This guide explains how to test the evaluation queries in your system.

## Quick Start

### Option 1: Test via Web Interface (Easiest)

1. **Start your system**:
   ```bash
   # Terminal 1: Start backend
   ./start_backend.sh
   
   # Terminal 2: Start frontend (optional, for visual testing)
   ./start_frontend.sh
   ```

2. **Open the chat interface**:
   - Go to `http://localhost:5173/chat` (or your frontend URL)
   - Or use the API directly at `http://localhost:8000`

3. **Test queries manually**:
   - Copy queries from `evaluation_test_report.txt`
   - Paste them into the chat interface
   - Compare responses with ground truth

### Option 2: Test via Python Script (Automated)

Use the simple test script:

```bash
# List all available scenarios
python test_queries_simple.py --list

# Test a specific scenario (e.g., O1: Performance by Department)
python test_queries_simple.py --scenario O1

# Test a specific query from a scenario (query index 0)
python test_queries_simple.py --scenario O1 --query-index 0

# Test all scenarios
python test_queries_simple.py --all

# Use API mode (if system is running separately)
python test_queries_simple.py --scenario O1 --api
```

## Available Scenarios

### Operational Scenarios (k=2)

- **O1**: Performance Score by Department (4 queries)
- **O2**: Absences by Employment Status (3 queries)
- **O3**: Engagement by Manager (3 queries)
- **O4**: Salary by Department (3 queries)
- **O5**: Performance by Recruitment Source (3 queries)

### Strategic Scenarios (k≥3)

- **S1**: Performance-Engagement-Status Risk Clusters (3 queries)
- **S2**: Recruitment Channel Quality (3 queries)
- **S3**: Department Compensation-Performance Analysis (3 queries)

**Total**: 25 queries across 8 scenarios

## Testing Methods

### Method 1: Direct Mode (System Modules)

If you're running the script in the same environment as your system:

```bash
python test_queries_simple.py --scenario O1
```

This uses the system modules directly (no HTTP needed).

### Method 2: API Mode (HTTP Requests)

If your system is running as a server:

```bash
# Make sure server is running
./start_backend.sh

# In another terminal, test via API
python test_queries_simple.py --scenario O1 --api
```

### Method 3: Manual Testing via Web UI

1. Start your system
2. Open chat interface
3. Copy queries from scenarios
4. Test manually and note results

## Example Test Session

```bash
# 1. List scenarios
$ python test_queries_simple.py --list

Available Scenarios:
================================================================================

O1: Performance Score by Department
  Type: operational
  Variables: PerformanceScore, Department
  Queries: 4
    0. What is the distribution of performance scores by department?
    1. How do performance scores vary across departments?
    2. Which department has the highest average performance score?
    3. Show me performance metrics by department

# 2. Test scenario O1
$ python test_queries_simple.py --scenario O1

🔧 Using direct mode (system modules)

================================================================================
Testing Scenario O1: Performance Score by Department
Variables: PerformanceScore, Department
Type: operational
Total queries: 4
================================================================================

[1/4] Testing query...
================================================================================
Query: What is the distribution of performance scores by department?
--------------------------------------------------------------------------------
✅ Response received (2.34s)
   Query Type: operational
   Routing Strategy: operational_agent
   Reason: Routed to operational agent

Response Preview:
Based on the operational insights from the knowledge graph, here is the 
distribution of performance scores by department...
================================================================================
```

## Comparing with Ground Truth

Each scenario includes ground truth data computed from the dataset. For example:

**Scenario O2 (Absences × EmploymentStatus)**:
- Active employees: mean=9.83 absences, n=207
- Terminated for Cause: mean=11.56 absences, n=16
- Voluntarily Terminated: mean=10.95 absences, n=88

Compare your system's response with these values to assess accuracy.

## Output Files

The test script generates:
- `test_results_YYYYMMDD_HHMMSS.json` - Detailed results in JSON format
- Console output with response previews

## Troubleshooting

### "System modules not found"
- Use `--api` flag to test via HTTP
- Or make sure you're in the correct Python environment

### "Could not connect to API"
- Make sure backend server is running: `./start_backend.sh`
- Check the API URL (default: `http://localhost:8000`)
- Use `--api-url` to specify different URL

### "Request timed out"
- Large queries may take time, especially on CPU
- The script has a 120s timeout
- Try simpler queries first

### "No scenarios found"
- Run `python evaluation_test_scenarios.py` first to generate scenarios

## Next Steps

After testing, you can:
1. Compare responses with ground truth
2. Calculate accuracy metrics
3. Generate evaluation tables for your paper
4. Test with LLM baselines (GPT-4, Claude, etc.) for comparison

## Quick Reference

```bash
# List scenarios
python test_queries_simple.py --list

# Test one scenario
python test_queries_simple.py --scenario O1

# Test one query
python test_queries_simple.py --scenario O1 --query-index 0

# Test all scenarios
python test_queries_simple.py --all

# Use API mode
python test_queries_simple.py --scenario O1 --api
```

