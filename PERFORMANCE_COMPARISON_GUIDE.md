# Performance Comparison Guide

This guide explains how to compare your knowledge graph-based system with LLM baselines (GPT-4, Claude, etc.) to demonstrate your system's advantages.

## 🎯 Why Compare?

Your system has unique advantages over LLMs:
1. **Traceability**: Can show evidence (facts from knowledge graph)
2. **Accuracy**: Grounded in your actual data (not training data)
3. **Speed**: Direct knowledge graph access (no LLM generation needed)
4. **Consistency**: Same query = same answer (deterministic)
5. **Up-to-date**: Uses your latest uploaded data

## 📊 Comparison Tools

### 1. Full Comparison Tool (`compare_with_llm.py`)

Comprehensive comparison that tests multiple queries and generates a detailed report.

**Usage:**
```bash
# Compare a specific scenario
python compare_with_llm.py --scenario O1

# Compare all scenarios
python compare_with_llm.py --all

# Use specific LLM
python compare_with_llm.py --llm gpt-4 --scenario O1

# Limit queries for faster testing
python compare_with_llm.py --scenario O1 --max-queries 2
```

**What it does:**
- Runs queries through your system
- Runs same queries through LLM (GPT-4, etc.)
- Compares accuracy against ground truth
- Measures response times
- Generates comprehensive report

**Output:**
- `llm_comparison_report.txt` - Human-readable report
- `llm_comparison_report.json` - Machine-readable results

### 2. Evidence Comparison Tool (`show_evidence_comparison.py`)

Quick demonstration of traceability advantage.

**Usage:**
```bash
python show_evidence_comparison.py "Which department has the highest average salary?"
```

**What it shows:**
- Your system's answer with evidence (facts)
- LLM's answer (no evidence)
- Side-by-side comparison

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install openai python-dotenv
```

### 2. Set OpenAI API Key

Create a `.env` file or export:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add to `.env`:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Ensure Your System is Ready

Make sure your knowledge graph is loaded:
```bash
python answer_query_terminal.py "test query"
```

## 📈 Metrics to Compare

### 1. Accuracy

Compare against ground truth from `test_scenarios.json`:
- **Your System**: Uses facts from knowledge graph
- **LLM**: Uses training data (may not have your specific data)

### 2. Response Time

- **Your System**: Direct KG lookup (fast)
- **LLM**: API call + generation (slower)

### 3. Traceability

- **Your System**: Can show evidence (facts used)
- **LLM**: Black box (no evidence)

### 4. Consistency

- **Your System**: Deterministic (same query = same answer)
- **LLM**: May vary between calls

## 📋 Example Workflow

### Step 1: Quick Evidence Demo

```bash
# Show traceability advantage
python show_evidence_comparison.py "Which department has the highest average performance score?"
```

### Step 2: Full Comparison

```bash
# Compare one scenario
python compare_with_llm.py --scenario O1 --max-queries 3

# Review the report
cat llm_comparison_report.txt
```

### Step 3: Comprehensive Comparison

```bash
# Compare all operational scenarios
python compare_with_llm.py --all --max-queries 2
```

### Step 4: Analyze Results

The report will show:
- Accuracy comparison
- Response time comparison
- Evidence count
- Winner for each query

## 🎯 Key Scenarios to Test

### Operational Queries (k=2)

These are your system's strength - direct data access:

1. **O1: Performance by Department**
   ```bash
   python compare_with_llm.py --scenario O1
   ```

2. **O2: Absences by Employment Status**
   ```bash
   python compare_with_llm.py --scenario O2
   ```

3. **O4: Salary by Department**
   ```bash
   python compare_with_llm.py --scenario O4
   ```

### Strategic Queries (k≥3)

More complex, but your system can still provide evidence:

```bash
python compare_with_llm.py --scenario S1
```

## 📊 Interpreting Results

### Accuracy Metrics

- **Your System > LLM**: Your system is more accurate (grounded in your data)
- **LLM > Your System**: LLM may have seen similar data in training
- **Tie**: Both correct, but your system has evidence advantage

### Response Time

- **Your System faster**: Direct KG access is faster than LLM generation
- **LLM faster**: API optimization, but lacks traceability

### Evidence

- **Your System**: Always shows evidence (if available)
- **LLM**: Never shows evidence (black box)

## 🎓 Presenting Results

### For Papers/Presentations

1. **Accuracy Table**: Show accuracy percentages
2. **Response Time Chart**: Bar chart comparing speeds
3. **Evidence Examples**: Show actual facts used
4. **Case Studies**: Pick 2-3 queries showing clear advantages

### Example Presentation Format

```
Query: "Which department has the highest average salary?"

Your System:
  Answer: "IT/IS has the highest average salary of $97,065"
  Evidence: 15 facts from knowledge graph
  Response Time: 0.5s
  Accuracy: ✓ Correct

LLM Baseline (GPT-4):
  Answer: "Based on typical HR data, IT departments..."
  Evidence: None
  Response Time: 2.3s
  Accuracy: ✗ Incorrect (used generic training data)

Winner: Your System
```

## 🔍 Advanced: Custom Comparisons

You can modify `compare_with_llm.py` to:
- Add more LLM providers (Claude, Gemini, etc.)
- Custom accuracy metrics
- Different evaluation criteria
- Batch processing

## 💡 Tips

1. **Start Small**: Test 1-2 queries first
2. **Use Ground Truth**: Always compare against `test_scenarios.json`
3. **Show Evidence**: This is your unique advantage
4. **Document Everything**: Save reports for papers/presentations
5. **Be Fair**: Acknowledge when LLM performs better (and explain why)

## 🚀 Next Steps

1. Run quick evidence demo
2. Run full comparison on key scenarios
3. Generate report for your paper
4. Create visualizations (charts, tables)
5. Document findings

## 📝 Example Report Structure

```
KNOWLEDGE GRAPH SYSTEM vs LLM BASELINE COMPARISON
==================================================

SUMMARY STATISTICS
------------------
Total queries compared: 25
Accuracy:
  Your System: 22/25 (88.0%)
  LLM Baseline: 15/25 (60.0%)

Wins:
  Your System: 18
  LLM Baseline: 5
  Ties: 2

Response Times:
  Your System: 0.45s average
  LLM Baseline: 2.1s average
  Speedup: 4.7x faster

Evidence/Traceability:
  Your System: 156 facts provided
  LLM Baseline: 0 (no traceable evidence)

KEY ADVANTAGES
--------------
✓ Higher accuracy (88% vs 60%)
✓ Faster responses (0.45s vs 2.1s)
✓ Provides traceable evidence (156 facts)
✓ Wins more comparisons (18 vs 5)
```

## ❓ Troubleshooting

### "OPENAI_API_KEY not found"
- Set it in `.env` file or export as environment variable

### "System not available"
- Make sure `answer_query_terminal.py` works first
- Check that knowledge graph is loaded

### "Query timed out"
- Your system may be slow on first query (loading KG)
- LLM API may be rate-limited

### "Incorrect accuracy evaluation"
- Ground truth may need adjustment
- Check `test_scenarios.json` format

## 📚 Related Files

- `test_scenarios.json` - Ground truth data
- `answer_query_terminal.py` - Your system's query tool
- `evaluation_metrics.py` - Dataset metrics
- `compare_with_llm.py` - Full comparison tool
- `show_evidence_comparison.py` - Evidence demo

