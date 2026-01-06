# Evaluation Guide

This folder contains all evaluation tools, scripts, and documentation for testing the knowledge graph system.

## Quick Start

### Run Full Offline Evaluation

The main evaluation script tests the system against ground truth without requiring LLM API access:

```bash
# Run all test scenarios
python evaluation/evaluate_offline.py --all

# Run a specific scenario (e.g., O1: Performance by Department)
python evaluation/evaluate_offline.py --scenario O1

# Generate detailed report with methodology
python evaluation/evaluate_offline.py --all --output offline_evaluation_report.txt
```

### Compute Evaluation Metrics

After running evaluation, compute metrics for analysis:

```bash
# Compute metrics from evaluation report
python evaluation/compute_evaluation_metrics.py --report offline_evaluation_report.txt

# Group metrics by scenario type
python evaluation/compute_evaluation_metrics.py --group-by scenario_type

# Generate graph-ready data
python evaluation/compute_evaluation_metrics.py --graph-data
```

## Evaluation Scripts

### Main Scripts

- **`evaluate_offline.py`** - Main offline evaluation tool
  - Tests accuracy against ground truth
  - Measures response latency
  - Evaluates evidence retrieval
  - Generates comprehensive reports

- **`compute_evaluation_metrics.py`** - Metrics computation
  - Calculates Precision, Recall, F1
  - Computes Traceability Completeness
  - Estimates Hallucination Resistance
  - Generates summary statistics

- **`run_evaluation_32_queries.py`** - Run standard 32-query test suite
- **`run_evaluation_tests.py`** - Alternative test runner

### Utility Scripts

- **`extract_queries_with_responses.py`** - Extract query-response pairs from reports
- **`add_methodology_to_report.py`** - Add methodology sections to reports
- **`create_metrics_summary.py`** - Create metrics summary tables
- **`update_evaluation_summary.py`** - Update evaluation summaries

### Test Scripts

- **`test_queries_simple.py`** - Simple query testing
- **`test_employee_fact_retrieval.py`** - Employee fact retrieval tests
- **`test_operational_insights.py`** - Operational insights tests
- **`test_facts_simple.py`** - Simple fact tests

## Evaluation Data

### Test Scenarios

- **`test_scenarios.json`** - Ground truth test scenarios with expected results

### Reports

- **`offline_evaluation_report.txt`** - Main evaluation report
- **`offline_evaluation_report_with_methodology.txt`** - Report with detailed methodology
- **`offline_evaluation_report.json`** - JSON format report

### Metrics

- **`evaluation_metrics.json`** - Computed metrics in JSON format
- **`evaluation_metrics_summary.txt`** - Summary statistics
- **`query_metrics_table.csv`** - Per-query metrics table

## Documentation

### Main Documentation

- **`EVALUATION_METRICS_SUMMARY.md`** - Summary of evaluation results and methodology
- **`HOW_TO_TEST_QUERIES.md`** - Detailed testing guide
- **`OFFLINE_TESTING_GUIDE.md`** - Offline testing instructions

### Reference Documentation

- **`EVALUATION_METRICS_GUIDE.md`** - Detailed metrics explanation
- **`EVALUATION_METRICS_FOR_GRAPHS.md`** - Metrics formatted for visualization
- **`GROUND_TRUTH_EXPLANATION.md`** - Ground truth definition methodology
- **`EVIDENCE_RETRIEVAL_GUIDE.md`** - Evidence retrieval evaluation guide
- **`ALL_TESTED_QUERIES.md`** - List of all tested queries
- **`QUERIES_WITH_RESPONSES.md`** - Query-response pairs

## Evaluation Metrics

The system evaluates the following metrics:

1. **Accuracy** - Query-level correctness (binary: correct/incorrect)
2. **Precision/Recall/F1** - Fact retrieval quality (for evidence queries)
3. **Traceability Completeness** - Proportion of required facts shown in evidence
4. **Hallucination Resistance** - Proportion of claims supported by evidence
5. **Response Latency** - End-to-end response time

See `EVALUATION_METRICS_SUMMARY.md` for detailed methodology and results.

## Requirements

All evaluation scripts require:
- Python 3.9+
- Knowledge graph loaded (`knowledge.py`)
- Test scenarios file (`test_scenarios.json`)
- System dependencies (see main `requirements.txt`)

## Notes

- Evaluation is fully automated - no manual annotation required
- Gold facts are automatically estimated from responses (see methodology)
- All metrics are computed programmatically for reproducibility
- Results are hardware-dependent (latency measurements)

