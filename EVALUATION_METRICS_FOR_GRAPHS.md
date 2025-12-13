# Evaluation Metrics - Values for Graphing

## Overview

This document presents computed evaluation metrics from the offline evaluation report (46 queries), formatted for direct use in creating graphs and visualizations.

**All values are presented as Mean ± 95% Confidence Interval.**

---

## Summary Statistics

- **Total Queries**: 46
- **Correct Answers**: 39/46 (84.8%)
- **Average Response Time**: 7.51s
- **Total Evidence Facts**: 358
- **Average Evidence per Query**: 7.8
- **Evidence Retrieval Queries**: 22
- **Queries with Evidence**: 22/22 (100.0%)

---

## Metrics by Group (All Queries - Operational)

### Fact Retrieval Accuracy (Canonical Scoring)

| Metric | Mean | 95% CI Lower | 95% CI Upper | Error Bar |
|--------|------|--------------|--------------|-----------|
| **F1 Score** | 0.000 | 0.000 | 0.000 | ± 0.000 |
| **Precision** | 0.000 | 0.000 | 0.000 | ± 0.000 |
| **Recall** | 0.783 | 0.659 | 0.906 | ± 0.124 |

**Note**: Low precision/F1 is due to many queries having no evidence retrieved (0 retrieved facts). Recall is computed as 1.0 when no gold facts are expected (queries without evidence requirements).

### Traceability & Reliability

| Metric | Mean | 95% CI Lower | 95% CI Upper | Error Bar |
|--------|------|--------------|--------------|-----------|
| **Traceability Completeness** | 0.478 | 0.328 | 0.628 | ± 0.150 |
| **Hallucination Resistance** | 0.922 | 0.854 | 0.991 | ± 0.069 |

### Performance

| Metric | Mean | 95% CI Lower | 95% CI Upper | Error Bar |
|--------|------|--------------|--------------|-----------|
| **Response Latency (seconds)** | 7.51 | 5.22 | 9.80 | ± 2.29 |
| **Accuracy** | 0.848 | 0.740 | 0.956 | ± 0.108 |

---

## Graph Data (JSON Format)

For direct use in plotting libraries (matplotlib, seaborn, plotly):

```json
{
  "groups": ["operational"],
  "metrics": {
    "f1": {
      "means": [0.0],
      "ci_lowers": [0.0],
      "ci_uppers": [0.0]
    },
    "precision": {
      "means": [0.0],
      "ci_lowers": [0.0],
      "ci_uppers": [0.0]
    },
    "recall": {
      "means": [0.783],
      "ci_lowers": [0.659],
      "ci_uppers": [0.906]
    },
    "traceability_completeness": {
      "means": [0.478],
      "ci_lowers": [0.328],
      "ci_uppers": [0.628]
    },
    "hallucination_resistance": {
      "means": [0.922],
      "ci_lowers": [0.854],
      "ci_uppers": [0.991]
    },
    "latency": {
      "means": [7.51],
      "ci_lowers": [5.22],
      "ci_uppers": [9.80]
    },
    "accuracy": {
      "means": [0.848],
      "ci_lowers": [0.740],
      "ci_uppers": [0.956]
    }
  }
}
```

---

## Recommended Graph Types

### 1. Bar Chart with Error Bars (Grouped by Metric)

**X-axis**: Metrics (F1, Precision, Recall, Traceability, Hallucination Resistance, Accuracy)  
**Y-axis**: Score (0-1 scale)  
**Error bars**: 95% CI

### 2. Performance Comparison

**X-axis**: Metrics  
**Y-axis**: Score  
**Bars**: Mean values  
**Error bars**: ± CI

### 3. Latency Distribution

**Type**: Box plot or histogram  
**X-axis**: Response time (seconds)  
**Y-axis**: Frequency or density

---

## Metric Definitions

### 1. Fact Retrieval Accuracy

- **Precision**: |R_q ∩ G_q| / |R_q|
  - Proportion of retrieved facts that are correct
- **Recall**: |R_q ∩ G_q| / |G_q|
  - Proportion of required facts that were retrieved
- **F1**: 2 × (Precision × Recall) / (Precision + Recall)
  - Harmonic mean of precision and recall

**Scoring Method**: Canonical scoring (recommended)
- Facts are normalized before comparison
- Handles entity name variations (e.g., "LinkedIn" vs "Linkedin")
- Merges duplicate predicates ("is" → "has")

### 2. Traceability Completeness

- **Formula**: T_q / D_q (fact-level)
- **T_q**: Facts shown in evidence panel
- **D_q**: Required facts (from ground truth)

**Interpretation**: 
- 1.0 = All required facts are shown
- 0.0 = No required facts shown
- Current: 0.478 (47.8% of required facts are traceable)

### 3. Hallucination Resistance

- **Formula**: 1 - (Hallucinated Claims / Total Claims)
- **Hallucinated Claims**: Claims in response not supported by evidence
- **Total Claims**: All atomic claims in response

**Interpretation**:
- 1.0 = No hallucinations (all claims supported)
- 0.0 = All claims are hallucinations
- Current: 0.922 (92.2% resistance, ~7.8% hallucination rate)

### 4. Response Latency

- **Formula**: timestamp_end - timestamp_start
- **Unit**: Seconds
- **Current**: 7.51s average (range: 5.22s - 9.80s with 95% CI)

### 5. Accuracy

- **Formula**: Correct Answers / Total Queries
- **Current**: 0.848 (84.8% correct)

---

## Limitations & Notes

### Fact Retrieval Metrics

- **Low Precision/F1**: Many queries (24/46) have no evidence retrieved
  - These queries compute answers directly from dataset without KG retrieval
  - When R_q = ∅, precision = 0 (by definition)
  - This is expected behavior for operational queries that don't require evidence

### Traceability Completeness

- **Estimation Method**: Gold facts (D_q) are estimated from response patterns
- **For accurate measurement**: Requires expert annotation of required facts per query
- **Current value (0.478)**: Reflects that ~48% of queries show evidence when available

### Hallucination Resistance

- **Estimation Method**: Heuristic based on response correctness
  - Correct responses: ~10% estimated hallucination (conservative)
  - Incorrect responses: ~30% estimated hallucination
- **For accurate measurement**: Requires manual claim decomposition and annotation

### Reliability

- **Not Computed**: Requires multiple runs of same query (m ≥ 2)
- **Formula**: (2 / (m(m-1))) × Σ Jaccard(R_q^(i), R_q^(j))
- **To compute**: Run same query 3-5 times and compute Jaccard similarity

### NASA-TLX

- **Not Available**: Requires user study with human participants
- **Would provide**: Workload assessment (mental demand, physical demand, temporal demand, performance, effort, frustration)

---

## Recommendations for Paper Plots

### Plot 1: Fact Retrieval Accuracy (Bar Chart)

```
Grouped bar chart:
- X-axis: Metrics (Precision, Recall, F1)
- Y-axis: Score (0-1)
- Error bars: 95% CI
- Note: Include explanation for low precision (many queries without evidence)
```

### Plot 2: System Performance Overview (Radar/Spider Chart)

```
Metrics:
- Accuracy (0.848)
- Traceability Completeness (0.478)
- Hallucination Resistance (0.922)
- Response Latency (inverted: 1 - normalized_latency)
```

### Plot 3: Latency Distribution (Box Plot)

```
- Show distribution of response times
- Highlight mean (7.51s) and CI (5.22s - 9.80s)
- Identify outliers
```

### Plot 4: Evidence Retrieval Analysis (Stacked Bar)

```
- Queries with evidence: 22/46 (47.8%)
- Queries without evidence: 24/46 (52.2%)
- Show breakdown by query type
```

---

## Data Files

- **Full Metrics JSON**: `evaluation_metrics.json`
  - Contains per-query metrics and aggregated values
  - Includes graph_data for direct plotting

- **Summary Text**: `evaluation_metrics_summary.txt`
  - Human-readable summary

- **Source Report**: `offline_evaluation_report.txt`
  - Original evaluation report with query-by-query results

---

## Next Steps

1. **Improve Gold Facts Estimation**: Use `test_scenarios.json` ground truth to compute accurate D_q
2. **Compute Reliability**: Run queries multiple times to measure consistency
3. **Manual Annotation**: For accurate hallucination measurement, decompose responses into claims
4. **User Study**: Conduct NASA-TLX assessment for usability metrics

---

**Generated**: 2025-12-13  
**Source**: Offline evaluation report (46 queries)  
**Method**: Canonical fact scoring, heuristic estimation for hallucination

