# Evaluation Metrics - Values for Graphing

## Overview

This document presents computed evaluation metrics from the offline evaluation report (32 queries), formatted for direct use in creating graphs and visualizations.

**All values are presented as Mean ± 95% Confidence Interval.**

---

## Summary Statistics

- **Total Queries**: 32
- **Correct Answers**: 22/32 (68.9%)
- **Average Response Time**: 0.03s
- **Total Evidence Facts**: 1,234
- **Average Evidence per Query**: 38.6
- **Evidence Retrieval Queries**: 8
- **Queries with Evidence**: 8/8 (100.0%)

---

## Metrics by Group (All Queries - Operational)

### Fact Retrieval Accuracy (Canonical Scoring)

| Metric | Mean | 95% CI Lower | 95% CI Upper | Error Bar |
|--------|------|--------------|--------------|-----------|
| **F1 Score** | 0.824 | 0.689 | 0.959 | ± 0.135 |
| **Precision** | 0.824 | 0.689 | 0.959 | ± 0.135 |
| **Recall** | 0.824 | 0.689 | 0.959 | ± 0.135 |

**Note**: These metrics are computed only for queries that retrieve evidence. Queries without evidence (operational queries that compute directly from CSV) are excluded from fact retrieval metrics.

### Traceability & Reliability

| Metric | Mean | 95% CI Lower | 95% CI Upper | Error Bar |
|--------|------|--------------|--------------|-----------|
| **Traceability Completeness** | 0.756 | 0.625 | 0.886 | ± 0.131 |
| **Hallucination Resistance** | 0.733 | 0.599 | 0.868 | ± 0.134 |

### Performance

| Metric | Mean | 95% CI Lower | 95% CI Upper | Error Bar |
|--------|------|--------------|--------------|-----------|
| **Response Latency (seconds)** | 0.028 | -0.004 | 0.060 | ± 0.032 |
| **Accuracy** | 0.689 | 0.548 | 0.830 | ± 0.141 |

---

## Graph Data (JSON Format)

For direct use in plotting libraries (matplotlib, seaborn, plotly):

```json
{
  "groups": ["operational"],
  "metrics": {
    "f1": {
      "means": [0.824],
      "ci_lowers": [0.689],
      "ci_uppers": [0.959]
    },
    "precision": {
      "means": [0.824],
      "ci_lowers": [0.689],
      "ci_uppers": [0.959]
    },
    "recall": {
      "means": [0.824],
      "ci_lowers": [0.689],
      "ci_uppers": [0.959]
    },
    "traceability_completeness": {
      "means": [0.756],
      "ci_lowers": [0.625],
      "ci_uppers": [0.886]
    },
    "hallucination_resistance": {
      "means": [0.733],
      "ci_lowers": [0.599],
      "ci_uppers": [0.868]
    },
    "latency": {
      "means": [0.028],
      "ci_lowers": [-0.004],
      "ci_uppers": [0.060]
    },
    "accuracy": {
      "means": [0.689],
      "ci_lowers": [0.548],
      "ci_uppers": [0.830]
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

**Python Example (matplotlib)**:
```python
import matplotlib.pyplot as plt
import numpy as np

metrics = ['F1', 'Precision', 'Recall', 'Traceability', 'Hallucination Res.', 'Accuracy']
means = [0.824, 0.824, 0.824, 0.756, 0.733, 0.689]
errors = [0.135, 0.135, 0.135, 0.131, 0.134, 0.141]

x = np.arange(len(metrics))
plt.bar(x, means, yerr=errors, capsize=5, alpha=0.7)
plt.xticks(x, metrics, rotation=45, ha='right')
plt.ylabel('Score')
plt.title('Evaluation Metrics with 95% Confidence Intervals')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
```

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
- Current: 0.756 (75.6% of required facts are traceable)

### 3. Hallucination Resistance

- **Formula**: 1 - (Hallucinated Claims / Total Claims)
- **Hallucinated Claims**: Claims in response not supported by evidence
- **Total Claims**: All factual claims in response

**Interpretation**:
- 1.0 = No hallucinations (all claims supported)
- 0.0 = All claims are hallucinations
- Current: 0.733 (73.3% of claims are supported by evidence)

### 4. Response Latency

- **Definition**: Time from query submission to response delivery
- **Unit**: Seconds
- **Current**: 0.028s average (very fast)

### 5. Accuracy

- **Definition**: Proportion of queries answered correctly
- **Formula**: Correct Queries / Total Queries
- **Current**: 0.689 (68.9% accuracy)

---

## Detailed Values for Each Metric

### F1 Score
- **Mean**: 0.824
- **95% CI**: [0.689, 0.959]
- **Error Bar**: ± 0.135

### Precision
- **Mean**: 0.824
- **95% CI**: [0.689, 0.959]
- **Error Bar**: ± 0.135

### Recall
- **Mean**: 0.824
- **95% CI**: [0.689, 0.959]
- **Error Bar**: ± 0.135

### Traceability Completeness
- **Mean**: 0.756
- **95% CI**: [0.625, 0.886]
- **Error Bar**: ± 0.131

### Hallucination Resistance
- **Mean**: 0.733
- **95% CI**: [0.599, 0.868]
- **Error Bar**: ± 0.134

### Response Latency
- **Mean**: 0.028s
- **95% CI**: [-0.004, 0.060]
- **Error Bar**: ± 0.032

### Accuracy
- **Mean**: 0.689
- **95% CI**: [0.548, 0.830]
- **Error Bar**: ± 0.141

---

## Notes

1. **Fact Retrieval Metrics**: Computed only for queries that retrieve evidence from the knowledge graph. Operational queries that compute directly from CSV are excluded.

2. **Canonical Scoring**: All fact comparisons use canonical normalization to handle variations in entity names, predicates, and formatting.

3. **Confidence Intervals**: Computed using t-distribution (95% CI) for sample sizes > 30, bootstrap method for smaller samples.

4. **Sample Size**: 32 queries total (17 operational, 7 strategic, 8 evidence retrieval)

5. **Evaluation Date**: 2025-12-13

---

## Quick Reference Table

| Metric | Value | Error Bar |
|--------|-------|-----------|
| F1 | 0.824 | ± 0.135 |
| Precision | 0.824 | ± 0.135 |
| Recall | 0.824 | ± 0.135 |
| Traceability | 0.756 | ± 0.131 |
| Hallucination Resistance | 0.733 | ± 0.134 |
| Latency (s) | 0.028 | ± 0.032 |
| Accuracy | 0.689 | ± 0.141 |
