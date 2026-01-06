#!/usr/bin/env python3
"""
Update evaluation metrics summary with correct 32-query results.
"""

import json
import sys

# Load metrics
with open('evaluation_metrics.json', 'r') as f:
    data = json.load(f)

aggregated = data.get('aggregated_metrics', {})
graph_data = data.get('graph_data', {})

# Create updated summary
summary_lines = [
    "==================================================================================",
    "EVALUATION METRICS SUMMARY - VALUES FOR GRAPHING",
    "==================================================================================",
    "",
    "This document contains computed metrics from the offline evaluation report.",
    "All values are presented as Mean ± 95% Confidence Interval.",
    "",
    "Total Queries: 32 (as per ALL_TESTED_QUERIES.md)",
    "",
    "METRICS BY GROUP",
    "----------------------------------------------------------------------------------",
    ""
]

# Add metrics for each group
for group_name, metrics in sorted(aggregated.items()):
    n = metrics.get('n_queries', 0)
    if n == 0:
        continue
    
    summary_lines.append(f"Group: {group_name.upper()}")
    summary_lines.append(f"  Sample size (n): {n}")
    summary_lines.append("")
    
    # Fact Retrieval Accuracy
    f1 = metrics.get('f1', {})
    precision = metrics.get('precision', {})
    recall = metrics.get('recall', {})
    
    if f1.get('n', 0) > 0:
        f1_mean = f1.get('mean', 0)
        f1_ci = f1.get('ci_upper', f1_mean) - f1_mean
        prec_mean = precision.get('mean', 0)
        prec_ci = precision.get('ci_upper', prec_mean) - prec_mean
        rec_mean = recall.get('mean', 0)
        rec_ci = recall.get('ci_upper', rec_mean) - rec_mean
        
        summary_lines.append("  Fact Retrieval Accuracy (Canonical Scoring):")
        summary_lines.append(f"    F1 Score:           {f1_mean:.3f} ± {f1_ci:.3f}")
        summary_lines.append(f"    Precision:          {prec_mean:.3f} ± {prec_ci:.3f}")
        summary_lines.append(f"    Recall:             {rec_mean:.3f} ± {rec_ci:.3f}")
        summary_lines.append("")
    
    # Traceability & Reliability
    trace = metrics.get('traceability_completeness', {})
    hall = metrics.get('hallucination_resistance', {})
    
    trace_mean = trace.get('mean', 0)
    trace_ci = trace.get('ci_upper', trace_mean) - trace_mean
    hall_mean = hall.get('mean', 0)
    hall_ci = hall.get('ci_upper', hall_mean) - hall_mean
    
    summary_lines.append("  Traceability & Reliability:")
    summary_lines.append(f"    Traceability Completeness: {trace_mean:.3f} ± {trace_ci:.3f}")
    summary_lines.append(f"    Hallucination Resistance:  {hall_mean:.3f} ± {hall_ci:.3f}")
    summary_lines.append("")
    
    # Performance
    latency = metrics.get('latency', {})
    accuracy = metrics.get('accuracy', {})
    
    lat_mean = latency.get('mean', 0)
    lat_ci = latency.get('ci_upper', lat_mean) - lat_mean
    acc_mean = accuracy.get('mean', 0)
    acc_ci = accuracy.get('ci_upper', acc_mean) - acc_mean
    
    summary_lines.append("  Performance:")
    summary_lines.append(f"    Response Latency (seconds): {lat_mean:.3f} ± {lat_ci:.3f}")
    summary_lines.append(f"    Accuracy:                  {acc_mean:.3f} ± {acc_ci:.3f}")
    summary_lines.append("")
    summary_lines.append("----------------------------------------------------------------------------------")
    summary_lines.append("")

# Add graph data section
summary_lines.extend([
    "GRAPH DATA (for plotting)",
    "----------------------------------------------------------------------------------",
    "",
    "The following data is formatted for direct use in plotting libraries",
    "(matplotlib, seaborn, plotly, etc.):",
    ""
])

# Create table format
if graph_data:
    groups = graph_data.get('groups', [])
    metrics_dict = graph_data.get('metrics', {})
    
    # Header
    header = "Metric                    | " + " | ".join(groups)
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    
    # Metrics rows
    metric_names = ['F1', 'Precision', 'Recall', 'Traceability Completeness', 
                   'Hallucination Resistance', 'Latency', 'Accuracy']
    metric_keys = ['f1', 'precision', 'recall', 'traceability_completeness',
                  'hallucination_resistance', 'latency', 'accuracy']
    
    for metric_name, metric_key in zip(metric_names, metric_keys):
        if metric_key in metrics_dict:
            values = metrics_dict[metric_key].get('means', [])
            row = f"{metric_name:24} | " + " | ".join(f"{v:.3f}" for v in values)
            summary_lines.append(row)
    
    summary_lines.append("")
    summary_lines.append("Error bars (95% CI):")
    summary_lines.append("  Use 'ci_lowers' and 'ci_uppers' arrays from graph_data in JSON")
    summary_lines.append("")

# Add notes
summary_lines.extend([
    "==================================================================================",
    "NOTES",
    "==================================================================================",
    "",
    "1. Fact Retrieval Metrics:",
    "   - Precision: |R_q ∩ G_q| / |R_q|",
    "   - Recall: |R_q ∩ G_q| / |G_q|",
    "   - F1: 2 * (Precision * Recall) / (Precision + Recall)",
    "   - Uses canonical scoring (recommended)",
    "",
    "2. Traceability Completeness:",
    "   - T_q / D_q (fact-level)",
    "   - T_q = facts shown in evidence",
    "   - D_q = required facts (estimated from ground truth)",
    "",
    "3. Hallucination Resistance:",
    "   - 1 - (Hallucinated Claims / Total Claims)",
    "   - Estimated from response correctness",
    "",
    "4. Reliability:",
    "   - Requires multiple runs of same query",
    "   - Not computed in this evaluation (single run per query)",
    "",
    "5. NASA-TLX:",
    "   - Requires user study data",
    "   - Not available in offline evaluation",
    ""
])

# Write summary
with open('evaluation_metrics_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))

print("✅ Updated evaluation_metrics_summary.txt with 32-query results")

