"""
Create Summary Document with Metrics Values for Graphing
========================================================

Reads the evaluation_metrics.json and creates a formatted summary
with values ready for plotting (mean ± 95% CI).
"""

import json
import sys


def format_metric_value(mean: float, ci_lower: float, ci_upper: float, decimals: int = 3) -> str:
    """Format metric as mean ± error."""
    error = ci_upper - mean
    return f"{mean:.{decimals}f} ± {error:.{decimals}f}"


def create_summary_document(metrics_file: str, output_file: str):
    """Create formatted summary document."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    aggregated = data.get('aggregated_metrics', {})
    graph_data = data.get('graph_data', {})
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EVALUATION METRICS SUMMARY - VALUES FOR GRAPHING")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("This document contains computed metrics from the offline evaluation report.")
    summary_lines.append("All values are presented as Mean ± 95% Confidence Interval.")
    summary_lines.append("")
    
    # Summary by group
    summary_lines.append("METRICS BY GROUP")
    summary_lines.append("-" * 80)
    summary_lines.append("")
    
    for group_name, metrics in aggregated.items():
        summary_lines.append(f"Group: {group_name.upper()}")
        summary_lines.append(f"  Sample size (n): {metrics['n_queries']}")
        summary_lines.append("")
        summary_lines.append("  Fact Retrieval Accuracy (Canonical Scoring):")
        summary_lines.append(f"    F1 Score:           {format_metric_value(metrics['f1']['mean'], metrics['f1']['ci_lower'], metrics['f1']['ci_upper'])}")
        summary_lines.append(f"    Precision:          {format_metric_value(metrics['precision']['mean'], metrics['precision']['ci_lower'], metrics['precision']['ci_upper'])}")
        summary_lines.append(f"    Recall:             {format_metric_value(metrics['recall']['mean'], metrics['recall']['ci_lower'], metrics['recall']['ci_upper'])}")
        summary_lines.append("")
        summary_lines.append("  Traceability & Reliability:")
        summary_lines.append(f"    Traceability Completeness: {format_metric_value(metrics['traceability_completeness']['mean'], metrics['traceability_completeness']['ci_lower'], metrics['traceability_completeness']['ci_upper'])}")
        summary_lines.append(f"    Hallucination Resistance:  {format_metric_value(metrics['hallucination_resistance']['mean'], metrics['hallucination_resistance']['ci_lower'], metrics['hallucination_resistance']['ci_upper'])}")
        summary_lines.append("")
        summary_lines.append("  Performance:")
        summary_lines.append(f"    Response Latency (seconds): {format_metric_value(metrics['latency']['mean'], metrics['latency']['ci_lower'], metrics['latency']['ci_upper'], decimals=2)}")
        summary_lines.append(f"    Accuracy:                   {format_metric_value(metrics['accuracy']['mean'], metrics['accuracy']['ci_lower'], metrics['accuracy']['ci_upper'])}")
        summary_lines.append("")
        summary_lines.append("-" * 80)
        summary_lines.append("")
    
    # Graph data section
    if graph_data:
        summary_lines.append("GRAPH DATA (for plotting)")
        summary_lines.append("-" * 80)
        summary_lines.append("")
        summary_lines.append("The following data is formatted for direct use in plotting libraries")
        summary_lines.append("(matplotlib, seaborn, plotly, etc.):")
        summary_lines.append("")
        
        # Create table format
        groups = graph_data.get('groups', [])
        metrics_dict = graph_data.get('metrics', {})
        
        # Table header
        header = "Metric".ljust(25)
        for group in groups:
            header += f" | {group[:15]:<15}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        # For each metric, show mean values
        for metric_name in ['f1', 'precision', 'recall', 'traceability_completeness', 
                           'hallucination_resistance', 'latency', 'accuracy']:
            if metric_name in metrics_dict:
                row = metric_name.replace('_', ' ').title().ljust(25)
                means = metrics_dict[metric_name].get('means', [])
                for i, group in enumerate(groups):
                    if i < len(means):
                        row += f" | {means[i]:.3f}".ljust(17)
                    else:
                        row += " | N/A".ljust(17)
                summary_lines.append(row)
        
        summary_lines.append("")
        summary_lines.append("Error bars (95% CI):")
        summary_lines.append("  Use 'ci_lowers' and 'ci_uppers' arrays from graph_data in JSON")
        summary_lines.append("")
    
    # Detailed per-query breakdown (optional, can be long)
    summary_lines.append("=" * 80)
    summary_lines.append("NOTES")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("1. Fact Retrieval Metrics:")
    summary_lines.append("   - Precision: |R_q ∩ G_q| / |R_q|")
    summary_lines.append("   - Recall: |R_q ∩ G_q| / |G_q|")
    summary_lines.append("   - F1: 2 * (Precision * Recall) / (Precision + Recall)")
    summary_lines.append("   - Uses canonical scoring (recommended)")
    summary_lines.append("")
    summary_lines.append("2. Traceability Completeness:")
    summary_lines.append("   - T_q / D_q (fact-level)")
    summary_lines.append("   - T_q = facts shown in evidence")
    summary_lines.append("   - D_q = required facts (estimated from ground truth)")
    summary_lines.append("")
    summary_lines.append("3. Hallucination Resistance:")
    summary_lines.append("   - 1 - (Hallucinated Claims / Total Claims)")
    summary_lines.append("   - Estimated from response correctness")
    summary_lines.append("")
    summary_lines.append("4. Reliability:")
    summary_lines.append("   - Requires multiple runs of same query")
    summary_lines.append("   - Not computed in this evaluation (single run per query)")
    summary_lines.append("")
    summary_lines.append("5. NASA-TLX:")
    summary_lines.append("   - Requires user study data")
    summary_lines.append("   - Not available in offline evaluation")
    summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    print(f"✅ Summary document created: {output_file}")
    return summary_text


if __name__ == '__main__':
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else 'evaluation_metrics.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'evaluation_metrics_summary.txt'
    
    create_summary_document(metrics_file, output_file)

