"""
CSV to Evidence-Based Facts Converter
=====================================

Converts CSV analysis into high-quality, evidence-based facts for the knowledge graph.
Focuses on statistics, patterns, and insights rather than individual row data.

Author: Research Brain Team
Last Updated: 2025-01-15
"""

from typing import Dict, List, Any, Optional
from csv_analysis import analyze_csv
import pandas as pd

def create_evidence_based_facts_from_csv(file_path: str, document_name: str) -> str:
    """
    Convert CSV analysis into structured, evidence-based facts text.
    
    Instead of extracting facts from individual rows, this creates facts from:
    - Dataset-level statistics
    - Column-level statistics and distributions
    - Correlations and relationships
    - Data quality insights
    - Patterns and trends
    
    Returns formatted text that will be processed by knowledge extraction.
    """
    try:
        analysis = analyze_csv(file_path)
        
        if analysis.get("error"):
            error_msg = analysis.get('error', 'Unknown error')
            print(f"⚠️  CSV analysis error for {document_name}: {error_msg}")
            # Return basic info instead of error to allow processing to continue
            try:
                df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
                if len(df.columns) == 1:
                    df = pd.read_csv(file_path, sep=',', encoding='utf-8', on_bad_lines='skip', engine='python')
                return f"CSV Dataset: {len(df)} rows, {len(df.columns)} columns. Columns: {', '.join(df.columns[:10])}"
            except:
                return f"CSV Dataset: {document_name} (analysis failed: {error_msg})"
        
        summary = analysis.get("summary", {})
        columns = analysis.get("columns", {})
        correlations = analysis.get("correlations", {})
        insights = analysis.get("insights", [])
        
        facts_text = f"Dataset Analysis for {document_name}:\n\n"
        
        # Dataset-level facts
        facts_text += f"Dataset Statistics:\n"
        facts_text += f"- Total rows: {summary.get('total_rows', 0)}\n"
        facts_text += f"- Total columns: {summary.get('total_columns', 0)}\n"
        facts_text += f"- Column names: {', '.join(summary.get('column_names', []))}\n\n"
        
        # Column-level statistical facts
        facts_text += "Column Statistics:\n"
        for col_name, col_stats in columns.items():
            col_type = col_stats.get("type")
            
            if col_type == "numeric":
                facts_text += f"\n{col_name} (numeric):\n"
                if col_stats.get("mean") is not None:
                    facts_text += f"- Mean: {col_stats.get('mean'):.2f}\n"
                if col_stats.get("median") is not None:
                    facts_text += f"- Median: {col_stats.get('median'):.2f}\n"
                if col_stats.get("std") is not None:
                    facts_text += f"- Standard deviation: {col_stats.get('std'):.2f}\n"
                if col_stats.get("min") is not None:
                    facts_text += f"- Minimum: {col_stats.get('min')}\n"
                if col_stats.get("max") is not None:
                    facts_text += f"- Maximum: {col_stats.get('max')}\n"
                null_pct = col_stats.get("null_percentage", 0)
                if null_pct > 0:
                    facts_text += f"- Missing data: {null_pct:.1f}%\n"
                outlier_pct = col_stats.get("outlier_percentage", 0)
                if outlier_pct > 0:
                    facts_text += f"- Outliers: {outlier_pct:.1f}% ({col_stats.get('outlier_count', 0)} values)\n"
            
            elif col_type == "categorical":
                facts_text += f"\n{col_name} (categorical):\n"
                facts_text += f"- Unique values: {col_stats.get('unique_count', 0)}\n"
                mode = col_stats.get("mode")
                mode_pct = col_stats.get("mode_percentage", 0)
                if mode:
                    facts_text += f"- Most common: {mode} ({mode_pct:.1f}%)\n"
                null_pct = col_stats.get("null_percentage", 0)
                if null_pct > 0:
                    facts_text += f"- Missing data: {null_pct:.1f}%\n"
                # Top values
                top_values = col_stats.get("top_values", [])[:3]
                if top_values:
                    facts_text += f"- Top values:\n"
                    for tv in top_values:
                        facts_text += f"  * {tv.get('value')}: {tv.get('percentage'):.1f}%\n"
            
            elif col_type == "datetime":
                facts_text += f"\n{col_name} (datetime):\n"
                if col_stats.get("earliest"):
                    facts_text += f"- Earliest: {col_stats.get('earliest')}\n"
                if col_stats.get("latest"):
                    facts_text += f"- Latest: {col_stats.get('latest')}\n"
                span_days = col_stats.get("span_days", 0)
                if span_days > 0:
                    facts_text += f"- Time span: {span_days} days\n"
        
        # Correlation facts
        strong_corrs = correlations.get("strong_correlations", [])
        if strong_corrs:
            facts_text += "\n\nStrong Correlations:\n"
            for corr in strong_corrs[:5]:  # Top 5 correlations
                facts_text += f"- {corr.get('column1')} and {corr.get('column2')}: "
                facts_text += f"correlation {corr.get('correlation'):.3f} ({corr.get('strength')})\n"
        
        # Evidence-based insights
        if insights:
            facts_text += "\n\nKey Insights:\n"
            for insight in insights[:5]:  # Top 5 insights
                message = insight.get("message", "")
                evidence = insight.get("evidence", {})
                facts_text += f"- {message}\n"
                if isinstance(evidence, dict) and evidence:
                    # Add key evidence points
                    for key, value in list(evidence.items())[:2]:
                        facts_text += f"  Evidence: {key}: {value}\n"
        
        return facts_text.strip()
    
    except Exception as e:
        return f"Error creating facts from CSV analysis: {str(e)}"

def create_structured_facts_from_csv_analysis(analysis: Dict[str, Any], document_name: str) -> List[Dict[str, Any]]:
    """
    Create structured fact objects directly from CSV analysis.
    FOCUS ON INSIGHTS AND DECISION-SUPPORTING FACTS, not raw statistics.
    
    Returns list of fact dictionaries with subject, predicate, object, and metadata.
    """
    facts = []
    
    if analysis.get("error"):
        return facts
    
    summary = analysis.get("summary", {})
    columns = analysis.get("columns", {})
    correlations = analysis.get("correlations", {})
    insights = analysis.get("insights", [])
    
    # Dataset-level facts - only key metrics
    dataset_entity = f"Dataset_{document_name.replace('.csv', '').replace(' ', '_')}"
    
    # Only add dataset size if meaningful
    total_rows = summary.get("total_rows", 0)
    if total_rows > 0:
        facts.append({
            "subject": dataset_entity,
            "predicate": "has_total_rows",
            "object": str(total_rows),
            "source": document_name,
            "type": "statistic",
            "confidence": 1.0
        })
    
    # PRIORITY: Add insights first - these are most valuable for decision-making
    for insight in insights[:15]:  # Top 15 insights (most important)
        message = insight.get("message", "")
        category = insight.get("category", "general")
        evidence = insight.get("evidence", {})
        
        # Create a more structured insight fact
        insight_text = message
        if evidence:
            # Add key evidence numbers
            evidence_parts = []
            for key, value in list(evidence.items())[:2]:  # Top 2 evidence points
                if isinstance(value, (int, float)):
                    evidence_parts.append(f"{key}: {value}")
                elif isinstance(value, str) and len(value) < 50:
                    evidence_parts.append(f"{key}: {value}")
            
            if evidence_parts:
                insight_text = f"{message} ({', '.join(evidence_parts)})"
        
        facts.append({
            "subject": dataset_entity,
            "predicate": "has_insight",
            "object": insight_text,
            "source": document_name,
            "type": "insight",
            "category": category,
            "confidence": 0.95,  # High confidence for insights
            "details": str(evidence)
        })
    
    # Add strong correlations (only strong ones - actionable)
    strong_corrs = correlations.get("strong_correlations", [])
    for corr in strong_corrs[:8]:  # Top 8 strong correlations
        col1 = corr.get("column1", "")
        col2 = corr.get("column2", "")
        corr_value = corr.get("correlation", 0)
        strength = corr.get("strength", "")
        
        # Only add if correlation is meaningful (|r| > 0.5)
        if abs(corr_value) > 0.5:
            col1_clean = col1.replace(" ", "_").replace("-", "_")
            col2_clean = col2.replace(" ", "_").replace("-", "_")
            
            facts.append({
                "subject": f"{dataset_entity}_{col1_clean}",
                "predicate": "strongly_correlates_with",
                "object": f"{dataset_entity}_{col2_clean}",
                "source": document_name,
                "type": "relationship",
                "confidence": abs(corr_value),
                "details": f"{col1} and {col2} have {strength} correlation ({corr_value:.3f})"
            })
    
    # Add key column insights (only for important columns with meaningful patterns)
    for col_name, col_stats in columns.items():
        col_type = col_stats.get("type")
        
        # Skip if column has no meaningful data
        null_pct = col_stats.get("null_percentage", 0)
        if null_pct > 50:  # Skip columns with >50% missing data
            continue
        
        col_entity = f"{dataset_entity}_{col_name.replace(' ', '_').replace('-', '_')}"
        
        # For numeric columns: only add if there are interesting patterns
        if col_type == "numeric":
            mean_val = col_stats.get("mean")
            std_val = col_stats.get("std")
            outlier_pct = col_stats.get("outlier_percentage", 0)
            
            # Add if there are significant outliers (data quality issue)
            if outlier_pct > 5:
                facts.append({
                    "subject": col_entity,
                    "predicate": "has_data_quality_issue",
                    "object": f"{outlier_pct:.1f}% outliers detected",
                    "source": document_name,
                    "type": "data_quality",
                    "confidence": 0.9,
                    "details": f"Mean: {mean_val:.2f}, Std: {std_val:.2f}"
                })
            
            # Add distribution insight if meaningful
            if mean_val is not None and std_val is not None:
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                if cv > 50:  # High variability
                    facts.append({
                        "subject": col_entity,
                        "predicate": "has_high_variability",
                        "object": f"Coefficient of variation: {cv:.1f}%",
                        "source": document_name,
                        "type": "insight",
                        "confidence": 0.85
                    })
        
        # For categorical columns: add distribution insights
        elif col_type == "categorical":
            mode = col_stats.get("mode")
            mode_pct = col_stats.get("mode_percentage", 0)
            unique_count = col_stats.get("unique_count", 0)
            
            # Add if there's a dominant category (>60%)
            if mode and mode_pct > 60:
                facts.append({
                    "subject": col_entity,
                    "predicate": "has_dominant_category",
                    "object": f"{mode} ({mode_pct:.1f}%)",
                    "source": document_name,
                    "type": "insight",
                    "confidence": 0.9,
                    "details": f"{unique_count} unique values"
                })
            
            # Add if highly diverse (many categories, no dominant one)
            elif unique_count > 10 and mode_pct < 20:
                facts.append({
                    "subject": col_entity,
                    "predicate": "has_high_diversity",
                    "object": f"{unique_count} categories, most common: {mode} ({mode_pct:.1f}%)",
                    "source": document_name,
                    "type": "insight",
                    "confidence": 0.85
                })
    
    return facts

