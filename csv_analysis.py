"""
CSV Analysis Module
===================

Evidence-based CSV analysis with statistics, patterns, and ontology enhancement.

Features:
- Column type detection (numeric, categorical, datetime, text)
- Statistical analysis (mean, median, mode, distributions, correlations)
- Pattern detection (outliers, missing values, duplicates)
- Evidence-based insights with traceability
- Ontology enhancement suggestions based on column structure

Author: Research Brain Team
Last Updated: 2025-01-15
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
from collections import Counter

def detect_column_type(series: pd.Series) -> str:
    """Detect the type of a column based on its values"""
    # Remove nulls for analysis
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"
    
    # Try datetime
    if series.dtype == 'datetime64[ns]':
        return "datetime"
    
    # Try numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    
    # Try datetime strings
    try:
        pd.to_datetime(non_null.head(10))
        return "datetime"
    except:
        pass
    
    # Check if categorical (limited unique values)
    unique_ratio = len(non_null.unique()) / len(non_null)
    if unique_ratio < 0.1 and len(non_null.unique()) < 50:
        return "categorical"
    
    # Check if boolean
    if series.dtype == 'bool' or set(non_null.astype(str).str.lower().unique()).issubset({'true', 'false', '1', '0', 'yes', 'no'}):
        return "boolean"
    
    # Default to text
    return "text"

def analyze_numeric_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Generate comprehensive statistics for numeric columns"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"type": "numeric", "status": "empty"}
    
    stats = {
        "type": "numeric",
        "count": len(series),
        "non_null_count": len(non_null),
        "null_count": len(series) - len(non_null),
        "null_percentage": round((len(series) - len(non_null)) / len(series) * 100, 2),
        "mean": float(non_null.mean()) if len(non_null) > 0 else None,
        "median": float(non_null.median()) if len(non_null) > 0 else None,
        "std": float(non_null.std()) if len(non_null) > 0 else None,
        "min": float(non_null.min()) if len(non_null) > 0 else None,
        "max": float(non_null.max()) if len(non_null) > 0 else None,
        "q25": float(non_null.quantile(0.25)) if len(non_null) > 0 else None,
        "q75": float(non_null.quantile(0.75)) if len(non_null) > 0 else None,
        "skewness": float(non_null.skew()) if len(non_null) > 0 else None,
        "kurtosis": float(non_null.kurtosis()) if len(non_null) > 0 else None,
    }
    
    # Detect outliers using IQR method
    if stats["q25"] is not None and stats["q75"] is not None:
        iqr = stats["q75"] - stats["q25"]
        lower_bound = stats["q25"] - 1.5 * iqr
        upper_bound = stats["q75"] + 1.5 * iqr
        outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
        stats["outlier_count"] = len(outliers)
        stats["outlier_percentage"] = round(len(outliers) / len(non_null) * 100, 2)
        stats["outlier_values"] = outliers.tolist()[:10]  # Sample of outliers
    
    # Distribution insights
    if stats["std"] is not None and stats["std"] > 0:
        cv = stats["std"] / abs(stats["mean"]) if stats["mean"] != 0 else None
        stats["coefficient_of_variation"] = round(cv, 2) if cv else None
        if cv:
            if cv < 0.1:
                stats["variability"] = "low"
            elif cv < 0.3:
                stats["variability"] = "moderate"
            else:
                stats["variability"] = "high"
    
    return stats

def analyze_categorical_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Generate comprehensive statistics for categorical columns"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"type": "categorical", "status": "empty"}
    
    value_counts = non_null.value_counts()
    mode_value = value_counts.index[0] if len(value_counts) > 0 else None
    mode_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
    
    stats = {
        "type": "categorical",
        "count": len(series),
        "non_null_count": len(non_null),
        "null_count": len(series) - len(non_null),
        "null_percentage": round((len(series) - len(non_null)) / len(series) * 100, 2),
        "unique_count": len(non_null.unique()),
        "mode": str(mode_value) if mode_value is not None else None,
        "mode_count": int(mode_count),
        "mode_percentage": round(mode_count / len(non_null) * 100, 2) if len(non_null) > 0 else 0,
        "top_values": [
            {"value": str(val), "count": int(count), "percentage": round(count / len(non_null) * 100, 2)}
            for val, count in value_counts.head(10).items()
        ]
    }
    
    # Detect duplicates
    duplicates = non_null[non_null.duplicated()]
    stats["duplicate_count"] = len(duplicates)
    stats["duplicate_percentage"] = round(len(duplicates) / len(non_null) * 100, 2) if len(non_null) > 0 else 0
    
    return stats

def analyze_datetime_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Generate comprehensive statistics for datetime columns"""
    # Try to convert to datetime
    try:
        datetime_series = pd.to_datetime(series, errors='coerce')
    except:
        return {"type": "datetime", "status": "conversion_failed"}
    
    non_null = datetime_series.dropna()
    if len(non_null) == 0:
        return {"type": "datetime", "status": "empty"}
    
    stats = {
        "type": "datetime",
        "count": len(series),
        "non_null_count": len(non_null),
        "null_count": len(series) - len(non_null),
        "null_percentage": round((len(series) - len(non_null)) / len(series) * 100, 2),
        "earliest": str(non_null.min()),
        "latest": str(non_null.max()),
        "span_days": (non_null.max() - non_null.min()).days if len(non_null) > 1 else 0,
    }
    
    # Time distribution
    if len(non_null) > 1:
        time_diffs = non_null.diff().dropna()
        stats["avg_time_diff_days"] = round(time_diffs.mean().total_seconds() / 86400, 2) if len(time_diffs) > 0 else None
    
    return stats

def analyze_text_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Generate comprehensive statistics for text columns"""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return {"type": "text", "status": "empty"}
    
    lengths = non_null.str.len()
    
    stats = {
        "type": "text",
        "count": len(series),
        "non_null_count": len(non_null),
        "null_count": len(series) - len(non_null),
        "null_percentage": round((len(series) - len(non_null)) / len(series) * 100, 2),
        "unique_count": len(non_null.unique()),
        "avg_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "median_length": float(lengths.median()),
    }
    
    # Detect duplicates
    duplicates = non_null[non_null.duplicated()]
    stats["duplicate_count"] = len(duplicates)
    stats["duplicate_percentage"] = round(len(duplicates) / len(non_null) * 100, 2) if len(non_null) > 0 else 0
    
    return stats

def compute_correlations(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """Compute correlations between numeric columns"""
    if len(numeric_columns) < 2:
        return {}
    
    numeric_df = df[numeric_columns].select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return {}
    
    corr_matrix = numeric_df.corr()
    
    # Find strong correlations (>0.7 or <-0.7)
    strong_correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicates
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(corr_value, 3),
                        "strength": "strong" if abs(corr_value) > 0.9 else "moderate"
                    })
    
    return {
        "matrix": corr_matrix.to_dict(),
        "strong_correlations": strong_correlations
    }

def generate_insights(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate evidence-based insights from analysis"""
    insights = []
    
    # Overall data quality insights
    total_rows = analysis.get("summary", {}).get("total_rows", 0)
    total_columns = analysis.get("summary", {}).get("total_columns", 0)
    
    if total_rows == 0:
        insights.append({
            "type": "warning",
            "category": "data_quality",
            "message": "Dataset is empty",
            "evidence": f"Total rows: {total_rows}"
        })
        return insights
    
    # Missing data insights
    columns_with_missing = []
    for col_name, col_stats in analysis.get("columns", {}).items():
        null_pct = col_stats.get("null_percentage", 0)
        if null_pct > 50:
            columns_with_missing.append({
                "column": col_name,
                "missing_percentage": null_pct
            })
    
    if columns_with_missing:
        insights.append({
            "type": "warning",
            "category": "data_quality",
            "message": f"{len(columns_with_missing)} column(s) have >50% missing data",
            "evidence": columns_with_missing,
            "recommendation": "Consider data imputation or column removal"
        })
    
    # Outlier insights
    columns_with_outliers = []
    for col_name, col_stats in analysis.get("columns", {}).items():
        if col_stats.get("type") == "numeric" and col_stats.get("outlier_count", 0) > 0:
            outlier_pct = col_stats.get("outlier_percentage", 0)
            if outlier_pct > 5:
                columns_with_outliers.append({
                    "column": col_name,
                    "outlier_count": col_stats.get("outlier_count"),
                    "outlier_percentage": outlier_pct
                })
    
    if columns_with_outliers:
        insights.append({
            "type": "info",
            "category": "data_distribution",
            "message": f"{len(columns_with_outliers)} column(s) contain significant outliers (>5%)",
            "evidence": columns_with_outliers,
            "recommendation": "Review outliers for data quality issues or special cases"
        })
    
    # Correlation insights
    strong_corrs = analysis.get("correlations", {}).get("strong_correlations", [])
    if strong_corrs:
        insights.append({
            "type": "info",
            "category": "relationships",
            "message": f"Found {len(strong_corrs)} strong correlation(s) between columns",
            "evidence": strong_corrs,
            "recommendation": "These columns may represent related concepts in the ontology"
        })
    
    # Categorical distribution insights
    for col_name, col_stats in analysis.get("columns", {}).items():
        if col_stats.get("type") == "categorical":
            mode_pct = col_stats.get("mode_percentage", 0)
            if mode_pct > 80:
                insights.append({
                    "type": "info",
                    "category": "data_distribution",
                    "message": f"Column '{col_name}' is highly imbalanced ({mode_pct}% one value)",
                    "evidence": {
                        "column": col_name,
                        "mode": col_stats.get("mode"),
                        "mode_percentage": mode_pct
                    },
                    "recommendation": "Consider if this column provides meaningful information"
                })
    
    return insights

def analyze_csv(file_path: str) -> Dict[str, Any]:
    """
    Comprehensive CSV analysis with evidence-based statistics
    
    Returns:
        Dictionary with:
        - summary: Overall statistics
        - columns: Per-column analysis
        - correlations: Column correlations
        - insights: Evidence-based insights
        - ontology_suggestions: Suggested ontology enhancements
    """
    try:
        # Try to detect separator - check first line for common separators
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            # Count different separators
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            tab_count = first_line.count('\t')
            pipe_count = first_line.count('|')
            
            # Determine separator (prefer semicolon if it's more common than comma)
            if semicolon_count > comma_count and semicolon_count > 0:
                sep = ';'
            elif tab_count > comma_count and tab_count > 0:
                sep = '\t'
            elif pipe_count > comma_count and pipe_count > 0:
                sep = '|'
            else:
                sep = ','  # Default to comma
        
        # Read CSV with detected separator
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
        
        # If we got only 1 column, try semicolon as fallback
        if len(df.columns) == 1 and ';' in str(df.columns[0]):
            print(f"⚠️  CSV appears to have semicolon separator, retrying...")
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
        
        analysis = {
            "file_path": file_path,
            "analyzed_at": datetime.now().isoformat(),
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            },
            "columns": {},
            "correlations": {},
            "insights": [],
            "ontology_suggestions": {
                "entities": [],
                "relationships": [],
                "properties": {}
            }
        }
        
        # Analyze each column
        numeric_columns = []
        for col_name in df.columns:
            series = df[col_name]
            col_type = detect_column_type(series)
            
            if col_type == "numeric":
                col_stats = analyze_numeric_column(series, col_name)
                numeric_columns.append(col_name)
            elif col_type == "categorical":
                col_stats = analyze_categorical_column(series, col_name)
            elif col_type == "datetime":
                col_stats = analyze_datetime_column(series, col_name)
            elif col_type == "text":
                col_stats = analyze_text_column(series, col_name)
            else:
                col_stats = {"type": col_type, "status": "not_analyzed"}
            
            col_stats["column_name"] = col_name
            analysis["columns"][col_name] = col_stats
        
        # Compute correlations
        if len(numeric_columns) >= 2:
            analysis["correlations"] = compute_correlations(df, numeric_columns)
        
        # Generate insights
        analysis["insights"] = generate_insights(analysis)
        
        # Generate ontology suggestions based on column structure
        analysis["ontology_suggestions"] = suggest_ontology_from_csv(df, analysis)
        
        return analysis
    
    except Exception as e:
        return {
            "error": str(e),
            "file_path": file_path,
            "analyzed_at": datetime.now().isoformat()
        }

def suggest_ontology_from_csv(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest ontology enhancements based on CSV structure
    
    Analyzes column names and data patterns to suggest:
    - Entity types (based on ID columns, categorical columns)
    - Relationships (based on foreign key patterns, correlations)
    - Properties (based on column types and names)
    """
    suggestions = {
        "entities": [],
        "relationships": [],
        "properties": {}
    }
    
    column_names = df.columns.tolist()
    
    # Detect potential entity types from column names - MORE AGGRESSIVE
    for col_name in column_names:
        col_lower = col_name.lower().strip()
        
        # Skip if column name is too long or contains semicolons (likely parsing error)
        if len(col_name) > 100 or ";" in col_name:
            continue
        
        # Check if column name suggests an entity type (more comprehensive)
        entity_patterns = {
            "Employee": ["employee", "emp", "staff", "worker", "personnel", "attrition"],
            "Person": ["person", "people", "individual", "user", "customer", "client"],
            "Department": ["department", "dept", "division", "unit", "team", "group", "business"],
            "Project": ["project", "task", "assignment", "initiative"],
            "Skill": ["skill", "competency", "ability", "capability"],
            "Position": ["position", "role", "job", "title", "jobrole"],
            "Location": ["location", "office", "site", "place", "city", "country"],
            "Company": ["company", "organization", "org", "firm", "business"],
            "Education": ["education", "educationfield", "degree"],
            "Environment": ["environmentsatisfaction", "environmentsatisfaction"],
            "WorkLife": ["worklifebalance", "worklife"],
            "JobSatisfaction": ["jobsatisfaction", "jobsatisfaction"],
            "MaritalStatus": ["maritalstatus", "marital"],
            "OverTime": ["overtime", "overtime"],
        }
        
        for entity_name, keywords in entity_patterns.items():
            if any(keyword in col_lower for keyword in keywords):
                if entity_name not in suggestions["entities"]:
                    suggestions["entities"].append(entity_name)
                    print(f"  📌 Detected entity '{entity_name}' from column '{col_name}'")
        
        # Check for ID columns (potential entity identifiers) - MORE AGGRESSIVE
        # Match any column ending with _id, id, or starting with id_
        if ("_id" in col_lower or col_lower.endswith("id") or col_lower.startswith("id_") or 
            "id" in col_lower.split("_") or col_lower.count("id") > 0):
            # Extract base name - try multiple patterns
            base_name = col_lower
            for pattern in ["_id", "id_", "id"]:
                if pattern in base_name:
                    base_name = base_name.replace(pattern, " ").strip()
            
            # Clean up
            base_name = base_name.replace("_", " ").strip()
            if base_name and base_name not in ["", "id"] and len(base_name) > 1:
                # Capitalize properly (handle snake_case and camelCase)
                parts = base_name.split()
                entity_name = "".join(word.capitalize() for word in parts)
                if entity_name not in suggestions["entities"] and len(entity_name) > 1:
                    suggestions["entities"].append(entity_name)
                    print(f"  📌 Detected entity '{entity_name}' from ID column '{col_name}'")
        
        # Check for categorical columns that might represent entities - MORE AGGRESSIVE
        col_stats = analysis.get("columns", {}).get(col_name, {})
        if col_stats.get("type") == "categorical":
            unique_count = col_stats.get("unique_count", 0)
            # If categorical with reasonable number of unique values, might be an entity
            if 2 <= unique_count <= 200:  # Increased threshold
                # More lenient - if it's categorical and has a meaningful name, consider it
                if len(col_name) > 3:  # Not too short
                    # Extract entity name from column (capitalize properly)
                    entity_name = col_name.replace("_", " ").replace("-", " ").title().replace(" ", "")
                    # Clean up common suffixes
                    for suffix in ["Id", "Id", "Type", "Category", "Status", "Level"]:
                        if entity_name.endswith(suffix) and len(entity_name) > len(suffix):
                            entity_name = entity_name[:-len(suffix)]
                    
                    if entity_name and entity_name not in suggestions["entities"] and len(entity_name) > 2:
                        suggestions["entities"].append(entity_name)
                        print(f"  📌 Detected entity '{entity_name}' from categorical column '{col_name}' ({unique_count} unique values)")
        
        # Also check column name patterns - if it looks like an entity name
        # Skip if column name is too long or contains semicolons (parsing error)
        if len(col_name) > 100 or ";" in col_name:
            continue
            
        if col_lower not in ["id", "name", "value", "data", "column"]:
            # If column name is capitalized or has multiple words, might be an entity
            if col_name[0].isupper() or "_" in col_name or "-" in col_name:
                # Check if it's not a common property name
                property_keywords = ["age", "salary", "date", "time", "count", "total", "sum", "avg", "min", "max"]
                if not any(prop in col_lower for prop in property_keywords):
                    entity_name = col_name.replace("_", " ").replace("-", " ").title().replace(" ", "")
                    # Filter out bad entity names (too long, contains special chars)
                    if entity_name and entity_name not in suggestions["entities"] and len(entity_name) > 2 and len(entity_name) < 50:
                        suggestions["entities"].append(entity_name)
                        print(f"  📌 Detected entity '{entity_name}' from column name pattern '{col_name}'")
    
    # Detect relationships from column patterns
    for col_name in column_names:
        col_lower = col_name.lower().strip()
        
        # Foreign key patterns - more comprehensive
        if "_id" in col_lower or col_lower.endswith("id") or col_lower.startswith("id_"):
            base_name = col_lower.replace("_id", "").replace("id", "").replace("id_", "").strip()
            if base_name and base_name not in ["", "id"] and len(base_name) > 1:
                # Create relationship name
                parts = base_name.split("_")
                rel_base = "_".join(parts)
                rel_name = f"has_{rel_base}"
                if rel_name not in suggestions["relationships"]:
                    suggestions["relationships"].append(rel_name)
                    print(f"  🔗 Detected relationship '{rel_name}' from column '{col_name}'")
        
        # Relationship keywords - more comprehensive
        relationship_keywords = [
            "belongs_to", "works_in", "reports_to", "assigned_to", "manages",
            "located_in", "part_of", "member_of", "owner_of", "has_"
        ]
        for keyword in relationship_keywords:
            if keyword in col_lower:
                rel_name = col_lower.replace("_", "_")  # Keep as is
                if rel_name not in suggestions["relationships"]:
                    suggestions["relationships"].append(rel_name)
                    print(f"  🔗 Detected relationship '{rel_name}' from column '{col_name}'")
        
        # Check for correlation-based relationships
        strong_corrs = analysis.get("correlations", {}).get("strong_correlations", [])
        for corr in strong_corrs:
            if col_name in [corr.get("column1"), corr.get("column2")]:
                corr_strength = corr.get("strength", "")
                if corr_strength == "strong":
                    rel_name = f"strongly_correlates_with"
                    if rel_name not in suggestions["relationships"]:
                        suggestions["relationships"].append(rel_name)
    
    # Suggest properties based on column types
    for col_name, col_stats in analysis.get("columns", {}).items():
        col_type = col_stats.get("type")
        col_lower = col_name.lower()
        
        # Skip ID columns
        if col_lower.endswith("_id") or col_lower.endswith("id"):
            continue
        
        # Map column types to property types
        if col_type == "numeric":
            if "salary" in col_lower or "price" in col_lower or "cost" in col_lower:
                prop_type = "monetary"
            elif "age" in col_lower or "year" in col_lower or "count" in col_lower:
                prop_type = "integer"
            else:
                prop_type = "float"
        elif col_type == "datetime":
            prop_type = "datetime"
        elif col_type == "categorical":
            prop_type = "enum"
        else:
            prop_type = "string"
        
        # Group properties by potential entity (if we can infer it)
        # For now, add to a general properties list
        if "properties" not in suggestions:
            suggestions["properties"] = {}
        
        # Try to infer entity from column name context
        entity = "Entity"  # Default
        for ent in suggestions["entities"]:
            if ent.lower() in col_lower:
                entity = ent
                break
        
        if entity not in suggestions["properties"]:
            suggestions["properties"][entity] = []
        
        suggestions["properties"][entity].append({
            "name": col_name,
            "type": prop_type,
            "description": f"Column: {col_name}, Type: {col_type}"
        })
    
    return suggestions

