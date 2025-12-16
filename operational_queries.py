"""
Operational Queries Module
==========================

Provides functions for loading CSV data and computing operational insights.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any, List


def find_csv_file_path() -> Optional[str]:
    """Find the CSV file path."""
    paths = [
        "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv",
        "/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv",
        os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HRDataset_v14.csv"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HR_S.csv"),
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def normalize_column_name(df: pd.DataFrame, column_name: str) -> Optional[str]:
    """
    Find column name with case-insensitive and whitespace-tolerant matching.
    
    Args:
        df: DataFrame
        column_name: Desired column name
    
    Returns:
        Actual column name if found, None otherwise
    """
    if df is None or df.empty:
        return None
    
    column_name_lower = column_name.lower().strip()
    
    # Exact match
    if column_name in df.columns:
        return column_name
    
    # Case-insensitive match
    for col in df.columns:
        if col.lower().strip() == column_name_lower:
            return col
    
    # Partial match (handles spaces, underscores)
    column_name_normalized = column_name_lower.replace('_', '').replace(' ', '')
    for col in df.columns:
        col_normalized = col.lower().strip().replace('_', '').replace(' ', '')
        if col_normalized == column_name_normalized:
            return col
    
    return None


def load_csv_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV data from file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame or None if error
    """
    if not csv_path or not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def compute_operational_insights(df: Optional[pd.DataFrame] = None, csv_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Compute operational insights from CSV data.
    
    Args:
        df: Optional DataFrame (if provided, csv_path is ignored)
        csv_path: Optional path to CSV file
    
    Returns:
        Dictionary with operational insights
    """
    if df is None:
        if csv_path:
            df = load_csv_data(csv_path)
        else:
            csv_path = find_csv_file_path()
            if csv_path:
                df = load_csv_data(csv_path)
    
    if df is None or df.empty:
        return None
    
    insights = {
        "by_department": {},
        "by_manager": {},
        "by_recruitment": {}
    }
    
    # Get column names
    dept_col = normalize_column_name(df, "Department")
    salary_col = normalize_column_name(df, "Salary")
    perf_col = normalize_column_name(df, "PerformanceScore")
    manager_col = normalize_column_name(df, "ManagerName")
    recruitment_col = normalize_column_name(df, "RecruitmentSource")
    
    # Compute by department
    if dept_col:
        if salary_col:
            dept_salary = df.groupby(dept_col)[salary_col].mean().to_dict()
            insights["by_department"]["salary"] = dept_salary
        
        if perf_col:
            # Convert performance to numeric
            perf_map = {'Exceeds': 4, 'Fully Meets': 3, 'Needs Improvement': 2, 'PIP': 1}
            if df[perf_col].dtype == 'object':
                df['_PerfNumeric'] = df[perf_col].map(perf_map)
            else:
                df['_PerfNumeric'] = df[perf_col]
            dept_perf = df.groupby(dept_col)['_PerfNumeric'].mean().to_dict()
            insights["by_department"]["performance"] = dept_perf
    
    # Compute by manager
    if manager_col:
        eng_col = normalize_column_name(df, "EngagementSurvey")
        if eng_col:
            manager_eng = df.groupby(manager_col)[eng_col].mean().to_dict()
            insights["by_manager"]["engagement"] = manager_eng
    
    # Compute by recruitment
    if recruitment_col and perf_col:
        if df[perf_col].dtype == 'object':
            df['_PerfNumeric'] = df[perf_col].map(perf_map)
        else:
            df['_PerfNumeric'] = df[perf_col]
        recruitment_perf = df.groupby(recruitment_col)['_PerfNumeric'].mean().to_dict()
        insights["by_recruitment"]["performance"] = recruitment_perf
    
    return insights


def get_top_salary(df: pd.DataFrame, top_n: int = 1) -> List[Dict[str, Any]]:
    """
    Get top N employees by salary.
    
    Args:
        df: DataFrame
        top_n: Number of top employees to return
    
    Returns:
        List of employee dictionaries
    """
    salary_col = normalize_column_name(df, "Salary")
    name_col = normalize_column_name(df, "Employee_Name")
    
    if not salary_col or not name_col:
        return []
    
    top_employees = df.nlargest(top_n, salary_col)
    
    result = []
    for _, row in top_employees.iterrows():
        result.append({
            "name": row[name_col],
            "salary": row[salary_col]
        })
    
    return result


def get_bottom_performance(df: pd.DataFrame, bottom_n: int = 1) -> List[Dict[str, Any]]:
    """
    Get bottom N employees by performance.
    
    Args:
        df: DataFrame
        bottom_n: Number of bottom employees to return
    
    Returns:
        List of employee dictionaries
    """
    perf_col = normalize_column_name(df, "PerformanceScore")
    name_col = normalize_column_name(df, "Employee_Name")
    
    if not perf_col or not name_col:
        return []
    
    # Convert performance to numeric
    perf_map = {'Exceeds': 4, 'Fully Meets': 3, 'Needs Improvement': 2, 'PIP': 1}
    if df[perf_col].dtype == 'object':
        df['_PerfNumeric'] = df[perf_col].map(perf_map)
    else:
        df['_PerfNumeric'] = df[perf_col]
    
    bottom_employees = df.nsmallest(bottom_n, '_PerfNumeric')
    
    result = []
    for _, row in bottom_employees.iterrows():
        result.append({
            "name": row[name_col],
            "performance": row[perf_col],
            "performance_numeric": row['_PerfNumeric']
        })
    
    return result

