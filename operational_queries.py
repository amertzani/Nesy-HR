"""
Operational Query Processor - Pre-computes operational insights
Groups data by various columns and calculates aggregations (avg, min, max, top/bottom N)
Stores results as facts in knowledge base for LLM access and displays in insights page.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import os
import math
from agent_system import document_agents
from strategic_query_agent import normalize_column_name
from strategic_queries import find_csv_file_path, load_csv_data


def sanitize_float(value: Any) -> Optional[float]:
    """
    Convert a value to a JSON-safe float.
    Returns None for NaN, Infinity, or -Infinity values.
    """
    if value is None:
        return None
    try:
        float_val = float(value)
        if math.isnan(float_val) or math.isinf(float_val):
            return None
        return float_val
    except (ValueError, TypeError):
        return None


def compute_operational_insights(csv_file_path: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute operational insights by grouping data and calculating aggregations.
    Returns a dictionary with all computed insights organized by category.
    Focus: Manager-based insights and all departments.
    
    Can use either a pre-loaded DataFrame (preferred, faster) or load from CSV file.
    This ensures no data is missed and can run in parallel with fact extraction.
    
    Args:
        csv_file_path: Optional direct path to CSV file. Used only if df is not provided.
        df: Optional pre-loaded DataFrame. If provided, uses this instead of reading file.
    """
    # Use pre-loaded DataFrame if available (preferred - no file I/O needed)
    if df is not None and len(df) > 0:
        print(f"✅ Using pre-loaded DataFrame for operational insights ({len(df)} rows)")
    else:
        # Load DataFrame directly from CSV file (fallback)
        csv_path = csv_file_path
        if csv_path is None:
            csv_path = find_csv_file_path()
        
        if csv_path is None or not os.path.exists(csv_path):
            print(f"⚠️  No CSV file path available and no DataFrame provided")
            return {}
        
        print(f"📊 Loading DataFrame from CSV file: {csv_path}")
        df = load_csv_data(csv_path)
        
        if df is None or len(df) == 0:
            print(f"⚠️  Failed to load DataFrame from CSV file")
            return {}
    insights = {}
    
    # Manager-based insights (consolidated into one table)
    insights['by_manager'] = group_by_manager(df)
    
    # Department insights (all departments)
    insights['by_department'] = group_by_department(df)
    
    # Recruitment source insights
    insights['by_recruitment_source'] = group_by_recruitment_source(df)
    
    # Employee-level insights (only for active employees)
    insights['top_performance'] = get_top_performance(df, top_n=5)
    insights['bottom_performance'] = get_bottom_performance(df, bottom_n=5)
    insights['top_absences'] = get_top_absences(df, top_n=5)
    insights['bottom_engagement'] = get_bottom_engagement(df, bottom_n=5)
    insights['top_special_projects'] = get_top_special_projects(df, top_n=5)
    insights['top_salary'] = get_top_salary(df, top_n=5)
    
    # Store insights as facts for LLM access (so queries like "average salary in department 3" work)
    store_operational_insights_as_facts(insights)
    
    return insights


def find_manager_column(df: pd.DataFrame) -> Optional[str]:
    """Helper function to find manager column in DataFrame - prefers ManagerName over ManagerID"""
    # First try exact column name matching (case-insensitive) - prefer ManagerName
    manager_name_col = None
    manager_id_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'managername' or col_lower == 'manager_name':
            manager_name_col = col
        elif col_lower == 'managerid' or col_lower == 'manager_id':
            manager_id_col = col
    
    # Prefer ManagerName if both exist
    if manager_name_col:
        return manager_name_col
    elif manager_id_col:
        return manager_id_col
    
    # Try normalize_column_name
    mgr_col = normalize_column_name(df, "ManagerName")
    if mgr_col is None:
        mgr_col = normalize_column_name(df, "ManagerID")
    if mgr_col is None:
        mgr_col = normalize_column_name(df, "Manager")
    
    # If still None, try direct column matching with partial match
    if mgr_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'manager' in col_lower and ('name' in col_lower or 'id' in col_lower or col_lower == 'manager'):
                mgr_col = col
                break
    
    if mgr_col and mgr_col in df.columns:
        return mgr_col
    
    return None


def group_by_department(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by Department and calculate avg PerformanceScoreID, avg Absences, avg Salary"""
    results = []
    
    # Find department column
    dept_col = normalize_column_name(df, "Department")
    if dept_col is None:
        if "Department" in df.columns:
            dept_col = "Department"
        elif "DeptID" in df.columns:
            dept_col = "DeptID"
        else:
            return results
    
    # Find performance score column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df, "PerformanceScore")
    
    # Find absences column
    abs_col = normalize_column_name(df, "Absences")
    
    # Find salary column
    salary_col = normalize_column_name(df, "Salary")
    
    if dept_col not in df.columns:
        return results
    
    # Group by department - ensure we get all unique departments
    dept_groups = df.groupby(dept_col)
    
    for dept, group_df in dept_groups:
        dept_name = str(dept).strip()
        
        # If department is numeric (DeptID), try to find actual department name
        if dept_name.replace('.', '').replace('-', '').isdigit():
            # Try to find a text Department column
            for col in df.columns:
                if col != dept_col and ('department' in col.lower() or 'dept' in col.lower()):
                    dept_names = group_df[col].dropna().unique()
                    if len(dept_names) > 0:
                        text_name = str(dept_names[0]).strip()
                        if not text_name.replace('.', '').replace('-', '').isdigit():
                            dept_name = text_name
                            break
        
        dept_data = {
            "department": dept_name,
            "employee_count": len(group_df)
        }
        
        # Find satisfaction and engagement columns for department
        satisfaction_col = normalize_column_name(df, "Satisfaction")
        if satisfaction_col is None:
            satisfaction_col = normalize_column_name(df, "EngagementSurvey")
        
        engagement_col = normalize_column_name(df, "EngagementSurvey")
        
        # Average PerformanceScoreID - use PerfScoreID (numeric) not PerformanceScore (text)
        if perf_col and perf_col in group_df.columns:
            perf_series = pd.to_numeric(group_df[perf_col], errors='coerce')
            # Filter out NaN values for calculation
            perf_valid = perf_series.dropna()
            if len(perf_valid) > 0:
                dept_data["avg_performance_score"] = sanitize_float(perf_valid.mean())
            else:
                dept_data["avg_performance_score"] = None
        
        # Average Satisfaction
        if satisfaction_col and satisfaction_col in group_df.columns:
            sat_series = pd.to_numeric(group_df[satisfaction_col], errors='coerce')
            sat_valid = sat_series.dropna()
            if len(sat_valid) > 0:
                dept_data["avg_satisfaction"] = sanitize_float(sat_valid.mean())
            else:
                dept_data["avg_satisfaction"] = None
        
        # Average Engagement
        if engagement_col and engagement_col in group_df.columns:
            eng_series = pd.to_numeric(group_df[engagement_col], errors='coerce')
            eng_valid = eng_series.dropna()
            if len(eng_valid) > 0:
                dept_data["avg_engagement"] = sanitize_float(eng_valid.mean())
            else:
                dept_data["avg_engagement"] = None
        
        # Average Absences
        if abs_col and abs_col in group_df.columns:
            abs_series = pd.to_numeric(group_df[abs_col], errors='coerce')
            abs_valid = abs_series.dropna()
            if len(abs_valid) > 0:
                dept_data["avg_absences"] = sanitize_float(abs_valid.mean())
            else:
                dept_data["avg_absences"] = None
        
        # Average Salary - only average over valid numeric values in this group
        if salary_col and salary_col in group_df.columns:
            salary_series = pd.to_numeric(group_df[salary_col], errors='coerce')
            # Filter out NaN values for calculation - only average rows with valid salary in this department
            salary_valid = salary_series.dropna()
            if len(salary_valid) > 0:
                dept_data["avg_salary"] = sanitize_float(salary_valid.mean())
            else:
                dept_data["avg_salary"] = None
        
        results.append(dept_data)
    
    return results


def group_by_manager(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by Manager (ID or name) and calculate comprehensive metrics: avg Performance, Satisfaction, Engagement, Absences, Salary"""
    results = []
    
    # Find manager column
    mgr_col = find_manager_column(df)
    if mgr_col is None:
        return results
    
    # Find performance score column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df, "PerformanceScore")
    
    # Find satisfaction column (might be EngagementSurvey or separate)
    satisfaction_col = normalize_column_name(df, "Satisfaction")
    if satisfaction_col is None:
        satisfaction_col = normalize_column_name(df, "EngagementSurvey")
    
    # Find engagement column
    engagement_col = normalize_column_name(df, "EngagementSurvey")
    
    # Find absences column
    abs_col = normalize_column_name(df, "Absences")
    
    # Find salary column
    salary_col = normalize_column_name(df, "Salary")
    
    if mgr_col not in df.columns:
        return results
    
    # Group by manager
    mgr_groups = df.groupby(mgr_col)
    
    for mgr, group_df in mgr_groups:
        mgr_name = str(mgr).strip()
        mgr_data = {
            "manager": mgr_name,
            "employee_count": len(group_df)
        }
        
        # Average PerformanceScoreID - filter out NaN values
        if perf_col and perf_col in group_df.columns:
            perf_series = pd.to_numeric(group_df[perf_col], errors='coerce')
            perf_valid = perf_series.dropna()
            if len(perf_valid) > 0:
                mgr_data["avg_performance_score"] = sanitize_float(perf_valid.mean())
            else:
                mgr_data["avg_performance_score"] = None
        
        # Average Satisfaction
        if satisfaction_col and satisfaction_col in group_df.columns:
            sat_series = pd.to_numeric(group_df[satisfaction_col], errors='coerce')
            sat_valid = sat_series.dropna()
            if len(sat_valid) > 0:
                mgr_data["avg_satisfaction"] = sanitize_float(sat_valid.mean())
            else:
                mgr_data["avg_satisfaction"] = None
        
        # Average Engagement
        if engagement_col and engagement_col in group_df.columns:
            eng_series = pd.to_numeric(group_df[engagement_col], errors='coerce')
            eng_valid = eng_series.dropna()
            if len(eng_valid) > 0:
                mgr_data["avg_engagement"] = sanitize_float(eng_valid.mean())
            else:
                mgr_data["avg_engagement"] = None
        
        # Average Absences
        if abs_col and abs_col in group_df.columns:
            abs_series = pd.to_numeric(group_df[abs_col], errors='coerce')
            abs_valid = abs_series.dropna()
            if len(abs_valid) > 0:
                mgr_data["avg_absences"] = sanitize_float(abs_valid.mean())
            else:
                mgr_data["avg_absences"] = None
        
        # Average Salary and Total Salary
        if salary_col and salary_col in group_df.columns:
            salary_series = pd.to_numeric(group_df[salary_col], errors='coerce')
            salary_valid = salary_series.dropna()
            if len(salary_valid) > 0:
                mgr_data["avg_salary"] = sanitize_float(salary_valid.mean())
                mgr_data["total_salary"] = sanitize_float(salary_valid.sum())
            else:
                mgr_data["avg_salary"] = None
                mgr_data["total_salary"] = None
        
        results.append(mgr_data)
    
    return results


def get_top_absences(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top N employees with max Absences (only active employees)"""
    results = []
    
    # Filter for active employees only
    status_col = normalize_column_name(df, "EmploymentStatus")
    if status_col and status_col in df.columns:
        active_mask = df[status_col].astype(str).str.lower().str.contains('active', na=False)
        df_active = df[active_mask].copy()
    else:
        df_active = df.copy()
    
    if len(df_active) == 0:
        return results
    
    # Find absences column
    abs_col = normalize_column_name(df, "Absences")
    if abs_col is None or abs_col not in df_active.columns:
        return results
    
    # Find employee name column
    emp_name_col = "Employee_Name" if "Employee_Name" in df_active.columns else None
    if emp_name_col is None:
        for col in df_active.columns:
            if 'name' in col.lower() and 'employee' in col.lower():
                emp_name_col = col
                break
    
    # Convert to numeric
    abs_series = pd.to_numeric(df_active[abs_col], errors='coerce')
    
    # Get top N
    top_absences = abs_series.nlargest(top_n)
    
    # Find additional columns
    dept_col = normalize_column_name(df_active, "Department")
    mgr_col = find_manager_column(df_active)
    salary_col = normalize_column_name(df_active, "Salary")
    position_col = normalize_column_name(df_active, "Position")
    perf_col = normalize_column_name(df_active, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df_active, "PerformanceScore")
    eng_col = normalize_column_name(df_active, "EngagementSurvey")
    sat_col = normalize_column_name(df_active, "EmpSatisfaction")
    dayslate_col = normalize_column_name(df_active, "DaysLateLast30")
    
    for idx, value in top_absences.items():
        emp_data = {
            "absences": sanitize_float(value) if not pd.isna(value) else None,
            "rank": len(results) + 1
        }
        
        if emp_name_col and emp_name_col in df_active.columns:
            emp_data["employee_name"] = str(df_active.loc[idx, emp_name_col])
        else:
            emp_data["employee_name"] = f"Employee {idx}"
        
        # Add position (actual position name, not ID)
        if position_col and position_col in df_active.columns:
            pos_val = df_active.loc[idx, position_col]
            if pd.notna(pos_val):
                emp_data["position"] = str(pos_val).strip()
        
        # Add department
        if dept_col and dept_col in df_active.columns:
            emp_data["department"] = str(df_active.loc[idx, dept_col])
        
        # Add manager
        if mgr_col and mgr_col in df_active.columns:
            emp_data["manager"] = str(df_active.loc[idx, mgr_col])
        
        # Add salary
        if salary_col and salary_col in df_active.columns:
            salary_val = pd.to_numeric(df_active.loc[idx, salary_col], errors='coerce')
            emp_data["salary"] = sanitize_float(salary_val) if not pd.isna(salary_val) else None
        
        # Add performance
        if perf_col and perf_col in df_active.columns:
            perf_val = pd.to_numeric(df_active.loc[idx, perf_col], errors='coerce')
            emp_data["performance_score"] = sanitize_float(perf_val) if not pd.isna(perf_val) else None
        
        # Add engagement
        if eng_col and eng_col in df_active.columns:
            eng_val = pd.to_numeric(df_active.loc[idx, eng_col], errors='coerce')
            emp_data["engagement_score"] = sanitize_float(eng_val) if not pd.isna(eng_val) else None
        
        # Add satisfaction
        if sat_col and sat_col in df_active.columns:
            sat_val = pd.to_numeric(df_active.loc[idx, sat_col], errors='coerce')
            emp_data["satisfaction_score"] = sanitize_float(sat_val) if not pd.isna(sat_val) else None
        
        # Add DaysLateLast30
        if dayslate_col and dayslate_col in df_active.columns:
            dayslate_val = pd.to_numeric(df_active.loc[idx, dayslate_col], errors='coerce')
            emp_data["days_late_last30"] = sanitize_float(dayslate_val) if not pd.isna(dayslate_val) else None
        
        results.append(emp_data)
    
    return results


def get_bottom_engagement(df: pd.DataFrame, bottom_n: int = 5) -> List[Dict[str, Any]]:
    """Get bottom N employees with min EngagementSurvey (only active employees)"""
    results = []
    
    # Filter for active employees only
    status_col = normalize_column_name(df, "EmploymentStatus")
    if status_col and status_col in df.columns:
        active_mask = df[status_col].astype(str).str.lower().str.contains('active', na=False)
        df_active = df[active_mask].copy()
    else:
        df_active = df.copy()
    
    if len(df_active) == 0:
        return results
    
    # Find engagement column
    eng_col = normalize_column_name(df_active, "EngagementSurvey")
    if eng_col is None or eng_col not in df_active.columns:
        return results
    
    # Find employee name column
    emp_name_col = "Employee_Name" if "Employee_Name" in df_active.columns else None
    if emp_name_col is None:
        for col in df_active.columns:
            if 'name' in col.lower() and 'employee' in col.lower():
                emp_name_col = col
                break
    
    # Convert to numeric
    eng_series = pd.to_numeric(df_active[eng_col], errors='coerce')
    
    # Get bottom N
    bottom_engagement = eng_series.nsmallest(bottom_n)
    
    # Find additional columns
    dept_col = normalize_column_name(df_active, "Department")
    mgr_col = find_manager_column(df_active)
    salary_col = normalize_column_name(df_active, "Salary")
    abs_col = normalize_column_name(df_active, "Absences")
    perf_col = normalize_column_name(df_active, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df_active, "PerformanceScore")
    sat_col = normalize_column_name(df_active, "EmpSatisfaction")
    
    for idx, value in bottom_engagement.items():
        emp_data = {
            "engagement_score": sanitize_float(value) if not pd.isna(value) else None,
            "rank": len(results) + 1
        }
        
        if emp_name_col and emp_name_col in df_active.columns:
            emp_data["employee_name"] = str(df_active.loc[idx, emp_name_col])
        else:
            emp_data["employee_name"] = f"Employee {idx}"
        
        # Add department
        if dept_col and dept_col in df_active.columns:
            emp_data["department"] = str(df_active.loc[idx, dept_col])
        
        # Add manager
        if mgr_col and mgr_col in df_active.columns:
            emp_data["manager"] = str(df_active.loc[idx, mgr_col])
        
        # Add salary
        if salary_col and salary_col in df_active.columns:
            salary_val = pd.to_numeric(df_active.loc[idx, salary_col], errors='coerce')
            emp_data["salary"] = sanitize_float(salary_val) if not pd.isna(salary_val) else None
        
        # Add absences
        if abs_col and abs_col in df_active.columns:
            abs_val = pd.to_numeric(df_active.loc[idx, abs_col], errors='coerce')
            emp_data["absences"] = sanitize_float(abs_val) if not pd.isna(abs_val) else None
        
        # Add performance
        if perf_col and perf_col in df_active.columns:
            perf_val = pd.to_numeric(df_active.loc[idx, perf_col], errors='coerce')
            emp_data["performance_score"] = sanitize_float(perf_val) if not pd.isna(perf_val) else None
        
        # Add satisfaction
        if sat_col and sat_col in df_active.columns:
            sat_val = pd.to_numeric(df_active.loc[idx, sat_col], errors='coerce')
            emp_data["satisfaction_score"] = sanitize_float(sat_val) if not pd.isna(sat_val) else None
        
        results.append(emp_data)
    
    return results


def get_top_performance(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top N employees with max PerformanceScore (only active employees)"""
    results = []
    
    # Filter for active employees only
    status_col = normalize_column_name(df, "EmploymentStatus")
    if status_col and status_col in df.columns:
        active_mask = df[status_col].astype(str).str.lower().str.contains('active', na=False)
        df_active = df[active_mask].copy()
    else:
        df_active = df.copy()
    
    if len(df_active) == 0:
        return results
    
    # Find performance column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df_active, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df_active, "PerformanceScore")
    
    if perf_col is None or perf_col not in df_active.columns:
        return results
    
    # Find employee name column
    emp_name_col = "Employee_Name" if "Employee_Name" in df_active.columns else None
    if emp_name_col is None:
        for col in df_active.columns:
            if 'name' in col.lower() and 'employee' in col.lower():
                emp_name_col = col
                break
    
    # Convert to numeric
    perf_series = pd.to_numeric(df_active[perf_col], errors='coerce')
    
    # Get top N
    top_performance = perf_series.nlargest(top_n)
    
    # Find additional columns
    dept_col = normalize_column_name(df_active, "Department")
    mgr_col = find_manager_column(df_active)
    salary_col = normalize_column_name(df_active, "Salary")
    abs_col = normalize_column_name(df_active, "Absences")
    
    for idx, value in top_performance.items():
        emp_data = {
            "performance_score": sanitize_float(value) if not pd.isna(value) else None,
            "rank": len(results) + 1
        }
        
        if emp_name_col and emp_name_col in df_active.columns:
            emp_data["employee_name"] = str(df_active.loc[idx, emp_name_col])
        else:
            emp_data["employee_name"] = f"Employee {idx}"
        
        # Add department
        if dept_col and dept_col in df_active.columns:
            emp_data["department"] = str(df_active.loc[idx, dept_col])
        
        # Add manager
        if mgr_col and mgr_col in df_active.columns:
            emp_data["manager"] = str(df_active.loc[idx, mgr_col])
        
        # Add salary
        if salary_col and salary_col in df_active.columns:
            salary_val = pd.to_numeric(df_active.loc[idx, salary_col], errors='coerce')
            emp_data["salary"] = sanitize_float(salary_val) if not pd.isna(salary_val) else None
        
        # Add absences
        if abs_col and abs_col in df_active.columns:
            abs_val = pd.to_numeric(df_active.loc[idx, abs_col], errors='coerce')
            emp_data["absences"] = sanitize_float(abs_val) if not pd.isna(abs_val) else None
        
        results.append(emp_data)
    
    return results


def get_bottom_performance(df: pd.DataFrame, bottom_n: int = 5) -> List[Dict[str, Any]]:
    """Get bottom N employees with min PerformanceScore (only active employees)"""
    results = []
    
    # Filter for active employees only
    status_col = normalize_column_name(df, "EmploymentStatus")
    if status_col and status_col in df.columns:
        active_mask = df[status_col].astype(str).str.lower().str.contains('active', na=False)
        df_active = df[active_mask].copy()
    else:
        df_active = df.copy()
    
    if len(df_active) == 0:
        return results
    
    # Find performance column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df_active, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df_active, "PerformanceScore")
    
    if perf_col is None or perf_col not in df_active.columns:
        return results
    
    # Find employee name column
    emp_name_col = "Employee_Name" if "Employee_Name" in df_active.columns else None
    if emp_name_col is None:
        for col in df_active.columns:
            if 'name' in col.lower() and 'employee' in col.lower():
                emp_name_col = col
                break
    
    # Convert to numeric
    perf_series = pd.to_numeric(df_active[perf_col], errors='coerce')
    
    # Get bottom N
    bottom_performance = perf_series.nsmallest(bottom_n)
    
    # Find additional columns
    dept_col = normalize_column_name(df_active, "Department")
    mgr_col = find_manager_column(df_active)
    salary_col = normalize_column_name(df_active, "Salary")
    abs_col = normalize_column_name(df_active, "Absences")
    
    for idx, value in bottom_performance.items():
        emp_data = {
            "performance_score": sanitize_float(value) if not pd.isna(value) else None,
            "rank": len(results) + 1
        }
        
        if emp_name_col and emp_name_col in df_active.columns:
            emp_data["employee_name"] = str(df_active.loc[idx, emp_name_col])
        else:
            emp_data["employee_name"] = f"Employee {idx}"
        
        # Add department
        if dept_col and dept_col in df_active.columns:
            emp_data["department"] = str(df_active.loc[idx, dept_col])
        
        # Add manager
        if mgr_col and mgr_col in df_active.columns:
            emp_data["manager"] = str(df_active.loc[idx, mgr_col])
        
        # Add salary
        if salary_col and salary_col in df_active.columns:
            salary_val = pd.to_numeric(df_active.loc[idx, salary_col], errors='coerce')
            emp_data["salary"] = sanitize_float(salary_val) if not pd.isna(salary_val) else None
        
        # Add absences
        if abs_col and abs_col in df_active.columns:
            abs_val = pd.to_numeric(df_active.loc[idx, abs_col], errors='coerce')
            emp_data["absences"] = sanitize_float(abs_val) if not pd.isna(abs_val) else None
        
        results.append(emp_data)
    
    return results


def get_top_special_projects(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top N employees with max SpecialProjectsCount (only active employees)"""
    results = []
    
    # Filter for active employees only
    status_col = normalize_column_name(df, "EmploymentStatus")
    if status_col and status_col in df.columns:
        active_mask = df[status_col].astype(str).str.lower().str.contains('active', na=False)
        df_active = df[active_mask].copy()
    else:
        df_active = df.copy()
    
    if len(df_active) == 0:
        return results
    
    # Find special projects column
    sp_col = normalize_column_name(df_active, "SpecialProjectsCount")
    if sp_col is None or sp_col not in df_active.columns:
        return results
    
    # Find employee name column
    emp_name_col = "Employee_Name" if "Employee_Name" in df_active.columns else None
    if emp_name_col is None:
        for col in df_active.columns:
            if 'name' in col.lower() and 'employee' in col.lower():
                emp_name_col = col
                break
    
    # Convert to numeric
    sp_series = pd.to_numeric(df_active[sp_col], errors='coerce')
    
    # Get top N
    top_special_projects = sp_series.nlargest(top_n)
    
    # Find additional columns
    dept_col = normalize_column_name(df_active, "Department")
    mgr_col = find_manager_column(df_active)
    salary_col = normalize_column_name(df_active, "Salary")
    perf_col = normalize_column_name(df_active, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df_active, "PerformanceScore")
    eng_col = normalize_column_name(df_active, "EngagementSurvey")
    sat_col = normalize_column_name(df_active, "EmpSatisfaction")
    
    for idx, value in top_special_projects.items():
        emp_data = {
            "special_projects_count": sanitize_float(value) if not pd.isna(value) else None,
            "rank": len(results) + 1
        }
        
        if emp_name_col and emp_name_col in df_active.columns:
            emp_data["employee_name"] = str(df_active.loc[idx, emp_name_col])
        else:
            emp_data["employee_name"] = f"Employee {idx}"
        
        # Add department
        if dept_col and dept_col in df_active.columns:
            emp_data["department"] = str(df_active.loc[idx, dept_col])
        
        # Add manager
        if mgr_col and mgr_col in df_active.columns:
            emp_data["manager"] = str(df_active.loc[idx, mgr_col])
        
        # Add salary
        if salary_col and salary_col in df_active.columns:
            salary_val = pd.to_numeric(df_active.loc[idx, salary_col], errors='coerce')
            emp_data["salary"] = sanitize_float(salary_val) if not pd.isna(salary_val) else None
        
        # Add performance
        if perf_col and perf_col in df_active.columns:
            perf_val = pd.to_numeric(df_active.loc[idx, perf_col], errors='coerce')
            emp_data["performance_score"] = sanitize_float(perf_val) if not pd.isna(perf_val) else None
        
        # Add engagement
        if eng_col and eng_col in df_active.columns:
            eng_val = pd.to_numeric(df_active.loc[idx, eng_col], errors='coerce')
            emp_data["engagement_score"] = sanitize_float(eng_val) if not pd.isna(eng_val) else None
        
        # Add satisfaction
        if sat_col and sat_col in df_active.columns:
            sat_val = pd.to_numeric(df_active.loc[idx, sat_col], errors='coerce')
            emp_data["satisfaction_score"] = sanitize_float(sat_val) if not pd.isna(sat_val) else None
        
        results.append(emp_data)
    
    return results


def get_top_salary(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top N employees with max Salary (only active employees)"""
    results = []
    
    # Filter for active employees only
    status_col = normalize_column_name(df, "EmploymentStatus")
    if status_col and status_col in df.columns:
        active_mask = df[status_col].astype(str).str.lower().str.contains('active', na=False)
        df_active = df[active_mask].copy()
    else:
        df_active = df.copy()
    
    if len(df_active) == 0:
        return results
    
    # Find salary column
    salary_col = normalize_column_name(df_active, "Salary")
    if salary_col is None or salary_col not in df_active.columns:
        return results
    
    # Find employee name column
    emp_name_col = "Employee_Name" if "Employee_Name" in df_active.columns else None
    if emp_name_col is None:
        for col in df_active.columns:
            if 'name' in col.lower() and 'employee' in col.lower():
                emp_name_col = col
                break
    
    # Convert to numeric
    salary_series = pd.to_numeric(df_active[salary_col], errors='coerce')
    
    # Get top N
    top_salary = salary_series.nlargest(top_n)
    
    # Find additional columns
    dept_col = normalize_column_name(df_active, "Department")
    mgr_col = find_manager_column(df_active)
    perf_col = normalize_column_name(df_active, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df_active, "PerformanceScore")
    eng_col = normalize_column_name(df_active, "EngagementSurvey")
    sat_col = normalize_column_name(df_active, "EmpSatisfaction")
    
    for idx, value in top_salary.items():
        emp_data = {
            "salary": sanitize_float(value) if not pd.isna(value) else None,
            "rank": len(results) + 1
        }
        
        if emp_name_col and emp_name_col in df_active.columns:
            emp_data["employee_name"] = str(df_active.loc[idx, emp_name_col])
        else:
            emp_data["employee_name"] = f"Employee {idx}"
        
        # Add department
        if dept_col and dept_col in df_active.columns:
            emp_data["department"] = str(df_active.loc[idx, dept_col])
        
        # Add manager
        if mgr_col and mgr_col in df_active.columns:
            emp_data["manager"] = str(df_active.loc[idx, mgr_col])
        
        # Add performance
        if perf_col and perf_col in df_active.columns:
            perf_val = pd.to_numeric(df_active.loc[idx, perf_col], errors='coerce')
            emp_data["performance_score"] = sanitize_float(perf_val) if not pd.isna(perf_val) else None
        
        # Add engagement
        if eng_col and eng_col in df_active.columns:
            eng_val = pd.to_numeric(df_active.loc[idx, eng_col], errors='coerce')
            emp_data["engagement_score"] = sanitize_float(eng_val) if not pd.isna(eng_val) else None
        
        # Add satisfaction
        if sat_col and sat_col in df_active.columns:
            sat_val = pd.to_numeric(df_active.loc[idx, sat_col], errors='coerce')
            emp_data["satisfaction_score"] = sanitize_float(sat_val) if not pd.isna(sat_val) else None
        
        results.append(emp_data)
    
    return results


def group_by_recruitment_source(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by RecruitmentSource and calculate avg PerformanceScoreID, avg Salary, etc."""
    results = []
    
    # Find recruitment source column
    source_col = normalize_column_name(df, "RecruitmentSource")
    if source_col is None:
        if "RecruitmentSource" in df.columns:
            source_col = "RecruitmentSource"
        else:
            return results
    
    # Find performance score column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df, "PerformanceScore")
    
    # Find salary column
    salary_col = normalize_column_name(df, "Salary")
    
    # Find absences column
    abs_col = normalize_column_name(df, "Absences")
    
    # Find employment status column
    status_col = normalize_column_name(df, "EmploymentStatus")
    
    if source_col not in df.columns:
        return results
    
    # Group by recruitment source
    source_groups = df.groupby(source_col)
    
    for source, group_df in source_groups:
        source_name = str(source).strip()
        employee_count = len(group_df)
        source_data = {
            "recruitment_source": source_name,
            "employee_count": employee_count
        }
        
        # Average PerformanceScoreID - filter out NaN values
        if perf_col and perf_col in group_df.columns:
            perf_series = pd.to_numeric(group_df[perf_col], errors='coerce')
            perf_valid = perf_series.dropna()
            if len(perf_valid) > 0:
                source_data["avg_performance_score"] = sanitize_float(perf_valid.mean())
            else:
                source_data["avg_performance_score"] = None
        
        # Average Salary
        if salary_col and salary_col in group_df.columns:
            salary_series = pd.to_numeric(group_df[salary_col], errors='coerce')
            salary_valid = salary_series.dropna()
            if len(salary_valid) > 0:
                source_data["avg_salary"] = sanitize_float(salary_valid.mean())
            else:
                source_data["avg_salary"] = None
        
        # Average Absences
        if abs_col and abs_col in group_df.columns:
            abs_series = pd.to_numeric(group_df[abs_col], errors='coerce')
            abs_valid = abs_series.dropna()
            if len(abs_valid) > 0:
                source_data["avg_absences"] = sanitize_float(abs_valid.mean())
            else:
                source_data["avg_absences"] = None
        
        # Average Satisfaction
        satisfaction_col = normalize_column_name(df, "Satisfaction")
        if satisfaction_col is None:
            satisfaction_col = normalize_column_name(df, "EmpSatisfaction")
        if satisfaction_col and satisfaction_col in group_df.columns:
            sat_series = pd.to_numeric(group_df[satisfaction_col], errors='coerce')
            sat_valid = sat_series.dropna()
            if len(sat_valid) > 0:
                source_data["avg_satisfaction"] = sanitize_float(sat_valid.mean())
            else:
                source_data["avg_satisfaction"] = None
        
        # Average Engagement
        engagement_col = normalize_column_name(df, "EngagementSurvey")
        if engagement_col and engagement_col in group_df.columns:
            eng_series = pd.to_numeric(group_df[engagement_col], errors='coerce')
            eng_valid = eng_series.dropna()
            if len(eng_valid) > 0:
                source_data["avg_engagement"] = sanitize_float(eng_valid.mean())
            else:
                source_data["avg_engagement"] = None
        
        # Total number of active employees
        if status_col and status_col in group_df.columns:
            active_count = len(group_df[group_df[status_col].astype(str).str.lower().str.contains('active', na=False)])
            source_data["active_employees"] = active_count
            # Calculate percentage of active employees
            if employee_count > 0:
                source_data["active_percentage"] = sanitize_float((active_count / employee_count) * 100)
            else:
                source_data["active_percentage"] = None
        else:
            source_data["active_employees"] = None
            source_data["active_percentage"] = None
        
        results.append(source_data)
    
    return results


def compute_additional_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute additional insights for other columns"""
    additional = {}
    
    # Group by Position/Role if available
    pos_col = normalize_column_name(df, "Position")
    if pos_col is None:
        pos_col = normalize_column_name(df, "Role")
    if pos_col and pos_col in df.columns:
        perf_col = normalize_column_name(df, "PerfScoreID")
        if perf_col is None:
            perf_col = normalize_column_name(df, "PerformanceScore")
        
        # Find other columns
        salary_col = normalize_column_name(df, "Salary")
        satisfaction_col = normalize_column_name(df, "Satisfaction")
        if satisfaction_col is None:
            satisfaction_col = normalize_column_name(df, "EngagementSurvey")
        engagement_col = normalize_column_name(df, "EngagementSurvey")
        abs_col = normalize_column_name(df, "Absences")
        status_col = normalize_column_name(df, "EmploymentStatus")
        
        if perf_col and perf_col in df.columns:
            pos_groups = df.groupby(pos_col)
            position_insights = []
            
            for pos, group_df in pos_groups:
                pos_data = {
                    "position": str(pos).strip(),
                    "employee_count": len(group_df)
                }
                
                # Average performance score
                perf_series = pd.to_numeric(group_df[perf_col], errors='coerce')
                perf_valid = perf_series.dropna()
                pos_data["avg_performance_score"] = sanitize_float(perf_valid.mean()) if len(perf_valid) > 0 else None
                
                # Average salary
                if salary_col and salary_col in group_df.columns:
                    salary_series = pd.to_numeric(group_df[salary_col], errors='coerce')
                    salary_valid = salary_series.dropna()
                    if len(salary_valid) > 0:
                        pos_data["avg_salary"] = sanitize_float(salary_valid.mean())
                    else:
                        pos_data["avg_salary"] = None
                
                # Total number with employment status active
                if status_col and status_col in group_df.columns:
                    active_count = len(group_df[group_df[status_col].astype(str).str.lower().str.contains('active', na=False)])
                    pos_data["active_employees"] = active_count
                else:
                    pos_data["active_employees"] = None
                
                # Average satisfaction
                if satisfaction_col and satisfaction_col in group_df.columns:
                    sat_series = pd.to_numeric(group_df[satisfaction_col], errors='coerce')
                    sat_valid = sat_series.dropna()
                    if len(sat_valid) > 0:
                        pos_data["avg_satisfaction"] = sanitize_float(sat_valid.mean())
                    else:
                        pos_data["avg_satisfaction"] = None
                
                # Average engagement
                if engagement_col and engagement_col in group_df.columns:
                    eng_series = pd.to_numeric(group_df[engagement_col], errors='coerce')
                    eng_valid = eng_series.dropna()
                    if len(eng_valid) > 0:
                        pos_data["avg_engagement"] = sanitize_float(eng_valid.mean())
                    else:
                        pos_data["avg_engagement"] = None
                
                # Average absences
                if abs_col and abs_col in group_df.columns:
                    abs_series = pd.to_numeric(group_df[abs_col], errors='coerce')
                    abs_valid = abs_series.dropna()
                    if len(abs_valid) > 0:
                        pos_data["avg_absences"] = sanitize_float(abs_valid.mean())
                    else:
                        pos_data["avg_absences"] = None
                
                position_insights.append(pos_data)
            
            additional["by_position"] = position_insights
    
    # Group by EmploymentStatus if available
    status_col = normalize_column_name(df, "EmploymentStatus")
    if status_col and status_col in df.columns:
        perf_col = normalize_column_name(df, "PerfScoreID")
        if perf_col is None:
            perf_col = normalize_column_name(df, "PerformanceScore")
        
        # Find other columns
        satisfaction_col = normalize_column_name(df, "Satisfaction")
        if satisfaction_col is None:
            satisfaction_col = normalize_column_name(df, "EngagementSurvey")
        engagement_col = normalize_column_name(df, "EngagementSurvey")
        
        if perf_col and perf_col in df.columns:
            status_groups = df.groupby(status_col)
            status_insights = []
            
            for status, group_df in status_groups:
                status_data = {
                    "employment_status": str(status).strip(),
                    "employee_count": len(group_df)
                }
                
                # Average performance score
                perf_series = pd.to_numeric(group_df[perf_col], errors='coerce')
                perf_valid = perf_series.dropna()
                status_data["avg_performance_score"] = sanitize_float(perf_valid.mean()) if len(perf_valid) > 0 else None
                
                # Average engagement survey
                if engagement_col and engagement_col in group_df.columns:
                    eng_series = pd.to_numeric(group_df[engagement_col], errors='coerce')
                    eng_valid = eng_series.dropna()
                    if len(eng_valid) > 0:
                        status_data["avg_engagement"] = sanitize_float(eng_valid.mean())
                    else:
                        status_data["avg_engagement"] = None
                
                # Average satisfaction
                if satisfaction_col and satisfaction_col in group_df.columns:
                    sat_series = pd.to_numeric(group_df[satisfaction_col], errors='coerce')
                    sat_valid = sat_series.dropna()
                    if len(sat_valid) > 0:
                        status_data["avg_satisfaction"] = sanitize_float(sat_valid.mean())
                    else:
                        status_data["avg_satisfaction"] = None
                
                status_insights.append(status_data)
            
            additional["by_employment_status"] = status_insights
    
    return additional


def group_by_manager_performance(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extended manager analysis: Group by Manager and calculate comprehensive performance metrics"""
    results = []
    
    # Find manager column
    mgr_col = find_manager_column(df)
    if mgr_col is None:
        return results
    
    # Find performance score column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df, "PerformanceScore")
    
    if mgr_col not in df.columns:
        return results
    
    # Group by manager
    mgr_groups = df.groupby(mgr_col)
    
    for mgr, group_df in mgr_groups:
        mgr_name = str(mgr).strip()
        mgr_data = {
            "manager": mgr_name,
            "employee_count": len(group_df)
        }
        
        # Performance metrics - filter out NaN values
        if perf_col and perf_col in group_df.columns:
            perf_series = pd.to_numeric(group_df[perf_col], errors='coerce')
            perf_valid = perf_series.dropna()
            if len(perf_valid) > 0:
                mgr_data["avg_performance_score"] = sanitize_float(perf_valid.mean())
                mgr_data["min_performance_score"] = sanitize_float(perf_valid.min())
                mgr_data["max_performance_score"] = sanitize_float(perf_valid.max())
                mgr_data["median_performance_score"] = sanitize_float(perf_valid.median())
                mgr_data["std_performance_score"] = sanitize_float(perf_valid.std())
            else:
                mgr_data["avg_performance_score"] = None
                mgr_data["min_performance_score"] = None
                mgr_data["max_performance_score"] = None
                mgr_data["median_performance_score"] = None
                mgr_data["std_performance_score"] = None
        
        # Department distribution for this manager
        if "Department" in group_df.columns:
            dept_counts = group_df["Department"].value_counts().to_dict()
            mgr_data["departments"] = dept_counts
        
        results.append(mgr_data)
    
    return results


def group_by_manager_absences(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by Manager and calculate absence metrics"""
    results = []
    
    # Find manager column
    mgr_col = find_manager_column(df)
    if mgr_col is None:
        return results
    
    # Find absences column
    abs_col = normalize_column_name(df, "Absences")
    
    # Group by manager
    mgr_groups = df.groupby(mgr_col)
    
    for mgr, group_df in mgr_groups:
        mgr_name = str(mgr).strip()
        mgr_data = {
            "manager": mgr_name,
            "employee_count": len(group_df)
        }
        
        # Absence metrics
        if abs_col and abs_col in group_df.columns:
            abs_series = pd.to_numeric(group_df[abs_col], errors='coerce')
            abs_valid = abs_series.dropna()
            if len(abs_valid) > 0:
                mgr_data["avg_absences"] = sanitize_float(abs_valid.mean())
                mgr_data["min_absences"] = sanitize_float(abs_valid.min())
                mgr_data["max_absences"] = sanitize_float(abs_valid.max())
            else:
                mgr_data["avg_absences"] = None
                mgr_data["min_absences"] = None
                mgr_data["max_absences"] = None
            abs_valid = abs_series.dropna()
            mgr_data["total_absences"] = sanitize_float(abs_valid.sum()) if len(abs_valid) > 0 else None
        
        results.append(mgr_data)
    
    return results


def group_by_manager_salary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by Manager and calculate salary metrics"""
    results = []
    
    # Find manager column
    mgr_col = find_manager_column(df)
    if mgr_col is None:
        return results
    
    # Find salary column
    salary_col = normalize_column_name(df, "Salary")
    
    # Group by manager
    mgr_groups = df.groupby(mgr_col)
    
    for mgr, group_df in mgr_groups:
        mgr_name = str(mgr).strip()
        mgr_data = {
            "manager": mgr_name,
            "employee_count": len(group_df)
        }
        
        # Salary metrics
        if salary_col and salary_col in group_df.columns:
            salary_series = pd.to_numeric(group_df[salary_col], errors='coerce')
            salary_valid = salary_series.dropna()
            if len(salary_valid) > 0:
                mgr_data["avg_salary"] = sanitize_float(salary_valid.mean())
                mgr_data["min_salary"] = sanitize_float(salary_valid.min())
                mgr_data["max_salary"] = sanitize_float(salary_valid.max())
            else:
                mgr_data["avg_salary"] = None
                mgr_data["min_salary"] = None
                mgr_data["max_salary"] = None
            salary_valid = salary_series.dropna()
            mgr_data["total_salary"] = sanitize_float(salary_valid.sum()) if len(salary_valid) > 0 else None
        
        results.append(mgr_data)
    
    return results


def group_by_manager_department(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by Manager and Department combination"""
    results = []
    
    # Find manager column
    mgr_col = find_manager_column(df)
    if mgr_col is None:
        return results
    
    # Find department column
    dept_col = normalize_column_name(df, "Department")
    if dept_col is None:
        if "Department" in df.columns:
            dept_col = "Department"
        elif "DeptID" in df.columns:
            dept_col = "DeptID"
        else:
            return results
    
    # Find performance score column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df, "PerformanceScore")
    
    if mgr_col not in df.columns or dept_col not in df.columns:
        return results
    
    # Group by manager and department
    mgr_dept_groups = df.groupby([mgr_col, dept_col])
    
    for (mgr, dept), group_df in mgr_dept_groups:
        mgr_name = str(mgr).strip()
        dept_name = str(dept).strip()
        
        mgr_dept_data = {
            "manager": mgr_name,
            "department": dept_name,
            "employee_count": len(group_df)
        }
        
        # Performance metrics - filter out NaN values
        if perf_col and perf_col in group_df.columns:
            perf_series = pd.to_numeric(group_df[perf_col], errors='coerce')
            perf_valid = perf_series.dropna()
            mgr_dept_data["avg_performance_score"] = sanitize_float(perf_valid.mean()) if len(perf_valid) > 0 else None
        
        # Absence metrics
        abs_col = normalize_column_name(df, "Absences")
        if abs_col and abs_col in group_df.columns:
            abs_series = pd.to_numeric(group_df[abs_col], errors='coerce')
            abs_valid = abs_series.dropna()
            mgr_dept_data["avg_absences"] = sanitize_float(abs_valid.mean()) if len(abs_valid) > 0 else None
        
        # Engagement metrics
        eng_col = normalize_column_name(df, "EngagementSurvey")
        if eng_col and eng_col in group_df.columns:
            eng_series = pd.to_numeric(group_df[eng_col], errors='coerce')
            eng_valid = eng_series.dropna()
            mgr_dept_data["avg_engagement"] = sanitize_float(eng_valid.mean()) if len(eng_valid) > 0 else None
        
        results.append(mgr_dept_data)
    
    return results


def get_top_managers_by_performance(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """Get top N managers by average team performance"""
    results = []
    
    # Find manager column
    mgr_col = find_manager_column(df)
    if mgr_col is None:
        return results
    
    # Find performance score column - prefer PerfScoreID (numeric) over PerformanceScore (text)
    perf_col = normalize_column_name(df, "PerfScoreID")
    if perf_col is None:
        perf_col = normalize_column_name(df, "PerformanceScore")
    
    if mgr_col not in df.columns or not perf_col or perf_col not in df.columns:
        return results
    
    # Group by manager and calculate average performance - filter out NaN
    mgr_perf = df.groupby(mgr_col)[perf_col].apply(
        lambda x: pd.to_numeric(x, errors='coerce').dropna().mean() if len(pd.to_numeric(x, errors='coerce').dropna()) > 0 else None
    ).dropna().sort_values(ascending=False)
    
    # Get top N
    top_managers = mgr_perf.head(top_n)
    
    for rank, (mgr, avg_perf) in enumerate(top_managers.items(), 1):
        mgr_name = str(mgr).strip()
        mgr_df = df[df[mgr_col] == mgr]
        
        mgr_data = {
            "manager": mgr_name,
            "rank": rank,
            "avg_performance_score": sanitize_float(avg_perf) if not pd.isna(avg_perf) else None,
            "employee_count": len(mgr_df)
        }
        
        # Add department info
        if "Department" in mgr_df.columns:
            dept_counts = mgr_df["Department"].value_counts().to_dict()
            mgr_data["departments"] = dept_counts
        
        results.append(mgr_data)
    
    return results


def get_bottom_managers_by_engagement(df: pd.DataFrame, bottom_n: int = 10) -> List[Dict[str, Any]]:
    """Get bottom N managers by average team engagement"""
    results = []
    
    # Find manager column
    mgr_col = find_manager_column(df)
    if mgr_col is None:
        return results
    
    # Find engagement column
    eng_col = normalize_column_name(df, "EngagementSurvey")
    
    if mgr_col not in df.columns or not eng_col or eng_col not in df.columns:
        return results
    
    # Group by manager and calculate average engagement - filter out NaN
    mgr_eng = df.groupby(mgr_col)[eng_col].apply(
        lambda x: pd.to_numeric(x, errors='coerce').dropna().mean() if len(pd.to_numeric(x, errors='coerce').dropna()) > 0 else None
    ).dropna().sort_values(ascending=True)
    
    # Get bottom N
    bottom_managers = mgr_eng.head(bottom_n)
    
    for rank, (mgr, avg_eng) in enumerate(bottom_managers.items(), 1):
        mgr_name = str(mgr).strip()
        mgr_df = df[df[mgr_col] == mgr]
        
        mgr_data = {
            "manager": mgr_name,
            "rank": rank,
            "avg_engagement": sanitize_float(avg_eng) if not pd.isna(avg_eng) else None,
            "employee_count": len(mgr_df)
        }
        
        # Add department info
        if "Department" in mgr_df.columns:
            dept_counts = mgr_df["Department"].value_counts().to_dict()
            mgr_data["departments"] = dept_counts
        
        results.append(mgr_data)
    
    return results


def store_operational_insights_as_facts(insights: Dict[str, Any]) -> None:
    """
    Store operational insights as facts in the knowledge base.
    This allows the LLM to access these insights when answering questions.
    """
    try:
        from knowledge import add_to_graph
        from datetime import datetime
        
        # Store department insights - make them queryable for questions like "average salary in department 3"
        if 'by_department' in insights:
            for dept_data in insights['by_department']:
                dept_name = str(dept_data.get('department', 'Unknown')).strip()
                # Normalize department name (handle "3.0" -> "3" for better matching)
                dept_name_normalized = dept_name
                try:
                    if dept_name.replace('.', '').replace('-', '').isdigit():
                        dept_num = float(dept_name)
                        if dept_num.is_integer():
                            dept_name_normalized = str(int(dept_num))
                except:
                    pass
                
                # Store multiple fact formats for better queryability
                dept_employee_count = dept_data.get('employee_count', 0)
                
                # Average performance score
                if dept_data.get('avg_performance_score') is not None:
                    avg_perf = dept_data['avg_performance_score']
                    # Store in multiple formats for better queryability
                    fact_texts = [
                        f"Department {dept_name} has average performance score of {avg_perf:.2f}",
                        f"Department {dept_name_normalized} has average performance score of {avg_perf:.2f}",
                        f"The average performance score in department {dept_name} is {avg_perf:.2f}",
                        f"The average performance score in department {dept_name_normalized} is {avg_perf:.2f}",
                    ]
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
                
                # Average absences
                if dept_data.get('avg_absences') is not None:
                    avg_abs = dept_data['avg_absences']
                    fact_texts = [
                        f"Department {dept_name} has average absences of {avg_abs:.2f} days",
                        f"Department {dept_name_normalized} has average absences of {avg_abs:.2f} days",
                        f"The average absences in department {dept_name} is {avg_abs:.2f} days",
                        f"The average absences in department {dept_name_normalized} is {avg_abs:.2f} days",
                    ]
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
                
                # Average salary - CRITICAL for queries like "average salary in department 3"
                if dept_data.get('avg_salary') is not None:
                    avg_sal = dept_data['avg_salary']
                    fact_texts = [
                        f"Department {dept_name} has average salary of {avg_sal:.2f}",
                        f"Department {dept_name_normalized} has average salary of {avg_sal:.2f}",
                        f"The average salary in department {dept_name} is {avg_sal:.2f}",
                        f"The average salary in department {dept_name_normalized} is {avg_sal:.2f}",
                        f"Average salary for department {dept_name} is {avg_sal:.2f}",
                        f"Average salary for department {dept_name_normalized} is {avg_sal:.2f}",
                    ]
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
                
                # Store employee count
                if dept_employee_count > 0:
                    fact_text = f"Department {dept_name} has {dept_employee_count} employees"
                    add_to_graph(
                        fact_text,
                        source_document="operational_insights",
                        uploaded_at=datetime.now().isoformat(),
                        agent_id="operational_query_agent"
                    )
        
        # Store manager insights
        if 'by_manager' in insights:
            for mgr_data in insights['by_manager']:
                mgr_name = str(mgr_data.get('manager', 'Unknown')).strip()
                mgr_team_size = mgr_data.get('employee_count', 0)
                
                # Average performance score
                if mgr_data.get('avg_performance_score') is not None:
                    avg_perf = mgr_data['avg_performance_score']
                    fact_texts = [
                        f"Manager {mgr_name} has average team performance score of {avg_perf:.2f}",
                        f"Manager {mgr_name} manages a team with average performance score of {avg_perf:.2f}",
                        f"The average performance score for manager {mgr_name}'s team is {avg_perf:.2f}",
                    ]
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
                
                # Average engagement
                if mgr_data.get('avg_engagement') is not None:
                    avg_eng = mgr_data['avg_engagement']
                    fact_texts = [
                        f"Manager {mgr_name} has average team engagement score of {avg_eng:.2f}",
                        f"Manager {mgr_name} manages a team with average engagement score of {avg_eng:.2f}",
                        f"The average engagement score for manager {mgr_name}'s team is {avg_eng:.2f}",
                    ]
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
                
                # Average satisfaction
                if mgr_data.get('avg_satisfaction') is not None:
                    avg_sat = mgr_data['avg_satisfaction']
                    fact_text = f"Manager {mgr_name} has average team satisfaction score of {avg_sat:.2f}"
                    add_to_graph(
                        fact_text,
                        source_document="operational_insights",
                        uploaded_at=datetime.now().isoformat(),
                        agent_id="operational_query_agent"
                    )
                
                # Team size
                if mgr_team_size > 0:
                    fact_text = f"Manager {mgr_name} manages {mgr_team_size} employees"
                    add_to_graph(
                        fact_text,
                        source_document="operational_insights",
                        uploaded_at=datetime.now().isoformat(),
                        agent_id="operational_query_agent"
                    )
        
        # Store top absences - multiple formats for better queryability
        if 'top_absences' in insights:
            for i, emp_data in enumerate(insights['top_absences'][:5], 1):
                emp_name = emp_data.get('employee_name', 'Unknown')
                absences = emp_data.get('absences')
                dept = emp_data.get('department', '')
                position = emp_data.get('position', '')
                
                if absences is not None:
                    # Store in multiple formats for better queryability
                    fact_texts = [
                        f"Employee {emp_name} has {absences:.0f} absences (rank {i} highest)",
                        f"{emp_name} has {absences:.0f} absences and is ranked {i} in top employees by absences",
                        f"Top {i} employee by absences is {emp_name} with {absences:.0f} absences",
                        f"The employee with rank {i} highest absences is {emp_name} with {absences:.0f} absences",
                    ]
                    if dept:
                        fact_texts.append(f"{emp_name} from {dept} department has {absences:.0f} absences (rank {i})")
                    if position:
                        fact_texts.append(f"{emp_name} ({position}) has {absences:.0f} absences (rank {i})")
                    
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
            
            # Store summary fact for "top 5 employees by absences" queries
            if len(insights['top_absences']) >= 5:
                top5_list = []
                for i, emp_data in enumerate(insights['top_absences'][:5], 1):
                    emp_name = emp_data.get('employee_name', 'Unknown')
                    absences = emp_data.get('absences', 0)
                    top5_list.append(f"{i}. {emp_name} ({absences:.0f} absences)")
                
                summary_fact = f"Top 5 employees by absences: {', '.join(top5_list)}"
                add_to_graph(
                    summary_fact,
                    source_document="operational_insights",
                    uploaded_at=datetime.now().isoformat(),
                    agent_id="operational_query_agent"
                )
        
        # Store bottom engagement - multiple formats for better queryability
        if 'bottom_engagement' in insights:
            for i, emp_data in enumerate(insights['bottom_engagement'][:5], 1):
                emp_name = emp_data.get('employee_name', 'Unknown')
                engagement = emp_data.get('engagement_score')
                dept = emp_data.get('department', '')
                manager = emp_data.get('manager', '')
                
                if engagement is not None:
                    fact_texts = [
                        f"Employee {emp_name} has engagement score of {engagement:.2f} (rank {i} lowest)",
                        f"{emp_name} has engagement score of {engagement:.2f} and is ranked {i} in bottom employees by engagement",
                        f"Bottom {i} employee by engagement is {emp_name} with engagement score of {engagement:.2f}",
                        f"The employee with rank {i} lowest engagement is {emp_name} with engagement score of {engagement:.2f}",
                    ]
                    if dept:
                        fact_texts.append(f"{emp_name} from {dept} department has engagement score of {engagement:.2f} (rank {i} lowest)")
                    if manager:
                        fact_texts.append(f"{emp_name} (managed by {manager}) has engagement score of {engagement:.2f} (rank {i} lowest)")
                    
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
            
            # Store summary fact for "bottom 5 employees by engagement" queries
            if len(insights['bottom_engagement']) >= 5:
                bottom5_list = []
                for i, emp_data in enumerate(insights['bottom_engagement'][:5], 1):
                    emp_name = emp_data.get('employee_name', 'Unknown')
                    engagement = emp_data.get('engagement_score', 0)
                    bottom5_list.append(f"{i}. {emp_name} ({engagement:.2f})")
                
                summary_fact = f"Bottom 5 employees by engagement: {', '.join(bottom5_list)}"
                add_to_graph(
                    summary_fact,
                    source_document="operational_insights",
                    uploaded_at=datetime.now().isoformat(),
                    agent_id="operational_query_agent"
                )
        
        # Store recruitment source insights
        if 'by_recruitment_source' in insights:
            for source_data in insights['by_recruitment_source']:
                source_name = source_data.get('recruitment_source', 'Unknown')
                employee_count = source_data.get('employee_count', 0)
                
                if source_data.get('avg_performance_score') is not None:
                    avg_perf = source_data['avg_performance_score']
                    fact_texts = [
                        f"Recruitment source {source_name} has average performance score of {avg_perf:.2f}",
                        f"Recruitment source {source_name} recruited {employee_count} employees with average performance score of {avg_perf:.2f}",
                        f"The average performance score for recruitment source {source_name} is {avg_perf:.2f}",
                    ]
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
                
                if source_data.get('avg_salary') is not None:
                    avg_sal = source_data['avg_salary']
                    fact_texts = [
                        f"Recruitment source {source_name} has average salary of {avg_sal:.2f}",
                        f"Employees from recruitment source {source_name} have average salary of {avg_sal:.2f}",
                    ]
                    for fact_text in fact_texts:
                        add_to_graph(
                            fact_text,
                            source_document="operational_insights",
                            uploaded_at=datetime.now().isoformat(),
                            agent_id="operational_query_agent"
                        )
                
                if source_data.get('avg_absences') is not None:
                    avg_abs = source_data['avg_absences']
                    fact_text = f"Recruitment source {source_name} has average absences of {avg_abs:.2f} days"
                    add_to_graph(
                        fact_text,
                        source_document="operational_insights",
                        uploaded_at=datetime.now().isoformat(),
                        agent_id="operational_query_agent"
                    )
        
    except Exception as e:
        print(f"⚠️  Error storing operational insights: {e}")


def process_operational_query(query_info: Dict[str, Any], question: str) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process operational query by retrieving facts from knowledge graph and answering the question.
    First checks for existing insights, then ensures they're stored as facts.
    """
    # First, try to get existing insights from document agents or documents_store
    insights = None
    
    try:
        from agent_system import document_agents
        # Check document agents for cached insights
        for agent_id, agent in document_agents.items():
            if hasattr(agent, 'document_type') and agent.document_type.lower() == '.csv':
                metadata = getattr(agent, 'metadata', {})
                if 'operational_insights' in metadata:
                    cached_insights = metadata.get('operational_insights')
                    if cached_insights and isinstance(cached_insights, dict) and len(cached_insights) > 0:
                        insights = cached_insights
                        break
    except Exception:
        pass
    
    # If no cached insights, try documents_store
    if not insights:
        try:
            from documents_store import get_all_documents
            documents = get_all_documents()
            csv_docs = [d for d in documents if d.get('type', '').lower() == 'csv']
            if csv_docs:
                latest_csv = csv_docs[-1]
                if 'operational_insights' in latest_csv:
                    cached_insights = latest_csv.get('operational_insights')
                    if cached_insights and isinstance(cached_insights, dict) and len(cached_insights) > 0:
                        insights = cached_insights
        except Exception:
            pass
    
    # If still no insights, try to compute from CSV file
    if not insights:
        try:
            insights = compute_operational_insights()
        except Exception as compute_error:
            print(f"❌ Error computing insights: {compute_error}")
            return (
                f"I couldn't compute the required data. Error: {str(compute_error)}. Please ensure a CSV file has been uploaded and processed.",
                [],
                {"strategy": "operational_query_agent", "reason": f"Computation error: {str(compute_error)}"}
            )
    
    if not insights or (isinstance(insights, dict) and len(insights) == 0):
        return (
            "I couldn't find the required data. Please ensure a CSV file has been uploaded and processed. If you just uploaded a file, try refreshing the page or waiting a moment for processing to complete.",
            [],
            {"strategy": "operational_query_agent", "reason": "No data available"}
        )
    
    # Store insights as facts for LLM access (if not already stored)
    # This is idempotent - won't duplicate facts if already stored
    store_operational_insights_as_facts(insights)
    
    # Retrieve relevant facts from knowledge graph, prioritizing operational_insights
    try:
        from knowledge import retrieve_context
        
        # Retrieve context with focus on operational insights
        # The retrieve_context function will boost facts from "operational_insights" source
        context = retrieve_context(question, limit=50)
        
        # If we have context, use it to answer the question
        if context and "No directly relevant facts found" not in context:
            # For operational queries, we want to use the facts directly
            # The LLM in responses.py will use this context to answer
            
            # Extract evidence facts from the context
            evidence_facts = []
            
            # Also check if we can directly answer from insights data
            question_lower = question.lower()
            
            # Direct answer for "top 5 employees by absences"
            if "top" in question_lower and ("absence" in question_lower or "absent" in question_lower):
                if 'top_absences' in insights and insights['top_absences']:
                    answer_parts = ["Based on available data, here's the list of top 5 employees with highest absences:\n"]
                    for i, emp_data in enumerate(insights['top_absences'][:5], 1):
                        emp_name = emp_data.get('employee_name', 'Unknown')
                        absences = emp_data.get('absences', 0)
                        dept = emp_data.get('department', '')
                        answer_parts.append(f"{i}. **{emp_name}**: {absences:.0f} absences" + (f" ({dept})" if dept else ""))
                    answer = "\n".join(answer_parts)
                    answer += "\n\n**Explanation:**\nThese are the employees with the highest number of absences based on the operational insights computed from the uploaded data."
                    
                    # Create evidence facts
                    for emp_data in insights['top_absences'][:5]:
                        evidence_facts.append({
                            "subject": emp_data.get('employee_name', 'Unknown'),
                            "predicate": "has_absences",
                            "object": str(emp_data.get('absences', 0)),
                            "source": ["operational_insights"]
                        })
                    
                    return answer, evidence_facts, {
                        "strategy": "operational_query_agent",
                        "reason": "Retrieved from operational insights",
                        "insights": insights
                    }
            
            # Direct answer for "best recruitment source" or "recruitment source" queries
            if "recruitment" in question_lower and ("best" in question_lower or "most" in question_lower or "highest" in question_lower or "number" in question_lower or "hires" in question_lower):
                if 'by_recruitment_source' in insights and insights['by_recruitment_source']:
                    # Find the recruitment source with highest employee_count
                    sources = insights['by_recruitment_source']
                    if sources:
                        # Sort by employee_count (hires)
                        sorted_sources = sorted(sources, key=lambda x: x.get('employee_count', 0), reverse=True)
                        top_source = sorted_sources[0]
                        source_name = top_source.get('recruitment_source', 'Unknown')
                        hire_count = top_source.get('employee_count', 0)
                        
                        answer_parts = [f"Based on the operational insights data, **{source_name}** has been the most common recruitment source with {hire_count} hires."]
                        if len(sorted_sources) > 1:
                            answer_parts.append(f"\n\nOther recruitment sources:")
                            for i, src in enumerate(sorted_sources[1:6], 2):  # Show top 5
                                answer_parts.append(f"{i}. {src.get('recruitment_source', 'Unknown')}: {src.get('employee_count', 0)} hires")
                        
                        answer = "\n".join(answer_parts)
                        answer += "\n\n**Explanation:**\nThis is based on the total number of employees hired from each recruitment source in the dataset."
                        
                        # Create evidence facts
                        for src in sorted_sources[:5]:
                            evidence_facts.append({
                                "subject": src.get('recruitment_source', 'Unknown'),
                                "predicate": "has_hires",
                                "object": str(src.get('employee_count', 0)),
                                "source": ["operational_insights"]
                            })
                        
                        return answer, evidence_facts, {
                            "strategy": "operational_query_agent",
                            "reason": "Retrieved from operational insights",
                            "insights": insights
                        }
            
            # Direct answer for "bottom 5 employees by engagement"
            if "bottom" in question_lower and "engagement" in question_lower:
                if 'bottom_engagement' in insights and insights['bottom_engagement']:
                    answer_parts = ["Based on available data, here's the list of bottom 5 employees with lowest engagement:\n"]
                    for i, emp_data in enumerate(insights['bottom_engagement'][:5], 1):
                        emp_name = emp_data.get('employee_name', 'Unknown')
                        engagement = emp_data.get('engagement_score', 0)
                        dept = emp_data.get('department', '')
                        answer_parts.append(f"{i}. **{emp_name}**: {engagement:.2f} engagement score" + (f" ({dept})" if dept else ""))
                    answer = "\n".join(answer_parts)
                    answer += "\n\n**Explanation:**\nThese are the employees with the lowest engagement scores based on the operational insights computed from the uploaded data."
                    
                    # Create evidence facts
                    for emp_data in insights['bottom_engagement'][:5]:
                        evidence_facts.append({
                            "subject": emp_data.get('employee_name', 'Unknown'),
                            "predicate": "has_engagement_score",
                            "object": str(emp_data.get('engagement_score', 0)),
                            "source": ["operational_insights"]
                        })
                    
                    return answer, evidence_facts, {
                        "strategy": "operational_query_agent",
                        "reason": "Retrieved from operational insights",
                        "insights": insights
                    }
            
            # For other queries, return context to be used by LLM
            # The context contains relevant facts from operational_insights
            answer = f"Based on operational insights from the uploaded data:\n\n{context}"
            
            # Create evidence facts from context
            evidence_facts = [{
                "subject": "operational_insights",
                "predicate": "contains_data",
                "object": question,
                "source": ["operational_insights"]
            }]
            
            return answer, evidence_facts, {
                "strategy": "operational_query_agent",
                "reason": "Retrieved from operational insights knowledge base",
                "insights": insights
            }
        else:
            # No relevant facts found, but we have insights - try to answer from insights directly
            return (
                "I couldn't find specific information to answer your question in the operational insights. Please ensure the data has been processed and try rephrasing your question.",
                [],
                {"strategy": "operational_query_agent", "reason": "No matching facts found", "insights": insights}
            )
    
    except Exception as e:
        print(f"⚠️  Error retrieving context for operational query: {e}")
        
        # Fallback: return generic message
        return (
            f"I encountered an error processing this operational query: {str(e)}",
            [],
            {"strategy": "operational_query_agent", "reason": f"Error: {str(e)}", "insights": insights}
        )
