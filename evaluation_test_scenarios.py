"""
Evaluation Test Scenarios
==========================

This script creates and tests evaluation scenarios based on the actual HR dataset.
It tests the system's ability to answer operational (k=2) and strategic (k≥3) queries.

Author: Research Brain Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
from evaluation_metrics import extract_dataset_metrics, calculate_combinatorial_metrics


def load_hr_dataset(csv_path: str) -> pd.DataFrame:
    """Load the HR dataset from CSV file."""
    try:
        # Try to detect separator
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            
            if semicolon_count > comma_count and semicolon_count > 0:
                sep = ';'
            else:
                sep = ','
        
        df = pd.read_csv(csv_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
        
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
        
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")


def create_operational_scenarios(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create operational scenarios (k=2) based on the dataset.
    These are 2-variable combinations for day-to-day monitoring.
    """
    scenarios = []
    
    # Scenario O1: PerformanceScore × Department
    if 'PerformanceScore' in df.columns and 'Department' in df.columns:
        scenarios.append({
            "id": "O1",
            "name": "Performance Score by Department",
            "variables": ["PerformanceScore", "Department"],
            "k": 2,
            "type": "operational",
            "queries": [
                "What is the distribution of performance scores by department?",
                "How do performance scores vary across departments?",
                "Which department has the highest average performance score?",
                "Show me performance metrics by department"
            ],
            "ground_truth": compute_ground_truth(df, ["PerformanceScore", "Department"])
        })
    
    # Scenario O2: Absences × EmploymentStatus
    if 'Absences' in df.columns and 'EmploymentStatus' in df.columns:
        scenarios.append({
            "id": "O2",
            "name": "Absences by Employment Status",
            "variables": ["Absences", "EmploymentStatus"],
            "k": 2,
            "type": "operational",
            "queries": [
                "How do absences differ between active and terminated employees?",
                "What are absence patterns by employment status?",
                "Compare absences for active vs terminated employees"
            ],
            "ground_truth": compute_ground_truth(df, ["Absences", "EmploymentStatus"])
        })
    
    # Scenario O3: EngagementSurvey × ManagerName
    if 'EngagementSurvey' in df.columns and 'ManagerName' in df.columns:
        scenarios.append({
            "id": "O3",
            "name": "Engagement by Manager",
            "variables": ["EngagementSurvey", "ManagerName"],
            "k": 2,
            "type": "operational",
            "queries": [
                "What is the team-level engagement by manager?",
                "How does engagement vary by manager?",
                "Which manager has the highest team engagement?"
            ],
            "ground_truth": compute_ground_truth(df, ["EngagementSurvey", "ManagerName"])
        })
    
    # Scenario O4: Salary × Department
    if 'Salary' in df.columns and 'Department' in df.columns:
        scenarios.append({
            "id": "O4",
            "name": "Salary by Department",
            "variables": ["Salary", "Department"],
            "k": 2,
            "type": "operational",
            "queries": [
                "What is the average salary by department?",
                "How does salary distribution vary across departments?",
                "Which department has the highest average salary?"
            ],
            "ground_truth": compute_ground_truth(df, ["Salary", "Department"])
        })
    
    # Scenario O5: PerformanceScore × RecruitmentSource
    if 'PerformanceScore' in df.columns and 'RecruitmentSource' in df.columns:
        scenarios.append({
            "id": "O5",
            "name": "Performance by Recruitment Source",
            "variables": ["PerformanceScore", "RecruitmentSource"],
            "k": 2,
            "type": "operational",
            "queries": [
                "How does performance vary by recruitment source?",
                "Which recruitment sources yield the best performers?",
                "What is the performance distribution by recruitment channel?"
            ],
            "ground_truth": compute_ground_truth(df, ["PerformanceScore", "RecruitmentSource"])
        })
    
    return scenarios


def create_strategic_scenarios(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create strategic scenarios (k≥3) based on the dataset.
    These are multi-variable combinations for long-term analysis.
    """
    scenarios = []
    
    # Scenario S1: PerformanceScore × EngagementSurvey × EmploymentStatus
    if all(col in df.columns for col in ["PerformanceScore", "EngagementSurvey", "EmploymentStatus"]):
        scenarios.append({
            "id": "S1",
            "name": "Performance-Engagement-Status Risk Clusters",
            "variables": ["PerformanceScore", "EngagementSurvey", "EmploymentStatus"],
            "k": 3,
            "type": "strategic",
            "queries": [
                "Identify employees with high performance but low engagement who are at risk of termination",
                "Which active employees have declining engagement and performance?",
                "Find risk clusters combining performance, engagement, and employment status"
            ],
            "ground_truth": compute_ground_truth(df, ["PerformanceScore", "EngagementSurvey", "EmploymentStatus"])
        })
    
    # Scenario S2: RecruitmentSource × PerformanceScore × EmploymentStatus
    if all(col in df.columns for col in ["RecruitmentSource", "PerformanceScore", "EmploymentStatus"]):
        scenarios.append({
            "id": "S2",
            "name": "Recruitment Channel Quality",
            "variables": ["RecruitmentSource", "PerformanceScore", "EmploymentStatus"],
            "k": 3,
            "type": "strategic",
            "queries": [
                "Which recruitment sources deliver high-performing employees who remain active?",
                "Rank recruitment channels by performance and retention",
                "Identify underperforming recruitment sources with high turnover"
            ],
            "ground_truth": compute_ground_truth(df, ["RecruitmentSource", "PerformanceScore", "EmploymentStatus"])
        })
    
    # Scenario S3: Department × Salary × PerformanceScore
    if all(col in df.columns for col in ["Department", "Salary", "PerformanceScore"]):
        scenarios.append({
            "id": "S3",
            "name": "Department Compensation-Performance Analysis",
            "variables": ["Department", "Salary", "PerformanceScore"],
            "k": 3,
            "type": "strategic",
            "queries": [
                "Which departments have high salaries but low performance?",
                "Analyze the relationship between salary, performance, and department",
                "Identify departments with compensation-performance misalignment"
            ],
            "ground_truth": compute_ground_truth(df, ["Department", "Salary", "PerformanceScore"])
        })
    
    return scenarios


def compute_ground_truth(df: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
    """
    Compute ground truth statistics for a set of variables.
    This serves as the reference for evaluating system responses.
    """
    ground_truth = {
        "variables": variables,
        "statistics": {}
    }
    
    # Filter out missing values
    df_clean = df[variables].dropna()
    
    if len(df_clean) == 0:
        return ground_truth
    
    # For 2-variable scenarios: compute groupby statistics
    if len(variables) == 2:
        var1, var2 = variables
        
        # If var1 is numeric and var2 is categorical
        if pd.api.types.is_numeric_dtype(df[var1]) and not pd.api.types.is_numeric_dtype(df[var2]):
            grouped = df_clean.groupby(var2)[var1].agg(['mean', 'median', 'std', 'count', 'min', 'max']).to_dict('index')
            ground_truth["statistics"] = {
                "groupby": var2,
                "metric": var1,
                "groups": grouped
            }
        
        # If both are categorical: cross-tabulation
        elif not pd.api.types.is_numeric_dtype(df[var1]) and not pd.api.types.is_numeric_dtype(df[var2]):
            crosstab = pd.crosstab(df_clean[var1], df_clean[var2], margins=True).to_dict()
            ground_truth["statistics"] = {
                "crosstab": crosstab
            }
        
        # If both are numeric: correlation
        elif pd.api.types.is_numeric_dtype(df[var1]) and pd.api.types.is_numeric_dtype(df[var2]):
            correlation = df_clean[var1].corr(df_clean[var2])
            ground_truth["statistics"] = {
                "correlation": float(correlation) if not np.isnan(correlation) else None
            }
    
    # For 3-variable scenarios: multi-dimensional analysis
    elif len(variables) == 3:
        var1, var2, var3 = variables
        
        # Try to identify which is the metric and which are grouping variables
        numeric_vars = [v for v in variables if pd.api.types.is_numeric_dtype(df[v])]
        categorical_vars = [v for v in variables if not pd.api.types.is_numeric_dtype(df[v])]
        
        if len(numeric_vars) == 1 and len(categorical_vars) == 2:
            metric = numeric_vars[0]
            group1, group2 = categorical_vars
            
            grouped = df_clean.groupby([group1, group2])[metric].agg(['mean', 'count']).to_dict('index')
            # Convert tuple keys to strings for JSON serialization
            grouped_str = {str(k): v for k, v in grouped.items()}
            ground_truth["statistics"] = {
                "groupby": [group1, group2],
                "metric": metric,
                "groups": grouped_str
            }
    
    return ground_truth


def generate_test_report(scenarios: List[Dict[str, Any]], output_file: str = "evaluation_test_report.txt"):
    """Generate a comprehensive test report with all scenarios."""
    report = []
    report.append("=" * 80)
    report.append("EVALUATION TEST SCENARIOS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary
    operational = [s for s in scenarios if s['type'] == 'operational']
    strategic = [s for s in scenarios if s['type'] == 'strategic']
    
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total scenarios: {len(scenarios)}")
    report.append(f"  - Operational (k=2): {len(operational)}")
    report.append(f"  - Strategic (k≥3): {len(strategic)}")
    report.append("")
    
    # Operational Scenarios
    report.append("OPERATIONAL SCENARIOS (k=2)")
    report.append("-" * 80)
    for scenario in operational:
        report.append(f"\nScenario {scenario['id']}: {scenario['name']}")
        report.append(f"  Variables: {', '.join(scenario['variables'])}")
        report.append(f"  Queries ({len(scenario['queries'])}):")
        for i, query in enumerate(scenario['queries'], 1):
            report.append(f"    {i}. {query}")
        
        # Show sample ground truth
        if scenario['ground_truth'].get('statistics'):
            stats = scenario['ground_truth']['statistics']
            if 'groups' in stats:
                report.append(f"  Ground Truth Sample (first 3 groups):")
                for group, values in list(stats['groups'].items())[:3]:
                    report.append(f"    {group}: {values}")
            elif 'correlation' in stats:
                report.append(f"  Ground Truth Correlation: {stats['correlation']:.3f}")
        report.append("")
    
    # Strategic Scenarios
    report.append("\nSTRATEGIC SCENARIOS (k≥3)")
    report.append("-" * 80)
    for scenario in strategic:
        report.append(f"\nScenario {scenario['id']}: {scenario['name']}")
        report.append(f"  Variables: {', '.join(scenario['variables'])}")
        report.append(f"  Queries ({len(scenario['queries'])}):")
        for i, query in enumerate(scenario['queries'], 1):
            report.append(f"    {i}. {query}")
        report.append("")
    
    report_str = "\n".join(report)
    
    with open(output_file, 'w') as f:
        f.write(report_str)
    
    print(f"✅ Test report saved to {output_file}")
    return report_str


def main():
    """Main function to generate test scenarios."""
    csv_path = "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv"
    
    print("🔍 Loading HR dataset and creating test scenarios...")
    print()
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    try:
        # Load dataset
        df = load_hr_dataset(csv_path)
        print(f"✅ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {', '.join(df.columns.tolist()[:10])}...")
        print()
        
        # Create scenarios
        operational_scenarios = create_operational_scenarios(df)
        strategic_scenarios = create_strategic_scenarios(df)
        all_scenarios = operational_scenarios + strategic_scenarios
        
        print(f"✅ Created {len(all_scenarios)} test scenarios:")
        print(f"   - {len(operational_scenarios)} operational (k=2)")
        print(f"   - {len(strategic_scenarios)} strategic (k≥3)")
        print()
        
        # Generate report
        report = generate_test_report(all_scenarios)
        print()
        print(report)
        print()
        
        # Save scenarios as JSON
        scenarios_json = {
            "dataset": {
                "path": csv_path,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            },
            "scenarios": all_scenarios,
            "generated_at": datetime.now().isoformat()
        }
        
        with open("test_scenarios.json", 'w') as f:
            json.dump(scenarios_json, f, indent=2, default=str)
        print("✅ Scenarios saved to test_scenarios.json")
        
        # Print summary of queries
        total_queries = sum(len(s['queries']) for s in all_scenarios)
        print()
        print(f"📊 Total test queries: {total_queries}")
        print("   These queries can be used to evaluate system performance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

