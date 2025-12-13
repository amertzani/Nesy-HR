"""
Evaluation Metrics Calculator
==============================

This script extracts real metrics from the HR dataset and system for use in the paper.
It calculates:
- Dataset dimensions (R = rows, N = variables)
- Combinatorial metrics (binomial coefficients)
- Query performance metrics
- System accuracy metrics

Author: Research Brain Team
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import json
from scipy.special import comb
import os
from strategic_queries import find_csv_file_path, load_csv_data


def calculate_binomial_coefficient(n: int, k: int) -> int:
    """Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!)"""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    # Use scipy for large numbers
    return int(comb(n, k, exact=True))


def calculate_pairwise_interactions(k: int) -> int:
    """Calculate number of pairwise interactions for k variables: C(k, 2)"""
    return calculate_binomial_coefficient(k, 2)


def find_cutoff_order(emax: int) -> int:
    """
    Find the cutoff order k* such that E(k) <= E_max.
    k* = floor((1 + sqrt(1 + 8*E_max)) / 2)
    """
    k_star = int((1 + math.sqrt(1 + 8 * emax)) / 2)
    # Verify it satisfies the condition
    while calculate_pairwise_interactions(k_star) > emax:
        k_star -= 1
    return k_star


def extract_metrics_from_knowledge_graph() -> Optional[Dict[str, any]]:
    """
    Fallback: Extract dataset dimensions from knowledge graph when CSV file is not available.
    This estimates R and N by analyzing the knowledge graph structure.
    """
    try:
        from knowledge import graph
        from urllib.parse import unquote
        
        if graph is None or len(graph) == 0:
            return None
        
        # Extract unique employee names (subjects that look like employee names)
        employee_names = set()
        predicates = set()
        
        for s, p, o in graph:
            # Skip metadata triples
            predicate_str = str(p)
            if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str or
                'is_inferred' in predicate_str or 'confidence' in predicate_str or
                'agent_id' in predicate_str):
                continue
            
            # Extract subject (employee name)
            subject_uri_str = str(s)
            if 'urn:entity:' in subject_uri_str:
                subject = subject_uri_str.split('urn:entity:')[-1]
            else:
                subject = subject_uri_str
            subject = unquote(subject).replace('_', ' ')
            
            # Check if subject looks like an employee name (Last, First format)
            if ',' in subject and len(subject.split(',')) == 2:
                employee_names.add(subject)
            
            # Extract predicate (column name)
            predicate_uri_str = str(p)
            if 'urn:predicate:' in predicate_uri_str:
                predicate = predicate_uri_str.split('urn:predicate:')[-1]
            elif 'has_' in predicate_uri_str:
                predicate = predicate_uri_str.split('has_')[-1]
            else:
                predicate = predicate_uri_str
            predicate = unquote(predicate).replace('_', ' ').replace('has ', '')
            
            # Filter out common non-column predicates
            if predicate and predicate not in ['source document', 'uploaded at', 'agent id', 'confidence']:
                predicates.add(predicate)
        
        if len(employee_names) == 0:
            return None
        
        R = len(employee_names)
        N = len(predicates)
        
        # Try to classify predicates as numeric/categorical (heuristic)
        numeric_cols = []
        categorical_cols = []
        
        # Common numeric column indicators
        numeric_keywords = ['salary', 'score', 'count', 'number', 'age', 'year', 'rate', 
                           'percentage', 'amount', 'value', 'id', 'num']
        
        for pred in predicates:
            pred_lower = pred.lower()
            if any(keyword in pred_lower for keyword in numeric_keywords):
                numeric_cols.append(pred)
            else:
                categorical_cols.append(pred)
        
        return {
            "R": R,
            "N": N,
            "column_names": sorted(list(predicates)),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "file_path": None,
            "source": "knowledge_graph_estimation"
        }
    except Exception as e:
        print(f"⚠️  Error extracting from knowledge graph: {e}")
        return None


def extract_dataset_metrics(csv_file_path: Optional[str] = None) -> Dict[str, any]:
    """
    Extract dataset metrics: R (rows) and N (variables/columns).
    
    Returns:
        Dictionary with:
        - R: number of employee records
        - N: number of variables
        - column_names: list of column names
        - numeric_columns: list of numeric column names
        - categorical_columns: list of categorical column names
    """
    if csv_file_path is None:
        csv_file_path = find_csv_file_path()
    
    # If still not found, try to find from documents_store.json
    if csv_file_path is None or not os.path.exists(csv_file_path):
        try:
            import json
            import glob
            import tempfile
            
            # Read documents_store.json
            docs_file = "documents_store.json"
            if os.path.exists(docs_file):
                with open(docs_file, 'r') as f:
                    docs_data = json.load(f)
                
                csv_docs = [d for d in docs_data.get('documents', []) 
                           if d.get('type', '').lower() == 'csv']
                
                if csv_docs:
                    doc_name = csv_docs[-1]['name']  # Most recent
                    
                    # Search in multiple locations
                    search_paths = [
                        doc_name,  # Current directory
                        os.path.join(os.getcwd(), doc_name),
                        os.path.join(tempfile.gettempdir(), doc_name),
                        os.path.join('/tmp', doc_name),
                        os.path.join('/var/tmp', doc_name),
                    ]
                    
                    # Also search with glob patterns
                    for temp_dir in [tempfile.gettempdir(), '/tmp', '/var/tmp']:
                        if os.path.exists(temp_dir):
                            pattern = os.path.join(temp_dir, f'*{doc_name}*')
                            search_paths.extend(glob.glob(pattern))
                            pattern = os.path.join(temp_dir, f'*HRDataset*')
                            search_paths.extend(glob.glob(pattern))
                    
                    for path in search_paths:
                        if path and os.path.exists(path) and path.endswith('.csv'):
                            csv_file_path = path
                            print(f"✅ Found CSV file: {csv_file_path}")
                            break
        except Exception as e:
            print(f"⚠️  Error searching for CSV: {e}")
    
    # Try to load from CSV file
    if csv_file_path and os.path.exists(csv_file_path):
        df = load_csv_data(csv_file_path)
        if df is not None and len(df) > 0:
            R = len(df)
            N = len(df.columns)
            
            # Classify columns
            numeric_cols = []
            categorical_cols = []
            
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            return {
                "R": R,
                "N": N,
                "column_names": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "file_path": csv_file_path,
                "source": "csv_file"
            }
    
    # Fallback: Extract from knowledge graph
    print("⚠️  CSV file not found. Attempting to extract metrics from knowledge graph...")
    kg_metrics = extract_metrics_from_knowledge_graph()
    
    if kg_metrics:
        print(f"✅ Extracted metrics from knowledge graph: R={kg_metrics['R']}, N={kg_metrics['N']}")
        print("   Note: These are estimates based on knowledge graph structure.")
        return kg_metrics
    
    # Final fallback: Use documents_store.json info
    try:
        import json
        docs_file = "documents_store.json"
        if os.path.exists(docs_file):
            with open(docs_file, 'r') as f:
                docs_data = json.load(f)
            
            csv_docs = [d for d in docs_data.get('documents', []) 
                       if d.get('type', '').lower() == 'csv']
            
            if csv_docs:
                latest_doc = csv_docs[-1]
                facts_count = latest_doc.get('facts_extracted', 0)
                
                # Rough estimate: if we have facts, we can estimate dimensions
                # Typical: ~70 facts per employee (one per column)
                if facts_count > 0:
                    estimated_R = max(1, facts_count // 70)
                    estimated_N = 36  # Common HR dataset size
                    
                    print(f"⚠️  Using rough estimates from documents_store.json")
                    print(f"   Estimated R={estimated_R} (based on {facts_count} facts)")
                    print(f"   Estimated N={estimated_N} (default value)")
                    
                    return {
                        "R": estimated_R,
                        "N": estimated_N,
                        "column_names": [],
                        "numeric_columns": [],
                        "categorical_columns": [],
                        "file_path": None,
                        "source": "documents_store_estimate"
                    }
    except Exception as e:
        print(f"⚠️  Error reading documents_store.json: {e}")
    
    # If all else fails, raise error with helpful message
    raise FileNotFoundError(
        f"CSV file not found and could not extract metrics from knowledge graph.\n"
        f"Please either:\n"
        f"  1. Provide the CSV file path: python evaluation_metrics.py /path/to/file.csv\n"
        f"  2. Re-upload the CSV file through the system interface\n"
        f"  3. Ensure the knowledge graph has been populated with data"
    )


def calculate_combinatorial_metrics(N: int, K_max: int = 5) -> Dict[str, any]:
    """
    Calculate combinatorial metrics for the evaluation section.
    
    Args:
        N: Number of variables
        K_max: Maximum order to consider
    
    Returns:
        Dictionary with combinatorial metrics
    """
    metrics = {
        "N": N,
        "K_max": K_max,
        "combinations_by_k": {},
        "total_combinations": 0,
        "pairwise_interactions": {}
    }
    
    # Calculate combinations for each k from 2 to K_max
    for k in range(2, K_max + 1):
        c_k = calculate_binomial_coefficient(N, k)
        metrics["combinations_by_k"][k] = c_k
        metrics["total_combinations"] += c_k
        
        # Calculate pairwise interactions for k variables
        E_k = calculate_pairwise_interactions(k)
        metrics["pairwise_interactions"][k] = E_k
    
    # Special focus on k=2 (operational queries)
    metrics["operational_combinations"] = metrics["combinations_by_k"][2]
    
    # Calculate cutoff order for different E_max values
    metrics["cutoff_orders"] = {}
    for emax in [1, 2, 3, 6, 10]:
        k_star = find_cutoff_order(emax)
        metrics["cutoff_orders"][emax] = k_star
    
    return metrics


def generate_latex_equations(metrics: Dict[str, any]) -> str:
    """
    Generate LaTeX code with real values to replace in the paper.
    
    Returns:
        String with LaTeX code snippets ready to paste
    """
    R = metrics["dataset"]["R"]
    N = metrics["dataset"]["N"]
    comb_metrics = metrics["combinatorial"]
    
    latex_output = []
    latex_output.append("%" + "=" * 70)
    latex_output.append("% REAL METRICS FROM DATASET - Replace dummy values in paper")
    latex_output.append("%" + "=" * 70)
    latex_output.append("")
    
    # Dataset dimensions
    latex_output.append(f"% Dataset dimensions:")
    latex_output.append(f"% R = {R} (number of employee records)")
    latex_output.append(f"% N = {N} (number of variables)")
    latex_output.append("")
    
    # Equation 1: k-combinations
    latex_output.append(f"% Equation: |C_k| = C(N, k)")
    for k in range(2, min(6, comb_metrics["K_max"] + 1)):
        c_k = comb_metrics["combinations_by_k"][k]
        latex_output.append(f"% C({N}, {k}) = {c_k}")
    latex_output.append("")
    
    # Equation 2: Total combinations
    total = comb_metrics["total_combinations"]
    latex_output.append(f"% Total combinations up to K_max={comb_metrics['K_max']}:")
    latex_output.append(f"% |C_<=K_max| = {total}")
    latex_output.append("")
    
    # Equation 3: Operational combinations (k=2)
    op_comb = comb_metrics["operational_combinations"]
    latex_output.append(f"% Operational combinations (k=2):")
    latex_output.append(f"% |C_2| = C({N}, 2) = {N} × {N-1} / 2 = {op_comb}")
    latex_output.append("")
    
    # Equation 4: Pairwise interactions
    latex_output.append(f"% Pairwise interactions E(k) = C(k, 2):")
    for k in range(2, min(6, comb_metrics["K_max"] + 1)):
        E_k = comb_metrics["pairwise_interactions"][k]
        latex_output.append(f"% E({k}) = C({k}, 2) = {E_k}")
    latex_output.append("")
    
    # Equation 5: Cutoff order
    latex_output.append(f"% Cutoff orders for different E_max:")
    for emax, k_star in comb_metrics["cutoff_orders"].items():
        E_k_star = calculate_pairwise_interactions(k_star)
        latex_output.append(f"% E_max = {emax} → k* = {k_star} (E({k_star}) = {E_k_star})")
    latex_output.append("")
    
    # Ready-to-use LaTeX snippets
    latex_output.append("%" + "=" * 70)
    latex_output.append("% READY-TO-USE LATEX SNIPPETS")
    latex_output.append("%" + "=" * 70)
    latex_output.append("")
    
    # Replace in text
    latex_output.append(f"% Replace: 'R = 311' with:")
    latex_output.append(f"R = {R}")
    latex_output.append("")
    
    latex_output.append(f"% Replace: 'N = 36' with:")
    latex_output.append(f"N = {N}")
    latex_output.append("")
    
    latex_output.append(f"% Replace in equation (eq:operational-combinations):")
    latex_output.append(f"\\left|\\mathcal{{C}}_2\\right| = \\binom{{{N}}}{{2}} = \\frac{{{N} \\times {N-1}}}{{2}} = {op_comb},")
    latex_output.append("")
    
    # Example variable names
    cols = metrics["dataset"]["column_names"]
    hr_vars = [c for c in cols if any(keyword in c.lower() for keyword in 
                ['salary', 'department', 'employment', 'recruitment', 'performance', 
                 'engagement', 'manager', 'absence', 'status'])]
    
    if hr_vars:
        latex_output.append(f"% Example HR variables found in dataset (for text examples):")
        for var in hr_vars[:8]:
            latex_output.append(f"% \\texttt{{{var}}}")
        latex_output.append("")
    
    return "\n".join(latex_output)


def generate_evaluation_report(metrics: Dict[str, any], output_file: Optional[str] = None) -> str:
    """
    Generate a comprehensive evaluation report with all metrics.
    
    Args:
        metrics: Dictionary with dataset and combinatorial metrics
        output_file: Optional file path to save the report
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("EVALUATION METRICS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Dataset Information
    report.append("DATASET DIMENSIONS")
    report.append("-" * 80)
    dataset = metrics["dataset"]
    report.append(f"Number of employee records (R): {dataset['R']}")
    report.append(f"Number of variables (N): {dataset['N']}")
    report.append(f"Dataset file: {dataset['file_path']}")
    report.append("")
    report.append(f"Numeric columns ({len(dataset['numeric_columns'])}):")
    for col in dataset['numeric_columns'][:10]:
        report.append(f"  - {col}")
    if len(dataset['numeric_columns']) > 10:
        report.append(f"  ... and {len(dataset['numeric_columns']) - 10} more")
    report.append("")
    report.append(f"Categorical columns ({len(dataset['categorical_columns'])}):")
    for col in dataset['categorical_columns'][:10]:
        report.append(f"  - {col}")
    if len(dataset['categorical_columns']) > 10:
        report.append(f"  ... and {len(dataset['categorical_columns']) - 10} more")
    report.append("")
    
    # Combinatorial Metrics
    report.append("COMBINATORIAL METRICS")
    report.append("-" * 80)
    comb_metrics = metrics["combinatorial"]
    report.append(f"Maximum order considered (K_max): {comb_metrics['K_max']}")
    report.append("")
    report.append("Combinations by order k:")
    for k in sorted(comb_metrics["combinations_by_k"].keys()):
        c_k = comb_metrics["combinations_by_k"][k]
        E_k = comb_metrics["pairwise_interactions"][k]
        report.append(f"  k = {k}: C({comb_metrics['N']}, {k}) = {c_k:,} combinations, E({k}) = {E_k} pairwise interactions")
    report.append("")
    report.append(f"Total combinations (k=2 to k={comb_metrics['K_max']}): {comb_metrics['total_combinations']:,}")
    report.append(f"Operational combinations (k=2): {comb_metrics['operational_combinations']:,}")
    report.append("")
    
    # Cutoff Orders
    report.append("CUTOFF ORDERS (Operational vs Strategic)")
    report.append("-" * 80)
    report.append("For different values of E_max (maximum pairwise interactions for operational queries):")
    for emax in sorted(comb_metrics["cutoff_orders"].keys()):
        k_star = comb_metrics["cutoff_orders"][emax]
        E_k_star = calculate_pairwise_interactions(k_star)
        report.append(f"  E_max = {emax} → k* = {k_star} (E({k_star}) = {E_k_star})")
        if k_star == 2:
            report.append(f"    → Operational: k = 2, Strategic: k ≥ 3")
    report.append("")
    
    report_str = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_str)
        print(f"✅ Report saved to {output_file}")
    
    return report_str


def main():
    """Main function to extract and display all metrics."""
    print("🔍 Extracting evaluation metrics from dataset...")
    print()
    
    # Allow CSV path to be provided as command line argument or environment variable
    import sys
    csv_path = None
    
    # Default to the actual CSV file location
    default_csv_path = "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"📁 Using provided CSV path: {csv_path}")
    elif os.getenv('CSV_FILE_PATH'):
        csv_path = os.getenv('CSV_FILE_PATH')
        print(f"📁 Using CSV path from environment: {csv_path}")
    elif os.path.exists(default_csv_path):
        csv_path = default_csv_path
        print(f"📁 Using default CSV path: {csv_path}")
    
    try:
        # Extract dataset metrics
        dataset_metrics = extract_dataset_metrics(csv_path)
        print(f"✅ Loaded dataset: {dataset_metrics['R']} rows, {dataset_metrics['N']} columns")
        
        # Calculate combinatorial metrics
        combinatorial_metrics = calculate_combinatorial_metrics(
            N=dataset_metrics['N'],
            K_max=5
        )
        
        # Combine all metrics
        all_metrics = {
            "dataset": dataset_metrics,
            "combinatorial": combinatorial_metrics
        }
        
        # Generate report
        report = generate_evaluation_report(all_metrics, "evaluation_metrics_report.txt")
        print()
        print(report)
        print()
        
        # Generate LaTeX snippets
        latex_snippets = generate_latex_equations(all_metrics)
        print("📝 LaTeX snippets for paper:")
        print()
        print(latex_snippets)
        print()
        
        # Save LaTeX snippets
        with open("latex_metrics_snippets.tex", 'w') as f:
            f.write(latex_snippets)
        print("✅ LaTeX snippets saved to latex_metrics_snippets.tex")
        
        # Save JSON metrics
        with open("evaluation_metrics.json", 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print("✅ Metrics saved to evaluation_metrics.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

