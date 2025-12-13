"""
Extract Operational Data Directly
==================================

This script extracts operational insights directly from the knowledge graph
and formats them as tables, bypassing the LLM completely.

Usage:
    python extract_operational_data.py --type manager
    python extract_operational_data.py --type department
    python extract_operational_data.py --type all
"""

import sys
import os
import argparse
import re
from typing import Dict, List, Any, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from knowledge import graph, get_fact_source_document, load_knowledge_graph
    from urllib.parse import unquote
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False


def extract_manager_insights() -> Dict[str, Dict[str, Any]]:
    """Extract manager-level insights from KG."""
    managers = defaultdict(lambda: {
        "engagement": [],
        "performance": [],
        "salary": [],
        "absences": [],
        "team_size": []
    })
    
    if not KG_AVAILABLE or graph is None:
        return {}
    
    for s, p, o in graph:
        # Skip metadata
        predicate_str = str(p)
        if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                            'has_details', 'source_document', 'uploaded_at',
                                            'is_inferred', 'confidence', 'agent_id']):
            continue
        
        # Get source
        sources = get_fact_source_document(
            unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' '),
            unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' '),
            str(o)
        )
        
        # Filter operational insights
        if not any('operational_insights' in str(src).lower() for src, _ in sources):
            continue
        
        # Extract fact
        subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
        predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
        obj = str(o)
        
        fact_text = f"{subject} {predicate} {obj}".lower()
        
        # Look for manager patterns
        manager_match = re.search(r'manager\s+([^,\s]+(?:\s+[^,\s]+)*)', fact_text)
        if not manager_match:
            continue
        
        manager_name = manager_match.group(1).strip()
        
        # Extract metrics
        if 'engagement' in fact_text:
            val_match = re.search(r'(\d+\.?\d*)', obj)
            if val_match:
                managers[manager_name]["engagement"].append(float(val_match.group(1)))
        
        if 'performance' in fact_text:
            val_match = re.search(r'(\d+\.?\d*)', obj)
            if val_match:
                managers[manager_name]["performance"].append(float(val_match.group(1)))
        
        if 'salary' in fact_text:
            val_match = re.search(r'(\d+\.?\d*)', obj)
            if val_match:
                managers[manager_name]["salary"].append(float(val_match.group(1)))
        
        if 'absence' in fact_text:
            val_match = re.search(r'(\d+\.?\d*)', obj)
            if val_match:
                managers[manager_name]["absences"].append(float(val_match.group(1)))
        
        if 'team' in fact_text and ('size' in fact_text or 'count' in fact_text):
            val_match = re.search(r'(\d+)', obj)
            if val_match:
                managers[manager_name]["team_size"].append(int(val_match.group(1)))
    
    # Calculate averages
    result = {}
    for manager, metrics in managers.items():
        result[manager] = {
            "avg_engagement": sum(metrics["engagement"]) / len(metrics["engagement"]) if metrics["engagement"] else None,
            "avg_performance": sum(metrics["performance"]) / len(metrics["performance"]) if metrics["performance"] else None,
            "avg_salary": sum(metrics["salary"]) / len(metrics["salary"]) if metrics["salary"] else None,
            "avg_absences": sum(metrics["absences"]) / len(metrics["absences"]) if metrics["absences"] else None,
            "team_size": metrics["team_size"][0] if metrics["team_size"] else None
        }
    
    return result


def extract_department_insights() -> Dict[str, Dict[str, Any]]:
    """Extract department-level insights from KG."""
    departments = defaultdict(lambda: {
        "salary": [],
        "performance": [],
        "absences": [],
        "count": []
    })
    
    if not KG_AVAILABLE or graph is None:
        return {}
    
    for s, p, o in graph:
        predicate_str = str(p)
        if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                            'has_details', 'source_document', 'uploaded_at',
                                            'is_inferred', 'confidence', 'agent_id']):
            continue
        
        sources = get_fact_source_document(
            unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' '),
            unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' '),
            str(o)
        )
        
        if not any('operational_insights' in str(src).lower() for src, _ in sources):
            continue
        
        subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
        predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
        obj = str(o)
        
        fact_text = f"{subject} {predicate} {obj}".lower()
        
        # Look for department patterns
        dept_match = re.search(r'department\s+([^,\s]+(?:\s+[^,\s]+)*)', fact_text)
        if not dept_match:
            # Try direct department name
            for dept in ['production', 'it/is', 'sales', 'admin', 'executive', 'software engineering']:
                if dept.lower() in fact_text:
                    dept_match = type('obj', (object,), {'group': lambda x: dept})()
                    break
        
        if not dept_match:
            continue
        
        dept_name = dept_match.group(1) if hasattr(dept_match, 'group') else dept_match.group
        
        # Extract metrics
        if 'salary' in fact_text:
            val_match = re.search(r'(\d+\.?\d*)', obj)
            if val_match:
                departments[dept_name]["salary"].append(float(val_match.group(1)))
        
        if 'performance' in fact_text:
            val_match = re.search(r'(\d+\.?\d*)', obj)
            if val_match:
                departments[dept_name]["performance"].append(float(val_match.group(1)))
        
        if 'absence' in fact_text:
            val_match = re.search(r'(\d+\.?\d*)', obj)
            if val_match:
                departments[dept_name]["absences"].append(float(val_match.group(1)))
        
        if 'count' in fact_text or 'employee' in fact_text:
            val_match = re.search(r'(\d+)', obj)
            if val_match:
                departments[dept_name]["count"].append(int(val_match.group(1)))
    
    # Calculate averages
    result = {}
    for dept, metrics in departments.items():
        result[dept] = {
            "avg_salary": sum(metrics["salary"]) / len(metrics["salary"]) if metrics["salary"] else None,
            "avg_performance": sum(metrics["performance"]) / len(metrics["performance"]) if metrics["performance"] else None,
            "avg_absences": sum(metrics["absences"]) / len(metrics["absences"]) if metrics["absences"] else None,
            "employee_count": metrics["count"][0] if metrics["count"] else None
        }
    
    return result


def print_table(data: Dict[str, Dict[str, Any]], title: str):
    """Print data as a formatted table."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    if not data:
        print("\nNo data found.")
        return
    
    # Determine columns
    all_keys = set()
    for row in data.values():
        all_keys.update(row.keys())
    
    columns = sorted(all_keys)
    
    # Print header
    header = " | ".join([col.replace("_", " ").title() for col in ["Name"] + columns])
    print(f"\n{header}")
    print("-" * len(header))
    
    # Print rows
    for name, values in sorted(data.items()):
        row = [name]
        for col in columns:
            val = values.get(col)
            if val is None:
                row.append("N/A")
            elif isinstance(val, float):
                row.append(f"{val:.2f}")
            else:
                row.append(str(val))
        print(" | ".join(row))
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Extract operational data directly from KG")
    parser.add_argument("--type", choices=["manager", "department", "all"], 
                       default="all", help="Type of data to extract")
    
    args = parser.parse_args()
    
    if not KG_AVAILABLE:
        print("❌ Knowledge graph not available")
        return
    
    # Load knowledge graph from disk
    print("📂 Loading knowledge graph from disk...")
    try:
        load_result = load_knowledge_graph()
        if load_result:
            print(f"✅ {load_result}")
    except Exception as e:
        print(f"⚠️  Error loading knowledge graph: {e}")
    
    graph_size = len(graph) if graph else 0
    if graph_size == 0:
        print("\n⚠️  Knowledge graph is empty after loading.")
        print("   Make sure you've uploaded and processed a document first.")
        return
    
    print(f"📊 Knowledge graph size: {graph_size} facts")
    print("🔍 Extracting operational insights...")
    
    if args.type in ["manager", "all"]:
        manager_data = extract_manager_insights()
        print_table(manager_data, "MANAGER INSIGHTS")
    
    if args.type in ["department", "all"]:
        dept_data = extract_department_insights()
        print_table(dept_data, "DEPARTMENT INSIGHTS")


if __name__ == "__main__":
    main()

