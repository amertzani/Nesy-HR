"""
Terminal Query Answer Tool
==========================

Answers natural language queries by searching the knowledge graph directly.
Bypasses LLM and extracts answers from stored facts.

Usage:
    python answer_query_terminal.py "Which department has the highest average performance score?"
    python answer_query_terminal.py "What is the average engagement by manager?"
    python answer_query_terminal.py "Which manager has the highest team engagement?"
"""

import sys
import os
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import unquote
from collections import defaultdict
try:
    import pandas as pd
except ImportError:
    pd = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from knowledge import graph, get_fact_source_document, load_knowledge_graph
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    print("❌ Could not import knowledge graph modules")


def extract_query_intent(query: str) -> Dict[str, Any]:
    """
    Extract intent from natural language query.
    Returns what metric, group, and operation (max/min/average) is being asked.
    """
    query_lower = query.lower()
    
    intent = {
        "metric": None,
        "group_by": None,
        "operation": None,  # "max", "min", "average", "distribution", "fact_retrieval"
        "comparison": None,  # "highest", "lowest", "average"
        "employee_name": None,  # For fact-based queries
        "fact_query": False,  # True if query is asking for facts
        "strategic_query": False,  # True if query is strategic (multi-condition)
        "correlation_query": False,  # True if query compares multiple variables
        "conditions": [],  # List of conditions for strategic queries
        "department_name": None  # For department-specific queries
    }
    
    # Detect fact-based queries
    fact_keywords = ["retrieve facts", "give me facts", "show me facts", "facts about", 
                     "information about", "facts related", "what facts", "all facts", 
                     "stored about", "facts stored", "what information", "information do we have"]
    if any(kw in query_lower for kw in fact_keywords):
        intent["fact_query"] = True
        intent["operation"] = "fact_retrieval"
        
        # Also check for "show me information about" which should trigger fact retrieval
        if "show me information about" in query_lower or "information about" in query_lower:
            if "employee" in query_lower and ("worst" in query_lower or "lowest" in query_lower or "performance" in query_lower):
                intent["operation"] = "fact_retrieval_min"
                # Don't override if we already detected it
                if "worst" in query_lower or "lowest" in query_lower:
                    intent["operation"] = "fact_retrieval_min"
        
        # Try to extract employee name from query
        import re
        # Pattern: "employee Name, Last" or "Name, Last" or "about Name, Last"
        name_patterns = [
            r'employee\s+([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z])?)',
            r'about\s+([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z])?)',
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z])?)'
        ]
        for pattern in name_patterns:
            name_matches = re.findall(pattern, query)
            if name_matches:
                intent["employee_name"] = name_matches[0]
                break
        
        # Check if asking for highest/lowest
        if any(w in query_lower for w in ["highest", "maximum", "max", "top", "best", "highest paid", "paid employee"]):
            intent["operation"] = "fact_retrieval_max"
        elif any(w in query_lower for w in ["lowest", "minimum", "min", "bottom", "worst", "lowest performance", "worst performance"]):
            intent["operation"] = "fact_retrieval_min"
    
    # Detect metric
    metrics = {
        "performance": ["performance", "perfscore", "perf score"],
        "engagement": ["engagement", "engagement survey"],
        "salary": ["salary", "compensation", "pay"],
        "absence": ["absence", "absences", "absent"],
        "satisfaction": ["satisfaction", "emp satisfaction"],
        "special_projects": ["special projects", "specialprojects", "special project", "projects count"]
    }
    
    for metric_name, keywords in metrics.items():
        if any(kw in query_lower for kw in keywords):
            intent["metric"] = metric_name
            break
    
    # Detect group_by
    groups = {
        "department": ["department", "dept"],
        "manager": ["manager", "team"],
        "recruitment": ["recruitment", "source", "channel"],
        "employment": ["employment status", "status", "active", "terminated"]
    }
    
    for group_name, keywords in groups.items():
        if any(kw in query_lower for kw in keywords):
            intent["group_by"] = group_name
            break
    
    # Detect operation/comparison - but don't override fact_retrieval operations
    if not intent.get("fact_query") or intent.get("operation") not in ["fact_retrieval_max", "fact_retrieval_min"]:
        if any(word in query_lower for word in ["highest", "maximum", "max", "top", "best", "most"]):
            if intent.get("fact_query"):
                intent["operation"] = "fact_retrieval_max"
            else:
                intent["operation"] = "max"
            intent["comparison"] = "highest"
        elif any(word in query_lower for word in ["lowest", "minimum", "min", "bottom", "worst", "least"]):
            if intent.get("fact_query"):
                intent["operation"] = "fact_retrieval_min"
            else:
                intent["operation"] = "min"
            intent["comparison"] = "lowest"
    elif any(word in query_lower for word in ["average", "mean", "avg"]):
        intent["operation"] = "average"
        intent["comparison"] = "average"
    elif any(word in query_lower for word in ["distribution", "vary", "compare"]):
        intent["operation"] = "distribution"
    
    # Detect strategic queries (multi-condition employee finding)
    strategic_patterns = [
        ("identify employees", ["identify", "find", "show"]),
        ("with high performance", ["high performance", "high perf"]),
        ("with low engagement", ["low engagement"]),
        ("with low satisfaction", ["low satisfaction"]),
        ("many special projects", ["many special projects", "special projects"]),
        ("many absences", ["many absences", "high absences"])
    ]
    
    if any("identify employees" in query_lower or "find employees" in query_lower for _ in [1]):
        intent["strategic_query"] = True
        conditions = []
        if "high performance" in query_lower or "high perf" in query_lower:
            conditions.append({"metric": "performance", "operator": "high"})
        if "low engagement" in query_lower:
            conditions.append({"metric": "engagement", "operator": "low"})
        if "low satisfaction" in query_lower:
            conditions.append({"metric": "satisfaction", "operator": "low"})
        if "many special projects" in query_lower or ("special projects" in query_lower and "many" in query_lower):
            conditions.append({"metric": "special_projects", "operator": "high"})
        if "many absences" in query_lower or "high absences" in query_lower:
            conditions.append({"metric": "absences", "operator": "high"})
        intent["conditions"] = conditions
        
        # Mark as strategic query if conditions are present
        if conditions:
            intent["strategic_query"] = True
    
    # Detect correlation queries (department-salary-performance)
    if any(phrase in query_lower for phrase in ["high salaries but low performance", "low salary and high performance", 
                                                 "relationship between salary, performance", "analyze the relationship"]):
        intent["correlation_query"] = True
        if "high salaries but low performance" in query_lower:
            intent["operation"] = "high_salary_low_perf"
        elif "low salary and high performance" in query_lower:
            intent["operation"] = "low_salary_high_perf"
        elif "analyze" in query_lower or "relationship" in query_lower:
            intent["operation"] = "correlation_analysis"
    
    # Detect department-specific fact queries
    dept_patterns = {
        "production": ["production", "prod"],
        "sales": ["sales"],
        "it/is": ["it/is", "it/is department", "it department"],
        "admin": ["admin offices", "admin"]
    }
    for dept_name, patterns in dept_patterns.items():
        if any(p in query_lower for p in patterns) and ("facts" in query_lower or "information" in query_lower or "about" in query_lower):
            intent["department_name"] = dept_name
            intent["fact_query"] = True
            break
    
    # Detect queries about employees in a department
    if "employees in" in query_lower or "employees from" in query_lower:
        for dept_name, patterns in dept_patterns.items():
            if any(p in query_lower for p in patterns):
                intent["department_name"] = dept_name
                intent["fact_query"] = True
                intent["operation"] = "department_employees_facts"
                break
    
    return intent


def search_facts_for_answer(intent: Dict[str, Any], limit: int = 200) -> List[Dict[str, Any]]:
    """
    Search knowledge graph for facts matching the query intent.
    """
    if not KG_AVAILABLE or graph is None:
        return []
    
    # Build search keywords - be more specific
    keywords = []
    if intent["metric"]:
        keywords.append(intent["metric"])
        # Add specific variations
        if intent["metric"] == "performance":
            keywords.extend(["performance", "perfscore", "perf score", "performance score"])
        elif intent["metric"] == "engagement":
            keywords.extend(["engagement", "engagement survey", "engagement score"])
        elif intent["metric"] == "salary":
            keywords.extend(["salary", "compensation"])
        elif intent["metric"] == "absence":
            keywords.extend(["absence", "absences"])
        elif intent["metric"] == "special_projects":
            keywords.extend(["special projects", "specialprojects", "special project", "projects count"])
    
    if intent["group_by"]:
        keywords.append(intent["group_by"])
        if intent["group_by"] == "department":
            keywords.extend(["department", "dept"])
        elif intent["group_by"] == "manager":
            keywords.extend(["manager", "team"])
    
    # For max/min queries, also look for "average" in facts
    if intent["operation"] in ["max", "min"]:
        keywords.append("average")
    
    if not keywords:
        return []
    
    # Search facts
    matches = []
    keywords_lower = [k.lower() for k in keywords]
    
    for s, p, o in graph:
        # Skip metadata
        predicate_str = str(p)
        if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                            'has_details', 'source_document', 'uploaded_at',
                                            'is_inferred', 'confidence', 'agent_id']):
            continue
        
        # Extract fact components
        subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
        predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
        obj = str(o)
        
        fact_text = f"{subject} {predicate} {obj}".lower()
        
        # Check keyword match - require metric AND group_by to match
        metric_match = False
        group_match = False
        
        if intent["metric"]:
            metric_keywords = {
                "performance": ["performance", "perfscore", "perf score"],
                "engagement": ["engagement", "engagement survey"],
                "salary": ["salary", "compensation"],
                "absence": ["absence", "absences"]
            }
            metric_kws = metric_keywords.get(intent["metric"], [intent["metric"]])
            metric_match = any(kw in fact_text for kw in metric_kws)
        
        if intent["group_by"]:
            group_keywords = {
                "department": ["department", "dept"],
                "manager": ["manager", "team"],
                "recruitment": ["recruitment", "source"],
                "employment": ["employment", "status"]
            }
            group_kws = group_keywords.get(intent["group_by"], [intent["group_by"]])
            group_match = any(kw in fact_text for kw in group_kws)
        
        # Both must match (unless one is missing from intent)
        if intent["metric"] and intent["group_by"]:
            if not (metric_match and group_match):
                continue
        elif intent["metric"] and not metric_match:
            continue
        elif intent["group_by"] and not group_match:
            continue
        
        # Filter by source (prefer operational insights)
        sources = get_fact_source_document(subject, predicate, obj)
        has_operational = any('operational_insights' in str(src).lower() for src, _ in sources)
        
        matches.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "fact_text": f"{subject} → {predicate} → {obj}",
            "is_operational": has_operational,
            "sources": [str(src) for src, _ in sources]
        })
        
        if len(matches) >= limit:
            break
    
    # Sort: operational insights first
    matches.sort(key=lambda x: (not x["is_operational"], x["fact_text"]))
    
    return matches


def extract_value_from_fact(fact: Dict[str, Any]) -> Optional[float]:
    """Extract numeric value from a fact object."""
    obj = fact["object"]
    
    # Try to extract number
    match = re.search(r'(\d+\.?\d*)', str(obj))
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    
    return None


def extract_entity_from_fact(fact: Dict[str, Any], entity_type: str) -> Optional[str]:
    """Extract entity (department, manager, etc.) from fact."""
    subject = fact["subject"].lower()
    predicate = fact["predicate"].lower()
    fact_text = fact["fact_text"].lower()
    
    if entity_type == "department":
        # Known department names
        departments = [
            "production", "sales", "it/is", "it", "is", 
            "admin offices", "admin", "executive office", "executive",
            "software engineering", "software"
        ]
        
        # Check subject first (most reliable)
        for dept in departments:
            if dept in subject:
                # Return the full department name
                if dept in ["it", "is"]:
                    return "IT/IS"
                elif dept == "admin":
                    return "Admin Offices"
                elif dept == "executive":
                    return "Executive Office"
                elif dept == "software":
                    return "Software Engineering"
                else:
                    return dept.title()
        
        # Check fact text
        dept_patterns = [
            r'department\s+([^,\s→]+(?:\s+[^,\s→]+)*)',
            r'(production|sales|it/is|admin offices|executive office|software engineering)',
        ]
        for pattern in dept_patterns:
            match = re.search(pattern, fact_text)
            if match:
                dept_name = match.group(1).strip()
                # Normalize
                if dept_name.lower() in ["it", "is"]:
                    return "IT/IS"
                return dept_name.title()
    
    elif entity_type == "manager":
        # Check subject - managers are often in subject
        if "manager" in subject:
            # Extract name after "manager"
            # Pattern: "manager X" or "manager X's team"
            name_match = re.search(r'manager\s+([^,\s→\']+)(?:\s+[^,\s→\']+)*(?:\'s)?', subject)
            if name_match:
                name = name_match.group(0).replace("manager", "").strip()
                # Remove "'s team" if present
                name = re.sub(r'\'s\s+team$', '', name).strip()
                if name:
                    return name.title()
        
        # Check fact text
        manager_match = re.search(r'manager\s+([^,\s→\']+)(?:\s+[^,\s→\']+)*(?:\'s)?', fact_text)
        if manager_match:
            name = manager_match.group(0).replace("manager", "").strip()
            name = re.sub(r'\'s\s+team$', '', name).strip()
            if name:
                return name.title()
        
        # If subject itself looks like a manager name (no "manager" prefix)
        # Check if it's a known manager name pattern
        if not any(word in subject for word in ["department", "recruitment", "employee"]):
            # Might be a manager name directly
            if len(subject.split()) <= 3:  # Reasonable name length
                return subject.title()
    
    elif entity_type == "recruitment":
        if "recruitment" in subject or "source" in subject:
            # Extract source name
            source_match = re.search(r'(recruitment\s+source|source)\s+([^,\s→]+(?:\s+[^,\s→]+)*)', fact_text)
            if source_match:
                return source_match.group(2).strip().title()
    
    return None


def transform_insights_to_list_format(insights: Dict[str, Any], df: Any = None) -> Dict[str, Any]:
    """
    Transform insights from dict format to list format expected by answer_query.
    Preserves all decimal precision.
    """
    from operational_queries import normalize_column_name
    
    result = {
        "by_department": [],
        "by_manager": [],
        "by_recruitment_source": []
    }
    
    # Transform by_department
    if "by_department" in insights and isinstance(insights["by_department"], dict):
        dept_data = insights["by_department"]
        
        # Get all unique departments
        departments = set()
        if "performance" in dept_data:
            departments.update(dept_data["performance"].keys())
        if "salary" in dept_data:
            departments.update(dept_data["salary"].keys())
        if "engagement" in dept_data:
            departments.update(dept_data["engagement"].keys())
        
        # Compute employee counts if df is provided
        employee_counts = {}
        if df is not None:
            dept_col = normalize_column_name(df, "Department")
            if dept_col:
                employee_counts = df[dept_col].value_counts().to_dict()
        
        for dept in departments:
            dept_obj = {"department": dept}
            if "performance" in dept_data and dept in dept_data["performance"]:
                dept_obj["avg_performance_score"] = dept_data["performance"][dept]
            if "salary" in dept_data and dept in dept_data["salary"]:
                dept_obj["avg_salary"] = dept_data["salary"][dept]
            if "engagement" in dept_data and dept in dept_data["engagement"]:
                dept_obj["avg_engagement"] = dept_data["engagement"][dept]
            if dept in employee_counts:
                dept_obj["employee_count"] = employee_counts[dept]
            result["by_department"].append(dept_obj)
    
    # Transform by_manager
    if "by_manager" in insights and isinstance(insights["by_manager"], dict):
        mgr_data = insights["by_manager"]
        
        managers = set()
        if "engagement" in mgr_data:
            managers.update(mgr_data["engagement"].keys())
        if "performance" in mgr_data:
            managers.update(mgr_data["performance"].keys())
        if "salary" in mgr_data:
            managers.update(mgr_data["salary"].keys())
        
        employee_counts = {}
        if df is not None:
            mgr_col = normalize_column_name(df, "ManagerName")
            if mgr_col:
                employee_counts = df[mgr_col].value_counts().to_dict()
        
        for mgr in managers:
            mgr_obj = {"manager": mgr}
            if "engagement" in mgr_data and mgr in mgr_data["engagement"]:
                mgr_obj["avg_engagement"] = mgr_data["engagement"][mgr]
            if "performance" in mgr_data and mgr in mgr_data["performance"]:
                mgr_obj["avg_performance_score"] = mgr_data["performance"][mgr]
            if "salary" in mgr_data and mgr in mgr_data["salary"]:
                mgr_obj["avg_salary"] = mgr_data["salary"][mgr]
            if mgr in employee_counts:
                mgr_obj["employee_count"] = employee_counts[mgr]
            result["by_manager"].append(mgr_obj)
    
    # Transform by_recruitment
    if "by_recruitment" in insights and isinstance(insights["by_recruitment"], dict):
        rec_data = insights["by_recruitment"]
        
        sources = set()
        if "performance" in rec_data:
            sources.update(rec_data["performance"].keys())
        if "salary" in rec_data:
            sources.update(rec_data["salary"].keys())
        
        employee_counts = {}
        if df is not None:
            rec_col = normalize_column_name(df, "RecruitmentSource")
            if rec_col:
                employee_counts = df[rec_col].value_counts().to_dict()
        
        for source in sources:
            source_obj = {"recruitment_source": source}
            if "performance" in rec_data and source in rec_data["performance"]:
                source_obj["avg_performance_score"] = rec_data["performance"][source]
            if "salary" in rec_data and source in rec_data["salary"]:
                source_obj["avg_salary"] = rec_data["salary"][source]
            if source in employee_counts:
                source_obj["employee_count"] = employee_counts[source]
            result["by_recruitment_source"].append(source_obj)
    
    return result


def get_operational_insights_from_csv() -> Optional[Dict[str, Any]]:
    """
    Get operational insights by computing them directly from CSV (same as frontend).
    This ensures we get the exact same data the frontend sees.
    Returns insights in list format for answer_query.
    """
    try:
        from strategic_queries import find_csv_file_path
        from operational_queries import load_csv_data
        from operational_queries import compute_operational_insights
        from documents_store import get_all_documents
        import tempfile
        import glob
        
        # PRIORITY 1: Find the CSV that was actually uploaded (from documents_store.json)
        csv_path = None
        documents = get_all_documents()
        csv_docs = [d for d in documents if d.get('type', '').lower() == 'csv' or d.get('file_type', '').lower() == 'csv']
        
        if csv_docs:
            doc_name = csv_docs[-1]['name']  # Most recent
            print(f"📁 Looking for uploaded CSV: {doc_name}")
            
            # Check common upload locations
            temp_dir = tempfile.gettempdir()
            possible_paths = [
                os.path.join(temp_dir, doc_name),
                os.path.join('/tmp', doc_name),
                os.path.join('/var/tmp', doc_name),
                f"/Users/s20/Desktop/Gnoses/HR Data/{doc_name}",  # Original location
                doc_name,  # Current directory
            ]
            
            # Also search for the file
            for search_dir in [temp_dir, '/tmp', '/var/tmp', '/Users/s20/Desktop/Gnoses/HR Data']:
                if os.path.exists(search_dir):
                    pattern = os.path.join(search_dir, f'*{doc_name}*')
                    matches = glob.glob(pattern)
                    possible_paths.extend(matches)
            
            for path in possible_paths:
                if path and os.path.exists(path) and path.endswith('.csv'):
                    csv_path = path
                    print(f"✅ Found uploaded CSV: {csv_path}")
                    break
        
        # PRIORITY 2: Try find_csv_file_path (from document agents)
        if not csv_path:
            csv_path = find_csv_file_path()
            if csv_path:
                print(f"✅ Found CSV from document agents: {csv_path}")
        
        # PRIORITY 3: Try direct paths
        if not csv_path:
            direct_paths = [
                "/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv",  # User's uploaded file
                "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv",  # Fallback
            ]
            for path in direct_paths:
                if path and os.path.exists(path):
                    csv_path = path
                    print(f"✅ Using CSV: {csv_path}")
                    break
        
        if not csv_path:
            print("⚠️  Could not find CSV file for operational insights")
            return None
        
        # Load and compute insights
        print(f"📊 Loading CSV: {csv_path}")
        df = load_csv_data(csv_path)
        if df is None or len(df) == 0:
            print("⚠️  Could not load CSV data")
            return None
        
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        insights_dict = compute_operational_insights(df=df)
        if insights_dict:
            # Transform to list format
            insights = transform_insights_to_list_format(insights_dict, df)
            return insights
        return None
    except Exception as e:
        print(f"⚠️  Error computing operational insights: {e}")
        import traceback
        traceback.print_exc()
        return None


def answer_query(query: str) -> Dict[str, Any]:
    """
    Answer a natural language query using operational insights (same as frontend).
    Falls back to KG search if operational insights not available.
    """
    result = {
        "query": query,
        "answer": None,
        "facts_used": [],
        "intent": None,
        "method": None
    }
    
    if not KG_AVAILABLE:
        result["answer"] = "Knowledge graph not available"
        return result
    
    # Extract intent
    intent = extract_query_intent(query)
    result["intent"] = intent
    
    # Handle fact-based queries first
    if intent.get("fact_query"):
        return handle_fact_based_query(query, intent, result)
    
    # Handle strategic queries (multi-condition employee finding)
    if intent.get("strategic_query"):
        return handle_strategic_query(query, intent, result)
    
    # Handle correlation queries (department-salary-performance)
    if intent.get("correlation_query"):
        return handle_correlation_query(query, intent, result)
    
    if not intent["metric"] or not intent["group_by"]:
        result["answer"] = f"Could not parse query. Detected: metric={intent['metric']}, group_by={intent['group_by']}"
        return result
    
    # TRY OPERATIONAL INSIGHTS FIRST (same as frontend)
    insights = get_operational_insights_from_csv()
    
    if insights and intent["group_by"] == "manager" and "by_manager" in insights:
        result["method"] = "operational_insights_api"
        managers = insights["by_manager"]
        
        # Filter out non-manager entities
        excluded_entities = ["board of directors", "board", "directors", "executive office", "executive"]
        
        if intent["metric"] == "engagement":
            # Extract engagement values
            entity_averages = {}
            for mgr in managers:
                mgr_name = mgr.get("manager", "").lower().strip()
                # Skip if it's not a real manager name
                if any(excluded in mgr_name for excluded in excluded_entities):
                    continue
                if "manager" in mgr and "avg_engagement" in mgr and mgr["avg_engagement"] is not None:
                    entity_averages[mgr["manager"]] = mgr["avg_engagement"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    if len(best_entities) == 1:
                        result["answer"] = f"{best_entities[0]} has the highest average engagement of {best_value:.2f}"
                    else:
                        result["answer"] = f"Multiple managers have the highest average engagement of {best_value:.2f}:\n"
                        for entity in best_entities:
                            result["answer"] += f"  • {entity}\n"
                elif intent["operation"] == "min":
                    worst_value = min(entity_averages.values())
                    worst_entities = [e for e, v in entity_averages.items() if v == worst_value]
                    if len(worst_entities) == 1:
                        result["answer"] = f"{worst_entities[0]} has the lowest average engagement of {worst_value:.2f}"
                    else:
                        result["answer"] = f"Multiple managers have the lowest average engagement of {worst_value:.2f}:\n"
                        for entity in worst_entities:
                            result["answer"] += f"  • {entity}\n"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average engagement by manager:"]
                    for entity, avg in sorted_entities[:15]:
                        answer_parts.append(f"  • {entity}: {avg:.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
        
        elif intent["metric"] == "performance":
            entity_averages = {}
            for mgr in managers:
                mgr_name = mgr.get("manager", "").lower().strip()
                # Skip if it's not a real manager name
                if any(excluded in mgr_name for excluded in excluded_entities):
                    continue
                if "manager" in mgr and "avg_performance_score" in mgr and mgr["avg_performance_score"] is not None:
                    entity_averages[mgr["manager"]] = mgr["avg_performance_score"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average performance of {best_value:.2f}" if len(best_entities) == 1 else f"Multiple managers tie at {best_value:.2f}"
                elif intent["operation"] == "min":
                    worst_value = min(entity_averages.values())
                    worst_entities = [e for e, v in entity_averages.items() if v == worst_value]
                    result["answer"] = f"{worst_entities[0]} has the lowest average performance of {worst_value:.2f}" if len(worst_entities) == 1 else f"Multiple managers tie at {worst_value:.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average performance by manager:"]
                    for entity, avg in sorted_entities[:15]:
                        answer_parts.append(f"  • {entity}: {avg:.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
        
        elif intent["metric"] == "salary":
            entity_averages = {}
            for mgr in managers:
                mgr_name = mgr.get("manager", "").lower().strip()
                # Skip if it's not a real manager name
                if any(excluded in mgr_name for excluded in excluded_entities):
                    continue
                if "manager" in mgr and "avg_salary" in mgr and mgr["avg_salary"] is not None:
                    entity_averages[mgr["manager"]] = mgr["avg_salary"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average salary of ${best_value:,.2f}" if len(best_entities) == 1 else f"Multiple managers tie at ${best_value:,.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average salary by manager:"]
                    for entity, avg in sorted_entities[:15]:
                        answer_parts.append(f"  • {entity}: ${avg:,.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
    
    elif insights and intent["group_by"] == "department" and "by_department" in insights:
        result["method"] = "operational_insights_api"
        departments = insights["by_department"]
        
        if intent["metric"] == "engagement":
            entity_averages = {}
            for dept in departments:
                if "department" in dept and "avg_engagement" in dept and dept["avg_engagement"] is not None:
                    entity_averages[dept["department"]] = dept["avg_engagement"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average engagement of {best_value:.2f}" if len(best_entities) == 1 else f"Multiple departments tie at {best_value:.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average engagement by department:"]
                    for entity, avg in sorted_entities:
                        answer_parts.append(f"  • {entity}: {avg:.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
        
        elif intent["metric"] == "performance":
            entity_averages = {}
            for dept in departments:
                if "department" in dept and "avg_performance_score" in dept and dept["avg_performance_score"] is not None:
                    entity_averages[dept["department"]] = dept["avg_performance_score"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average performance of {best_value:.2f}" if len(best_entities) == 1 else f"Multiple departments tie at {best_value:.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average performance by department:"]
                    for entity, avg in sorted_entities:
                        answer_parts.append(f"  • {entity}: {avg:.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
        
        elif intent["metric"] == "salary":
            entity_averages = {}
            for dept in departments:
                if "department" in dept and "avg_salary" in dept and dept["avg_salary"] is not None:
                    entity_averages[dept["department"]] = dept["avg_salary"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average salary of ${best_value:,.2f}" if len(best_entities) == 1 else f"Multiple departments tie at ${best_value:,.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average salary by department:"]
                    for entity, avg in sorted_entities:
                        answer_parts.append(f"  • {entity}: ${avg:,.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
        
        elif intent["metric"] == "special_projects":
            entity_averages = {}
            for dept in departments:
                if "department" in dept and "avg_special_projects" in dept and dept["avg_special_projects"] is not None:
                    entity_averages[dept["department"]] = dept["avg_special_projects"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average special projects count of {best_value:.2f}" if len(best_entities) == 1 else f"Multiple departments tie at {best_value:.2f}"
                elif intent["operation"] == "min":
                    worst_value = min(entity_averages.values())
                    worst_entities = [e for e, v in entity_averages.items() if v == worst_value]
                    result["answer"] = f"{worst_entities[0]} has the lowest average special projects count of {worst_value:.2f}" if len(worst_entities) == 1 else f"Multiple departments tie at {worst_value:.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average special projects count by department:"]
                    for entity, avg in sorted_entities:
                        answer_parts.append(f"  • {entity}: {avg:.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
    
    elif insights and intent["group_by"] == "recruitment" and "by_recruitment_source" in insights:
        result["method"] = "operational_insights_api"
        sources = insights["by_recruitment_source"]
        
        if intent["metric"] == "performance":
            entity_averages = {}
            for source in sources:
                if "recruitment_source" in source and "avg_performance_score" in source and source["avg_performance_score"] is not None:
                    entity_averages[source["recruitment_source"]] = source["avg_performance_score"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average performance of {best_value:.2f}" if len(best_entities) == 1 else f"Multiple recruitment sources tie at {best_value:.2f}"
                elif intent["operation"] == "min":
                    worst_value = min(entity_averages.values())
                    worst_entities = [e for e, v in entity_averages.items() if v == worst_value]
                    result["answer"] = f"{worst_entities[0]} has the lowest average performance of {worst_value:.2f}" if len(worst_entities) == 1 else f"Multiple recruitment sources tie at {worst_value:.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Distribution of performance by recruitment:"]
                    for entity, avg in sorted_entities:
                        answer_parts.append(f"  • {entity}: {avg:.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
        
        elif intent["metric"] == "salary":
            entity_averages = {}
            for source in sources:
                if "recruitment_source" in source and "avg_salary" in source and source["avg_salary"] is not None:
                    entity_averages[source["recruitment_source"]] = source["avg_salary"]
            
            if entity_averages:
                if intent["operation"] == "max":
                    best_value = max(entity_averages.values())
                    best_entities = [e for e, v in entity_averages.items() if v == best_value]
                    result["answer"] = f"{best_entities[0]} has the highest average salary of ${best_value:,.2f}" if len(best_entities) == 1 else f"Multiple recruitment sources tie at ${best_value:,.2f}"
                else:
                    sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
                    answer_parts = [f"Average salary by recruitment source:"]
                    for entity, avg in sorted_entities:
                        answer_parts.append(f"  • {entity}: ${avg:,.2f}")
                    result["answer"] = "\n".join(answer_parts)
                # Retrieve facts for evidence
                facts = search_facts_for_answer(intent, limit=50)
                result["facts_used"] = facts[:50]
                return result
    
    # FALLBACK: Search knowledge graph (old method)
    result["method"] = "knowledge_graph_search"
    facts = search_facts_for_answer(intent, limit=200)
    
    if not facts:
        result["answer"] = "No relevant facts found in knowledge graph."
        return result
    
    # Filter to operational insights if available
    operational_facts = [f for f in facts if f["is_operational"]]
    if operational_facts:
        facts = operational_facts[:50]  # Use operational insights
    
    result["facts_used"] = facts[:10]  # Store first 10 for reference
    
    # Extract values and entities
    entity_values = defaultdict(list)  # entity -> [values]
    
    for fact in facts:
        entity = extract_entity_from_fact(fact, intent["group_by"])
        value = extract_value_from_fact(fact)
        
        if entity and value is not None:
            # Clean entity name (remove extra words, normalize)
            entity_clean = entity.strip()
            # Remove common prefixes/suffixes
            entity_clean = re.sub(r'^(manager|department|dept)\s+', '', entity_clean, flags=re.IGNORECASE)
            entity_clean = re.sub(r'\s+(department|dept)$', '', entity_clean, flags=re.IGNORECASE)
            entity_clean = entity_clean.strip()
            
            if entity_clean:
                entity_values[entity_clean].append(value)
    
    if not entity_values:
        result["answer"] = "Could not extract values from facts. Try a more specific query."
        return result
    
    # Calculate averages for each entity
    entity_averages = {}
    for entity, values in entity_values.items():
        if values:
            entity_averages[entity] = sum(values) / len(values)
    
    if not entity_averages:
        result["answer"] = "Could not calculate averages from extracted values."
        return result
    
    # Answer based on operation
    if intent["operation"] == "max":
        best_value = max(entity_averages.values())
        best_entities = [e for e, v in entity_averages.items() if v == best_value]
        
        if len(best_entities) == 1:
            result["answer"] = f"{best_entities[0]} has the highest average {intent['metric']} of {best_value:.2f}"
        else:
            result["answer"] = f"Multiple {intent['group_by']}s have the highest average {intent['metric']} of {best_value:.2f}:\n"
            for entity in best_entities:
                result["answer"] += f"  • {entity}\n"
            result["answer"] += f"\nAll have the same average value of {best_value:.2f}"
        result["method"] = "max_comparison"
    
    elif intent["operation"] == "min":
        worst_value = min(entity_averages.values())
        worst_entities = [e for e, v in entity_averages.items() if v == worst_value]
        
        if len(worst_entities) == 1:
            result["answer"] = f"{worst_entities[0]} has the lowest average {intent['metric']} of {worst_value:.2f}"
        else:
            result["answer"] = f"Multiple {intent['group_by']}s have the lowest average {intent['metric']} of {worst_value:.2f}:\n"
            for entity in worst_entities:
                result["answer"] += f"  • {entity}\n"
            result["answer"] += f"\nAll have the same average value of {worst_value:.2f}"
        result["method"] = "min_comparison"
    
    elif intent["operation"] == "average" or intent["operation"] is None:
        # Show all averages (deduplicated)
        sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
        answer_parts = [f"Average {intent['metric']} by {intent['group_by']}:"]
        seen_entities = set()
        for entity, avg in sorted_entities:
            # Deduplicate (same entity might appear multiple times)
            entity_key = entity.lower().strip()
            if entity_key not in seen_entities:
                answer_parts.append(f"  • {entity}: {avg:.2f}")
                seen_entities.add(entity_key)
                if len(seen_entities) >= 15:  # Limit to 15 unique entities
                    break
        result["answer"] = "\n".join(answer_parts)
        result["method"] = "average_listing"
    
    elif intent["operation"] == "distribution":
        sorted_entities = sorted(entity_averages.items(), key=lambda x: x[1], reverse=True)
        answer_parts = [f"Distribution of {intent['metric']} by {intent['group_by']}:"]
        for entity, avg in sorted_entities:
            answer_parts.append(f"  • {entity}: {avg:.2f}")
        result["answer"] = "\n".join(answer_parts)
        result["method"] = "distribution"
    
    return result


def handle_fact_based_query(query: str, intent: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle fact-based queries about employees.
    Examples:
    - "Retrieve facts related with employee Becker, Scott"
    - "Give me facts about the employee with the highest salary"
    - "Give me facts about the employee with the lowest performance"
    """
    try:
        from query_processor import extract_single_employee_facts
        from operational_queries import get_top_salary, get_bottom_performance, load_csv_data
        import os
        
        operation = intent.get("operation")
        employee_name = intent.get("employee_name")
        
        facts = []
        answer_parts = []
        
        # Case 1: Specific employee name provided OR department-specific facts
        if employee_name and operation == "fact_retrieval":
            result["method"] = "employee_fact_retrieval"
            
            # Try to extract facts using query_processor first
            employee_data = extract_single_employee_facts(employee_name)
            
            # Also search knowledge graph directly for facts containing this employee name
            # ONLY from Document Agent (source_document should be the CSV file, not operational_insights)
            emp_facts_from_kg = []
            if KG_AVAILABLE and graph:
                    from knowledge import get_fact_source_document
                    from urllib.parse import unquote
                    
                    emp_name_lower = employee_name.lower()
                    emp_name_parts = employee_name.split(',')
                    last_name = emp_name_parts[0].strip().lower() if len(emp_name_parts) > 0 else ""
                    first_name = emp_name_parts[1].strip().lower() if len(emp_name_parts) > 1 else ""
                    
                    for s, p, o in graph:
                        predicate_str = str(p)
                        if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                                            'has_details', 'source_document', 'uploaded_at',
                                                            'is_inferred', 'confidence', 'agent_id']):
                            continue
                        
                        # Check source document - ONLY include Document Agent facts (not operational_insights)
                        sources = get_fact_source_document(
                            unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' '),
                            unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' '),
                            str(o)
                        )
                        
                        # Only include facts from Document Agent (CSV file), exclude operational_insights
                        is_operational = False
                        for src, _ in sources:
                            if 'operational_insights' in str(src).lower():
                                is_operational = True
                                break
                        
                        if is_operational:
                            continue
                        
                        subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
                        predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
                        obj = str(o)
                        
                        fact_text = f"{subject} {predicate} {obj}".lower()
                        
                        # STRICT FILTER: Only include facts where the employee name appears in subject or object
                        # This ensures we only get facts ABOUT this employee, not just mentioning them
                        subject_lower = subject.lower()
                        obj_lower = obj.lower()
                        
                        # Check if employee name appears in subject (most common case)
                        name_in_subject = (emp_name_lower in subject_lower or 
                                         (last_name and last_name in subject_lower and first_name and first_name in subject_lower))
                        
                        # Check if employee name appears in object (less common but valid)
                        name_in_object = (emp_name_lower in obj_lower or
                                        (last_name and last_name in obj_lower and first_name and first_name in obj_lower))
                        
                        # Only include if name is in subject OR object (not just anywhere in the fact)
                        if name_in_subject or name_in_object:
                            emp_facts_from_kg.append({
                                "fact_text": f"{subject} {predicate} {obj}",
                                "subject": subject,
                                "predicate": predicate,
                                "object": obj,
                                "source": "document_agent"
                            })
            
            # Combine results - ONLY use Document Agent facts
            # Filter employee_data facts to only Document Agent
            if employee_data and len(employee_data) > 0:
                emp = employee_data[0]
                
                # Add facts from employee data - but filter to only Document Agent
                if "facts" in emp and emp["facts"]:
                    for fact in emp["facts"]:
                        fact_text = fact.get("fact_text", f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}")
                        # Only include if it mentions the employee AND is from Document Agent
                        if employee_name.lower() in fact_text.lower() or employee_name in fact_text:
                            # Check if this fact is from Document Agent (not operational insights)
                            if fact.get("source") == "document_agent" or "operational_insights" not in str(fact.get("source", "")).lower():
                                emp_facts_from_kg.append(fact)
            
            # Add facts from direct KG search
            if emp_facts_from_kg:
                # Remove duplicates
                seen = set()
                unique_facts = []
                for fact in emp_facts_from_kg:
                    fact_text = fact.get("fact_text", f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}")
                    if fact_text not in seen:
                        seen.add(fact_text)
                        unique_facts.append(fact)
                
                if unique_facts:
                    answer_parts.append(f"\nRelated facts ({len(unique_facts)}):")
                    for i, fact in enumerate(unique_facts[:50], 1):  # Limit to 50 facts
                        fact_text = fact.get("fact_text", f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}")
                        answer_parts.append(f"  {i}. {fact_text}")
                        facts.append(fact)
                
                result["answer"] = "\n".join(answer_parts)
                result["facts_used"] = facts[:50]
            else:
                result["answer"] = f"No facts found for employee {employee_name}"
        
        # Case 1b: Department-specific fact retrieval OR employees in department
        elif intent.get("department_name") and (operation == "fact_retrieval" or operation == "department_employees_facts"):
            result["method"] = "department_fact_retrieval"
            dept_name = intent.get("department_name")
            
            # Map department names
            dept_mapping = {
                "production": "Production",
                "sales": "Sales",
                "it/is": "IT/IS",
                "admin": "Admin Offices"
            }
            actual_dept_name = dept_mapping.get(dept_name, dept_name.title())
            
                # Search for facts about this department - ONLY facts that specifically mention this department
            if KG_AVAILABLE and graph:
                from knowledge import get_fact_source_document
                from urllib.parse import unquote
                
                dept_facts = []
                dept_name_lower = actual_dept_name.lower()
                # Handle variations
                dept_variations = {
                    "production": ["production", "prod"],
                    "sales": ["sales"],
                    "it/is": ["it/is", "it", "is"],
                    "admin offices": ["admin offices", "admin"]
                }
                dept_keywords = dept_variations.get(dept_name_lower, [dept_name_lower])
                
                for s, p, o in graph:
                    predicate_str = str(p)
                    if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                                        'has_details', 'source_document', 'uploaded_at',
                                                        'is_inferred', 'confidence', 'agent_id']):
                        continue
                    
                    subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
                    predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
                    obj = str(o)
                    
                    fact_text = f"{subject} {predicate} {obj}".lower()
                    
                    # Only include facts that mention THIS specific department (not others)
                    # Check if fact contains this department name AND doesn't contain other department names
                    other_depts = ["production", "sales", "it/is", "admin", "software engineering"]
                    other_depts = [d for d in other_depts if d not in dept_keywords]
                    
                    matches_this_dept = any(keyword in fact_text for keyword in dept_keywords)
                    matches_other_dept = any(other in fact_text for other in other_depts)
                    
                    # Also check source - prefer Document Agent facts, but allow operational if they're department-specific
                    sources = get_fact_source_document(subject, predicate, obj)
                    is_operational = any('operational_insights' in str(src).lower() for src, _ in sources)
                    
                    # For department queries, only include facts that specifically mention this department
                    # Exclude operational insights that are about other departments
                    if matches_this_dept and not matches_other_dept:
                        # If it's operational insight, make sure it's about THIS department
                        if is_operational:
                            # Check if the fact is specifically about this department
                            if any(f"{keyword} department" in fact_text or f"department {keyword}" in fact_text for keyword in dept_keywords):
                                dept_facts.append({
                                    "fact_text": f"{subject} {predicate} {obj}",
                                    "subject": subject,
                                    "predicate": predicate,
                                    "object": obj
                                })
                        else:
                            # Document Agent facts - include if they mention this department
                            dept_facts.append({
                                "fact_text": f"{subject} {predicate} {obj}",
                                "subject": subject,
                                "predicate": predicate,
                                "object": obj
                            })
                
                if dept_facts:
                    if operation == "department_employees_facts":
                        answer_parts.append(f"Facts about employees in {actual_dept_name} department ({len(dept_facts)} facts):")
                    else:
                        answer_parts.append(f"Facts about {actual_dept_name} department ({len(dept_facts)} facts):")
                    for i, fact in enumerate(dept_facts[:50], 1):
                        answer_parts.append(f"  {i}. {fact['fact_text']}")
                        facts.append(fact)
                    result["answer"] = "\n".join(answer_parts)
                    result["facts_used"] = facts[:50]
                else:
                    result["answer"] = f"No facts found specifically about {actual_dept_name} department."
            else:
                result["answer"] = "Knowledge graph not available."
        
        # Case 2: Highest salary employee
        elif operation == "fact_retrieval_max" and ("salary" in query.lower() or "paid" in query.lower() or "highest paid" in query.lower()):
            result["method"] = "highest_salary_facts"
            
            # Get highest salary employee from operational insights
            csv_paths = [
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HR_S.csv"),
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HRDataset_v14.csv"),
            ]
            
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path:
                df = load_csv_data(csv_path)
                if df is not None and len(df) > 0:
                    top_salary_list = get_top_salary(df, top_n=1)
                    if top_salary_list and len(top_salary_list) > 0:
                        top_emp = top_salary_list[0]
                        emp_name = top_emp.get("employee_name", "Unknown")
                        salary = top_emp.get("salary", 0)
                        
                        # Now get facts for this employee
                        employee_data = extract_single_employee_facts(emp_name)
                        if employee_data and len(employee_data) > 0:
                            emp = employee_data[0]
                            answer_parts.append(f"Employee with highest salary: {emp_name} (${salary:,.2f})")
                            answer_parts.append(f"\nFacts about {emp_name}:")
                            
                            if "attributes" in emp:
                                attrs = emp["attributes"]
                                for key, value in attrs.items():
                                    if value is not None:
                                        answer_parts.append(f"  • {key.replace('_', ' ').title()}: {value}")
                            
                            if "facts" in emp and emp["facts"]:
                                for i, fact in enumerate(emp["facts"][:20], 1):
                                    fact_text = fact.get("fact_text", f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}")
                                    answer_parts.append(f"  {i}. {fact_text}")
                                    facts.append(fact)
                            
                            result["answer"] = "\n".join(answer_parts)
                            result["facts_used"] = facts[:50]
                        else:
                            result["answer"] = f"Employee with highest salary: {emp_name} (${salary:,.2f}), but no additional facts found in knowledge graph."
                    else:
                        result["answer"] = "Could not determine employee with highest salary."
                else:
                    result["answer"] = "Could not load employee data."
            else:
                result["answer"] = "Could not find employee data file."
        
        # Case 3: Lowest performance employee
        elif operation == "fact_retrieval_min" and ("performance" in query.lower() or "worst" in query.lower()):
            result["method"] = "lowest_performance_facts"
            
            # Get lowest performance employee from operational insights
            csv_paths = [
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HR_S.csv"),
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HRDataset_v14.csv"),
            ]
            
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path:
                df = load_csv_data(csv_path)
                if df is not None and len(df) > 0:
                    bottom_perf_list = get_bottom_performance(df, bottom_n=1)
                    if bottom_perf_list and len(bottom_perf_list) > 0:
                        bottom_emp = bottom_perf_list[0]
                        emp_name = bottom_emp.get("employee_name", "Unknown")
                        perf = bottom_emp.get("performance_score", "Unknown")
                        
                        # Now get facts for this employee - ONLY Document Agent facts
                        # Search knowledge graph directly for facts about this employee
                        if KG_AVAILABLE and graph:
                            from knowledge import get_fact_source_document
                            from urllib.parse import unquote
                            
                            emp_name_lower = emp_name.lower()
                            emp_facts = []
                            
                            for s, p, o in graph:
                                predicate_str = str(p)
                                if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                                                    'has_details', 'source_document', 'uploaded_at',
                                                                    'is_inferred', 'confidence', 'agent_id']):
                                    continue
                                
                                # Check source - ONLY Document Agent facts
                                sources = get_fact_source_document(
                                    unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' '),
                                    unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' '),
                                    str(o)
                                )
                                
                                is_operational = False
                                for src, _ in sources:
                                    if 'operational_insights' in str(src).lower():
                                        is_operational = True
                                        break
                                
                                if is_operational:
                                    continue
                                
                                subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
                                predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
                                obj = str(o)
                                
                                fact_text = f"{subject} {predicate} {obj}".lower()
                                
                                # Only facts about this employee
                                if emp_name_lower in fact_text or emp_name in f"{subject} {predicate} {obj}":
                                    emp_facts.append({
                                        "fact_text": f"{subject} {predicate} {obj}",
                                        "subject": subject,
                                        "predicate": predicate,
                                        "object": obj
                                    })
                            
                            if emp_facts:
                                answer_parts.append(f"Employee with lowest performance: {emp_name} (Performance: {perf})")
                                answer_parts.append(f"\nFacts about {emp_name} ({len(emp_facts)} facts):")
                                for i, fact in enumerate(emp_facts[:50], 1):
                                    answer_parts.append(f"  {i}. {fact['fact_text']}")
                                    facts.append(fact)
                                result["answer"] = "\n".join(answer_parts)
                                result["facts_used"] = facts[:50]
                            else:
                                # Try alternative name formats
                                name_variations = [emp_name, emp_name.replace(',', ', '), emp_name.replace('  ', ' ')]
                                for name_var in name_variations:
                                    name_lower = name_var.lower()
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
                                        
                                        is_operational = any('operational_insights' in str(src).lower() for src, _ in sources)
                                        if is_operational:
                                            continue
                                        
                                        subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
                                        predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
                                        obj = str(o)
                                        
                                        fact_text = f"{subject} {predicate} {obj}".lower()
                                        if name_lower in fact_text:
                                            emp_facts.append({
                                                "fact_text": f"{subject} {predicate} {obj}",
                                                "subject": subject,
                                                "predicate": predicate,
                                                "object": obj
                                            })
                                
                                if emp_facts:
                                    answer_parts = [f"Employee with lowest performance: {emp_name} (Performance: {perf})"]
                                    answer_parts.append(f"\nFacts about {emp_name} ({len(emp_facts)} facts):")
                                    for i, fact in enumerate(emp_facts[:50], 1):
                                        answer_parts.append(f"  {i}. {fact['fact_text']}")
                                        facts.append(fact)
                                    result["answer"] = "\n".join(answer_parts)
                                    result["facts_used"] = facts[:50]
                                else:
                                    result["answer"] = f"Employee with lowest performance: {emp_name} (Performance: {perf}), but no facts found in knowledge graph."
                        else:
                            result["answer"] = f"Employee with lowest performance: {emp_name} (Performance: {perf}), but knowledge graph not available."
                    else:
                        result["answer"] = "Could not determine employee with lowest performance."
                else:
                    result["answer"] = "Could not load employee data."
            else:
                result["answer"] = "Could not find employee data file."
        
        else:
            result["answer"] = "Could not parse fact-based query. Please specify an employee name or ask for highest/lowest employee."
        
        return result
        
    except Exception as e:
        result["answer"] = f"Error retrieving facts: {str(e)}"
        import traceback
        traceback.print_exc()
        return result


def handle_strategic_query(query: str, intent: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle strategic queries that find employees matching multiple conditions.
    Examples:
    - "Identify employees with high performance, low engagement and many special projects"
    - "Find employees with high performance, low engagement and low satisfaction"
    """
    try:
        from operational_queries import load_csv_data, normalize_column_name
        from strategic_queries import find_csv_file_path
        from knowledge import graph, load_knowledge_graph
        from urllib.parse import unquote
        import pandas as pd
        import os
        
        # Ensure KG is loaded
        if graph is None:
            load_knowledge_graph()
        
        result["method"] = "strategic_employee_search"
        
        # Load CSV data - use same logic as other functions
        csv_path = find_csv_file_path()
        if not csv_path:
            # Fallback to direct paths
            csv_paths = [
                "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv",
                "/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv",
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HRDataset_v14.csv"),
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HR_S.csv"),
            ]
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        if not csv_path or not os.path.exists(csv_path):
            result["answer"] = "Could not find employee data file."
            return result
        
        df = load_csv_data(csv_path)
        if df is None or len(df) == 0:
            result["answer"] = "Could not load employee data."
            return result
        
        # Convert PerformanceScore to numeric if needed
        perf_col = normalize_column_name(df, "PerformanceScore")
        if perf_col and perf_col in df.columns:
            perf_map = {'Exceeds': 4, 'Fully Meets': 3, 'Needs Improvement': 2, 'PIP': 1}
            if df[perf_col].dtype == 'object':
                df['_PerfNumeric'] = df[perf_col].map(perf_map)
            else:
                df['_PerfNumeric'] = df[perf_col]
        else:
            # If no performance column, create dummy numeric column
            df['_PerfNumeric'] = 3.0
        
        # Apply filters based on conditions
        conditions = intent.get("conditions", [])
        filtered_df = df.copy()
        
        for condition in conditions:
            metric = condition.get("metric")
            operator = condition.get("operator")
            
            if metric == "performance" and operator == "high":
                if '_PerfNumeric' in filtered_df.columns:
                    # High performance: >= 3.0 (Fully Meets or Exceeds)
                    filtered_df = filtered_df[filtered_df['_PerfNumeric'] >= 3.0]
            
            elif metric == "engagement" and operator == "low":
                eng_col = normalize_column_name(df, "EngagementSurvey")
                if eng_col and eng_col in filtered_df.columns:
                    # Low engagement: < 4.0 (below average) - more lenient threshold
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[eng_col], errors='coerce') < 4.0]
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[eng_col], errors='coerce').notna()]
            
            elif metric == "satisfaction" and operator == "low":
                sat_col = normalize_column_name(df, "EmpSatisfaction")
                if sat_col and sat_col in filtered_df.columns:
                    # Low satisfaction: < 4.0 (below average) - more lenient threshold
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[sat_col], errors='coerce') < 4.0]
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[sat_col], errors='coerce').notna()]
            
            elif metric == "special_projects" and operator == "high":
                sp_col = normalize_column_name(df, "SpecialProjectsCount")
                if sp_col and sp_col in filtered_df.columns:
                    # Many special projects: >= 2 (more lenient)
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[sp_col], errors='coerce') >= 2]
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[sp_col], errors='coerce').notna()]
            
            elif metric == "absences" and operator == "high":
                abs_col = normalize_column_name(df, "Absences")
                if abs_col and abs_col in filtered_df.columns:
                    # Many absences: >= 5 (more lenient)
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[abs_col], errors='coerce') >= 5]
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[abs_col], errors='coerce').notna()]
        
        # Get employee names
        emp_name_col = normalize_column_name(df, "Employee_Name")
        if not emp_name_col or emp_name_col not in filtered_df.columns:
            # Try alternative column names
            for col in df.columns:
                if 'name' in col.lower() and 'employee' in col.lower():
                    emp_name_col = col
                    break
        
        if emp_name_col and emp_name_col in filtered_df.columns:
            employees = filtered_df[emp_name_col].dropna().unique().tolist()
            
            if len(employees) > 0:
                answer_parts = [f"Found {len(employees)} employee(s) matching the criteria:"]
                facts = []
                
                # Retrieve facts from KG for each employee
                if graph and len(graph) > 0:
                    from knowledge import get_fact_source_document
                    
                    for i, emp_name in enumerate(employees[:20], 1):  # Limit to 20
                        answer_parts.append(f"  {i}. {emp_name}")
                        
                        # Search KG for facts about this employee
                        emp_name_parts = emp_name.split(', ')
                        if len(emp_name_parts) == 2:
                            last_name, first_name = emp_name_parts[0].strip(), emp_name_parts[1].strip()
                            emp_name_lower = emp_name.lower()
                            
                            # Search KG for facts about this employee
                            emp_facts = []
                            for s, p, o in graph:
                                predicate_str = str(p)
                                if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                                                    'has_details', 'source_document', 'uploaded_at',
                                                                    'is_inferred', 'confidence', 'agent_id']):
                                    continue
                                
                                subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
                                predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
                                obj = str(o)
                                
                                fact_text = f"{subject} {predicate} {obj}".lower()
                                
                                # Check if fact mentions this employee
                                name_in_subject = (emp_name_lower in subject.lower() or 
                                                 (last_name.lower() in subject.lower() and first_name.lower() in subject.lower()))
                                name_in_object = (emp_name_lower in obj.lower() or
                                                (last_name.lower() in obj.lower() and first_name.lower() in obj.lower()))
                                
                                if name_in_subject or name_in_object:
                                    # Check source - prefer document_agent
                                    sources = get_fact_source_document(subject, predicate, obj)
                                    is_operational = any('operational_insights' in str(src).lower() for src, _ in sources)
                                    
                                    if not is_operational:  # Only document_agent facts
                                        emp_facts.append({
                                            "fact_text": f"{subject} {predicate} {obj}",
                                            "subject": subject,
                                            "predicate": predicate,
                                            "object": obj,
                                            "source": "document_agent"
                                        })
                            
                            # Add up to 5 facts per employee
                            for fact in emp_facts[:5]:
                                facts.append(fact)
                                answer_parts.append(f"      - {fact['fact_text']}")
                
                if len(employees) > 20:
                    answer_parts.append(f"  ... and {len(employees) - 20} more")
                
                result["answer"] = "\n".join(answer_parts)
                result["facts_used"] = facts[:100]  # Limit total facts
            else:
                result["answer"] = "No employees found matching the specified criteria."
        else:
            result["answer"] = f"Could not find employee names in data. Available columns: {list(df.columns)[:10]}"
        
        return result
        
    except Exception as e:
        result["answer"] = f"Error processing strategic query: {str(e)}"
        import traceback
        traceback.print_exc()
        return result


def handle_correlation_query(query: str, intent: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle correlation queries comparing departments by salary and performance.
    Examples:
    - "Which departments have high salaries but low performance?"
    - "Identify departments with low salary and high performance"
    """
    try:
        from operational_queries import load_csv_data, normalize_column_name, compute_operational_insights
        import pandas as pd
        import os
        
        result["method"] = "correlation_analysis"
        
        # Get operational insights - use same CSV finding logic
        from strategic_queries import find_csv_file_path
        from knowledge import graph, load_knowledge_graph
        from urllib.parse import unquote
        
        # Ensure KG is loaded
        if graph is None:
            load_knowledge_graph()
        
        csv_path = find_csv_file_path()
        if not csv_path:
            csv_paths = [
                "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv",
                "/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv",
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HRDataset_v14.csv"),
                os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HR_S.csv"),
            ]
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        if not csv_path or not os.path.exists(csv_path):
            result["answer"] = "Could not find employee data file."
            return result
        
        df = load_csv_data(csv_path)
        if df is None or len(df) == 0:
            result["answer"] = "Could not load employee data."
            return result
        
        insights = compute_operational_insights(df=df)
        
        if not insights or "by_department" not in insights:
            result["answer"] = "Could not load department insights."
            return result
        
        # Transform insights to list format (same as frontend)
        from answer_query_terminal import transform_insights_to_list_format
        insights_list = transform_insights_to_list_format(insights, df)
        departments = insights_list.get("by_department", [])
        
        operation = intent.get("operation")
        
        answer_parts = []
        facts = []
        matching_depts = []
        
        # Calculate average salary and performance for thresholds
        all_salaries = []
        all_perfs = []
        for d in departments:
            if isinstance(d, dict):
                if d.get("avg_salary"):
                    all_salaries.append(d.get("avg_salary", 0))
                if d.get("avg_performance_score"):
                    all_perfs.append(d.get("avg_performance_score", 0))
        avg_salary = sum(all_salaries) / len(all_salaries) if all_salaries else 70000
        avg_perf = sum(all_perfs) / len(all_perfs) if all_perfs else 3.0
        
        if operation == "high_salary_low_perf":
            # Find departments with high salary but low performance
            for dept in departments:
                if "department" in dept and "avg_salary" in dept and "avg_performance_score" in dept:
                    salary = dept.get("avg_salary", 0)
                    perf = dept.get("avg_performance_score", 0)
                    # High salary: above average, Low performance: below average
                    if salary and perf and salary > avg_salary and perf < avg_perf:
                        matching_depts.append((dept["department"], salary, perf))
            
            if matching_depts:
                answer_parts.append("Departments with high salaries but low performance:")
                for dept_name, salary, perf in sorted(matching_depts, key=lambda x: x[1], reverse=True):
                    answer_parts.append(f"  • {dept_name}: Salary ${salary:,.2f}, Performance {perf:.2f}")
            else:
                answer_parts.append("No departments found with high salaries but low performance.")
        
        elif operation == "low_salary_high_perf":
            # Find departments with low salary but high performance
            for dept in departments:
                if "department" in dept and "avg_salary" in dept and "avg_performance_score" in dept:
                    salary = dept.get("avg_salary", 0)
                    perf = dept.get("avg_performance_score", 0)
                    # Low salary: below average, High performance: above average
                    if salary and perf and salary < avg_salary and perf > avg_perf:
                        matching_depts.append((dept["department"], salary, perf))
            
            if matching_depts:
                answer_parts.append("Departments with low salary but high performance:")
                for dept_name, salary, perf in sorted(matching_depts, key=lambda x: x[2], reverse=True):
                    answer_parts.append(f"  • {dept_name}: Salary ${salary:,.2f}, Performance {perf:.2f}")
            else:
                answer_parts.append("No departments found with low salary but high performance.")
        
        elif operation == "correlation_analysis":
            # Analyze relationship between salary, performance, and department
            answer_parts.append("Department Analysis (Salary vs Performance):")
            dept_data = []
            for dept in departments:
                if "department" in dept and "avg_salary" in dept and "avg_performance_score" in dept:
                    salary = dept.get("avg_salary", 0)
                    perf = dept.get("avg_performance_score", 0)
                    if salary and perf:
                        dept_data.append((dept["department"], salary, perf))
            
            # Sort by performance
            dept_data.sort(key=lambda x: x[2], reverse=True)
            for dept_name, salary, perf in dept_data:
                answer_parts.append(f"  • {dept_name}: Salary ${salary:,.2f}, Performance {perf:.2f}")
        
        # Retrieve facts from KG for matching departments
        if graph and len(graph) > 0 and matching_depts:
            from knowledge import get_fact_source_document
            
            for dept_name, salary, perf in matching_depts[:5]:  # Limit to 5 departments
                dept_name_lower = dept_name.lower()
                dept_facts = []
                
                for s, p, o in graph:
                    predicate_str = str(p)
                    if any(x in predicate_str for x in ['fact_subject', 'fact_predicate', 'fact_object', 
                                                        'has_details', 'source_document', 'uploaded_at',
                                                        'is_inferred', 'confidence', 'agent_id']):
                        continue
                    
                    subject = unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' ')
                    predicate = unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' ')
                    obj = str(o)
                    
                    fact_text = f"{subject} {predicate} {obj}".lower()
                    
                    # Check if fact mentions this department
                    if dept_name_lower in fact_text or dept_name_lower.replace('/', ' ') in fact_text:
                        sources = get_fact_source_document(subject, predicate, obj)
                        is_operational = any('operational_insights' in str(src).lower() for src, _ in sources)
                        
                        if not is_operational:  # Only document_agent facts
                            dept_facts.append({
                                "fact_text": f"{subject} {predicate} {obj}",
                                "subject": subject,
                                "predicate": predicate,
                                "object": obj,
                                "source": "document_agent"
                            })
                
                # Add up to 3 facts per department
                for fact in dept_facts[:3]:
                    facts.append(fact)
                    answer_parts.append(f"      - {fact['fact_text']}")
        
        result["answer"] = "\n".join(answer_parts)
        result["facts_used"] = facts[:50]  # Limit total facts
        return result
        
    except Exception as e:
        result["answer"] = f"Error processing correlation query: {str(e)}"
        import traceback
        traceback.print_exc()
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Answer natural language queries using knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python answer_query_terminal.py "Which department has the highest average performance score?"
  python answer_query_terminal.py "What is the average engagement by manager?"
  python answer_query_terminal.py "Which manager has the highest team engagement?"
  python answer_query_terminal.py "Show me salary distribution by department"
        """
    )
    
    parser.add_argument("query", nargs="+", help="Query to answer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show facts used")
    
    args = parser.parse_args()
    
    query = " ".join(args.query)
    
    if not KG_AVAILABLE:
        print("❌ Knowledge graph not available")
        return
    
    # Load knowledge graph
    print("📂 Loading knowledge graph...")
    try:
        load_result = load_knowledge_graph()
        if load_result:
            print(f"✅ {load_result}")
    except Exception as e:
        print(f"⚠️  Error loading: {e}")
    
    graph_size = len(graph) if graph else 0
    if graph_size == 0:
        print("⚠️  Knowledge graph is empty")
        return
    
    print(f"📊 Knowledge graph: {graph_size} facts\n")
    
    # Answer query
    print(f"🔍 Query: {query}\n")
    result = answer_query(query)
    
    # Print answer
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])
    print()
    
    if args.verbose:
        print("=" * 80)
        print("DETAILS")
        print("=" * 80)
        print(f"Intent: {result['intent']}")
        print(f"Method: {result['method']}")
        print(f"Facts used: {len(result['facts_used'])}")
        if result['facts_used']:
            print("\nSample facts:")
            for i, fact in enumerate(result['facts_used'][:5], 1):
                print(f"  {i}. {fact['fact_text']}")
        print()


if __name__ == "__main__":
    main()

