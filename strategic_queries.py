"""
Strategic Query Processor - Multi-variable analytical queries
Handles complex queries involving 2-3 variable combinations for strategic and operational insights.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from agent_system import document_agents, agents_store, STATISTICS_AGENT_ID
import os


def detect_strategic_query(question: str) -> Dict[str, Any]:
    """
    Detect if a query is a strategic/operational multi-variable query.
    Returns query type and extracted variables.
    """
    question_lower = question.lower()
    
    # Strategic Query S1: Performance–Engagement–Status
    s1_patterns = [
        r'(?:performance|engagement|status|termination).*?(?:simultaneously|together|combine|cluster)',
        r'(?:high|low).*?(?:performance|engagement).*?(?:low|high).*?(?:engagement|performance).*?(?:termination|status)',
        r'(?:early.?warning|risk|retention).*?(?:cluster|segment|group)',
        r'(?:active|currently).*?(?:high|low).*?(?:performance|engagement).*?(?:declining|risk)',
    ]
    
    # Strategic Query S2: Recruitment–Performance–Retention
    s2_patterns = [
        r'(?:recruitment|source|channel).*?(?:performance|retention|status)',
        r'(?:rank|ranking).*?(?:recruitment|source|channel)',
        r'(?:underperforming|low).*?(?:recruitment|source).*?(?:turnover|performance)',
    ]
    
    # Operational Query O1: Performance × Department
    o1_patterns = [
        r'(?:department|departmental).*?(?:performance|monitoring|distribution)',
        r'(?:track|tracking).*?(?:low.?performance|needs.?improvement|pip).*?(?:department)',
    ]
    
    # Operational Query O2: Absences × EmploymentStatus
    o2_patterns = [
        r'(?:absence|absences).*?(?:employment|status|terminated|active)',
        r'(?:absence|absences).*?(?:pattern|distribution).*?(?:termination|status)',
    ]
    
    # Operational Query O3: Engagement × Manager
    # IMPORTANT: Make patterns more specific to avoid matching basic "who is the manager of X" queries
    o3_patterns = [
        r'(?:manager|team).*?(?:engagement|survey).*?(?:distribution|trend|low|declining|monitoring)',
        r'(?:team.?level|team-level).*?(?:engagement|survey).*?(?:monitoring|distribution|trend)',
        r'(?:engagement|survey).*?(?:by|per).*?(?:manager|team).*?(?:monitoring|distribution|trend)',
    ]
    
    # Operational Query O4: Average per group with min/max
    # "get the average performance score per manager and bring me the min"
    # "average over absences per department and return the department with max"
    o4_patterns = [
        r'(?:average|avg|mean).*?(?:performance|score|salary|absence|engagement).*?(?:per|by).*?(?:manager|department|role|position).*?(?:and|then|return).*?(?:min|max|minimum|maximum|lowest|highest)',
        r'(?:get|find|return).*?(?:the|a).*?(?:min|max|minimum|maximum|lowest|highest).*?(?:of|from).*?(?:average|avg|mean).*?(?:performance|score|salary|absence|engagement).*?(?:per|by).*?(?:manager|department|role|position)',
        r'(?:average|avg|mean).*?(?:over|of).*?(?:performance|score|salary|absence|engagement).*?(?:per|by).*?(?:manager|department|role|position).*?(?:and|then|return).*?(?:the|a).*?(?:department|manager|role|position).*?(?:with|having).*?(?:min|max|minimum|maximum|lowest|highest)',
    ]
    
    query_info = {
        "query_type": "general",
        "strategic_type": None,  # "S1", "S2", "O1", "O2", "O3", "O4"
        "subtype": None,  # "S1.1", "S1.2", etc.
        "variables": [],
        "operation": None,  # "cluster", "rank", "monitor", "track", "compare", "average_min", "average_max"
        "group_by": None,  # "manager", "department", "role", "position"
        "metric": None,  # "performance", "salary", "absence", "engagement"
        "aggregate_op": None,  # "min", "max"
    }
    
    # Check S1 patterns
    for pattern in s1_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "strategic"
            query_info["strategic_type"] = "S1"
            query_info["variables"] = ["PerformanceScore", "EngagementSurvey", "EmploymentStatus"]
            if "early-warning" in question_lower or "risk" in question_lower:
                query_info["subtype"] = "S1.1"
                query_info["operation"] = "cluster"
            elif "active" in question_lower and "declining" in question_lower:
                query_info["subtype"] = "S1.2"
                query_info["operation"] = "risk_detection"
            return query_info
    
    # Check S2 patterns
    for pattern in s2_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "strategic"
            query_info["strategic_type"] = "S2"
            query_info["variables"] = ["RecruitmentSource", "PerformanceScore", "EmploymentStatus"]
            if "rank" in question_lower:
                query_info["subtype"] = "S2.1"
                query_info["operation"] = "rank"
            elif "underperforming" in question_lower or "low" in question_lower:
                query_info["subtype"] = "S2.2"
                query_info["operation"] = "identify_underperformers"
            return query_info
    
    # Check O1 patterns
    for pattern in o1_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "strategic"
            query_info["strategic_type"] = "O1"
            query_info["variables"] = ["PerformanceScore", "Department"]
            if "monitor" in question_lower or "distribution" in question_lower:
                query_info["subtype"] = "O1.1"
                query_info["operation"] = "monitor"
            elif "track" in question_lower or "low" in question_lower:
                query_info["subtype"] = "O1.2"
                query_info["operation"] = "track"
            return query_info
    
    # Check O2 patterns
    for pattern in o2_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "strategic"
            query_info["strategic_type"] = "O2"
            query_info["variables"] = ["Absences", "EmploymentStatus"]
            query_info["subtype"] = "O2.1"
            query_info["operation"] = "compare"
            return query_info
    
    # Check O3 patterns
    for pattern in o3_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "strategic"
            query_info["strategic_type"] = "O3"
            query_info["variables"] = ["EngagementSurvey", "ManagerName"]
            query_info["subtype"] = "O3.1"
            query_info["operation"] = "monitor"
            return query_info
    
    # Check O4 patterns (average per group with min/max)
    for pattern in o4_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "strategic"
            query_info["strategic_type"] = "O4"
            
            # Extract metric (what to average)
            if "performance" in question_lower or "score" in question_lower:
                query_info["metric"] = "PerformanceScore"
            elif "absence" in question_lower:
                query_info["metric"] = "Absences"
            elif "salary" in question_lower:
                query_info["metric"] = "Salary"
            elif "engagement" in question_lower:
                query_info["metric"] = "EngagementSurvey"
            
            # Extract group_by (what to group by)
            if "manager" in question_lower:
                query_info["group_by"] = "ManagerName"
            elif "department" in question_lower:
                query_info["group_by"] = "Department"
            elif "role" in question_lower or "position" in question_lower:
                query_info["group_by"] = "Position"
            
            # Extract aggregate operation (min or max)
            if "min" in question_lower or "minimum" in question_lower or "lowest" in question_lower:
                query_info["aggregate_op"] = "min"
                query_info["operation"] = "average_min"
            elif "max" in question_lower or "maximum" in question_lower or "highest" in question_lower:
                query_info["aggregate_op"] = "max"
                query_info["operation"] = "average_max"
            
            query_info["variables"] = [query_info["metric"], query_info["group_by"]]
            query_info["subtype"] = "O4.1"
            return query_info
    
    return query_info


def find_csv_file_path() -> Optional[str]:
    """
    Find the file path of the most recently uploaded CSV file.
    Returns the file path if found, None otherwise.
    """
    # Check document agents for CSV files with stored file paths
    csv_agents = []
    for agent_id, agent in document_agents.items():
        if hasattr(agent, 'document_type') and agent.document_type.lower() == '.csv':
            file_path = getattr(agent, 'file_path', None)
            if file_path:
                if os.path.exists(file_path):
                    csv_agents.append((file_path, agent_id))
                else:
                    print(f"⚠️  Document agent {agent_id} has file_path but file doesn't exist: {file_path}")
    
    if csv_agents:
        # Return the most recent one (last in list)
        selected_path = csv_agents[-1][0]
        print(f"📂 Found CSV file from document agent: {selected_path}")
        return selected_path
    
    # Fallback: try to find the file in common locations
    from documents_store import get_all_documents
    
    try:
        documents = get_all_documents()
        csv_docs = [d for d in documents if d.get('type', '').lower() == 'csv']
        
        if csv_docs:
            # Try to find the file - check common locations
            doc_name = csv_docs[-1]['name']  # Most recent
            
            # Check if file exists in current directory or uploads
            possible_paths = [
                doc_name,
                f"uploads/{doc_name}",
                f"/tmp/{doc_name}",
                os.path.join(os.getcwd(), "uploads", doc_name),
                os.path.join(os.getcwd(), doc_name),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"📂 Found CSV file in fallback location: {path}")
                    return path
    except Exception as e:
        print(f"⚠️  Error checking documents store: {e}")
    
    print("⚠️  No CSV file found in document agents or common locations")
    return None


def load_csv_data(file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load CSV data from file path or find it from document agents.
    """
    if file_path is None:
        file_path = find_csv_file_path()
    
    if file_path is None or not os.path.exists(file_path):
        print(f"⚠️  CSV file not found: {file_path}")
        return None
    
    try:
        # Detect separator
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            tab_count = first_line.count('\t')
            
            if semicolon_count > comma_count and semicolon_count > 0:
                sep = ';'
            elif tab_count > comma_count and tab_count > 0:
                sep = '\t'
            else:
                sep = ','
        
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
        
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
        
        return df
    except Exception as e:
        print(f"⚠️  Error loading CSV: {e}")
        return None


def normalize_column_name(df: pd.DataFrame, target_name: str) -> Optional[str]:
    """
    Find the actual column name in the dataframe that matches the target name.
    Handles variations like "Performance Score" vs "PerformanceScore", "PerfScoreID", etc.
    """
    target_lower = target_name.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    # Handle common column name variations
    column_aliases = {
        'performancescore': ['performancescore', 'perfscoreid', 'perf_score', 'performance_score', 'perfscore'],
        'engagementsurvey': ['engagementsurvey', 'engagement_survey', 'engagement', 'engagementscore'],
        'employmentstatus': ['employmentstatus', 'employment_status', 'status', 'empstatus'],
        'department': ['department', 'dept', 'division'],
        'position': ['position', 'role', 'jobtitle', 'job_title'],
        'recruitmentsource': ['recruitmentsource', 'recruitment_source', 'source', 'recruitsource'],
        'absences': ['absences', 'absence', 'daysabsent', 'days_absent'],
        'managername': ['managername', 'manager_name', 'manager', 'supervisor'],
        'dateofhire': ['dateofhire', 'date_of_hire', 'hiredate', 'hire_date'],
        'specialprojectscount': ['specialprojectscount', 'special_projects_count', 'specialprojects', 'workload']
    }
    
    # Check if target matches any alias group
    for alias_group, variations in column_aliases.items():
        if target_lower in variations:
            # Try all variations in this group
            for variation in variations:
                for col in df.columns:
                    col_normalized = col.lower().replace(' ', '').replace('_', '').replace('-', '')
                    if variation in col_normalized or col_normalized in variation:
                        return col
    
    # Direct matching
    for col in df.columns:
        col_normalized = col.lower().replace(' ', '').replace('_', '').replace('-', '')
        if target_lower in col_normalized or col_normalized in target_lower:
            return col
    
    # Try partial matches
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in target_name.lower().split()):
            return col
    
    return None


def process_strategic_query(query_info: Dict[str, Any], question: str) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process a strategic query and return answer, evidence facts, and routing info.
    """
    strategic_type = query_info.get("strategic_type")
    subtype = query_info.get("subtype")
    operation = query_info.get("operation")
    variables = query_info.get("variables", [])
    
    # Load CSV data
    df = load_csv_data()
    if df is None:
        return (
            "I couldn't find the CSV dataset. Please ensure a CSV file has been uploaded.",
            [],
            {"strategy": "strategic_query", "reason": "CSV file not found"}
        )
    
    # Normalize column names - handle variations
    actual_columns = {}
    column_mapping = {
        "PerformanceScore": ["PerformanceScore", "PerfScoreID"],
        "EngagementSurvey": ["EngagementSurvey"],
        "EmploymentStatus": ["EmploymentStatus"],
        "Department": ["Department"],
        "Position": ["Position"],
        "RecruitmentSource": ["RecruitmentSource"],
        "Absences": ["Absences"],
        "ManagerName": ["ManagerName"],
        "DateofHire": ["DateofHire"],
        "SpecialProjectsCount": ["SpecialProjectsCount"]
    }
    
    for var in variables:
        # Try the primary name first
        actual_col = normalize_column_name(df, var)
        
        # If not found and we have a mapping, try alternatives
        if not actual_col and var in column_mapping:
            for alt_name in column_mapping[var]:
                actual_col = normalize_column_name(df, alt_name)
                if actual_col:
                    break
        
        if actual_col:
            actual_columns[var] = actual_col
        else:
            # Provide helpful error with available columns
            available_cols = ', '.join(df.columns[:15].tolist())
            return (
                f"I couldn't find the '{var}' column (or its variations) in the dataset. "
                f"Available columns: {available_cols}... "
                f"Please ensure the dataset contains columns matching: {', '.join(column_mapping.get(var, [var]))}",
                [],
                {"strategy": "strategic_query", "reason": f"Column '{var}' not found"}
            )
    
    answer_parts = []
    evidence_facts = []
    
    try:
        if strategic_type == "S1":
            # S1: Performance–Engagement–Status
            if subtype == "S1.1":
                answer_parts, evidence_facts = process_s1_1(df, actual_columns)
            elif subtype == "S1.2":
                answer_parts, evidence_facts = process_s1_2(df, actual_columns)
        
        elif strategic_type == "S2":
            # S2: Recruitment–Performance–Retention
            if subtype == "S2.1":
                answer_parts, evidence_facts = process_s2_1(df, actual_columns)
            elif subtype == "S2.2":
                answer_parts, evidence_facts = process_s2_2(df, actual_columns)
        
        elif strategic_type == "O1":
            # O1: Performance × Department
            if subtype == "O1.1":
                answer_parts, evidence_facts = process_o1_1(df, actual_columns)
            elif subtype == "O1.2":
                answer_parts, evidence_facts = process_o1_2(df, actual_columns)
        
        elif strategic_type == "O2":
            # O2: Absences × EmploymentStatus
            if subtype == "O2.1":
                answer_parts, evidence_facts = process_o2_1(df, actual_columns)
        
        elif strategic_type == "O3":
            # O3: Engagement × Manager
            if subtype == "O3.1":
                answer_parts, evidence_facts = process_o3_1(df, actual_columns)
        
        if answer_parts:
            answer = "\n\n".join(answer_parts)
            return answer, evidence_facts, {
                "strategy": "strategic_query",
                "reason": f"Processed {strategic_type} query using multi-variable analysis",
                "query_type": strategic_type,
                "subtype": subtype
            }
        else:
            return (
                "I processed the query but couldn't generate a complete answer. Please check if the required columns exist in the dataset.",
                evidence_facts,
                {"strategy": "strategic_query", "reason": "Incomplete analysis"}
            )
    
    except Exception as e:
        print(f"⚠️  Error processing strategic query: {e}")
        import traceback
        traceback.print_exc()
        return (
            f"I encountered an error processing this strategic query: {str(e)}",
            [],
            {"strategy": "strategic_query", "reason": f"Error: {str(e)}"}
        )


def process_s1_1(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    S1.1: Early-warning risk clusters
    Identify employee segments with high performance, low engagement, elevated termination rates.
    """
    perf_col = columns.get("PerformanceScore")
    eng_col = columns.get("EngagementSurvey")
    status_col = columns.get("EmploymentStatus")
    
    if not all([perf_col, eng_col, status_col]):
        return ["Missing required columns"], []
    
    # Ensure numeric columns are numeric
    if df[perf_col].dtype == 'object':
        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
    if df[eng_col].dtype == 'object':
        df[eng_col] = pd.to_numeric(df[eng_col], errors='coerce')
    
    # Remove rows with NaN in required columns
    df_clean = df.dropna(subset=[perf_col, eng_col])
    
    if len(df_clean) == 0:
        return ["No valid performance or engagement data found"], []
    
    # Define thresholds (adjust based on your data)
    perf_threshold = df_clean[perf_col].quantile(0.75)  # Top 25%
    eng_threshold = df_clean[eng_col].quantile(0.25)  # Bottom 25%
    
    # Filter for high performance, low engagement
    risk_df = df_clean[
        (df_clean[perf_col] >= perf_threshold) & 
        (df_clean[eng_col] <= eng_threshold)
    ].copy()
    
    if len(risk_df) == 0:
        return ["No employees found matching the high performance + low engagement criteria."], []
    
    # Analyze by department, role, tenure
    answer_parts = [
        f"## Early-Warning Risk Clusters Analysis\n\n"
        f"Found **{len(risk_df)} employees** ({len(risk_df)/len(df_clean)*100:.1f}% of workforce) "
        f"with high performance scores (≥{perf_threshold:.1f}) but low engagement (≤{eng_threshold:.1f})."
    ]
    
    # Analyze by department
    if "Department" in df_clean.columns:
        dept_analysis = risk_df.groupby("Department").agg({
            perf_col: ['count', 'mean'],
            eng_col: 'mean',
            status_col: lambda x: (x.str.contains('Terminated', case=False, na=False).sum() / len(x) * 100)
        }).round(2)
        dept_analysis.columns = ['Count', 'Avg_Performance', 'Avg_Engagement', 'Termination_Rate_%']
        dept_analysis = dept_analysis.sort_values('Count', ascending=False)
        
        answer_parts.append("### By Department:")
        for dept, row in dept_analysis.head(10).iterrows():
            answer_parts.append(
                f"- **{dept}**: {int(row['Count'])} employees, "
                f"Avg Performance: {row['Avg_Performance']:.1f}, "
                f"Avg Engagement: {row['Avg_Engagement']:.1f}, "
                f"Termination Rate: {row['Termination_Rate_%']:.1f}%"
            )
    
    # Analyze by role/position
    if "Position" in df.columns:
        role_analysis = risk_df.groupby("Position").size().sort_values(ascending=False)
        answer_parts.append("\n### By Role (Top 10):")
        for role, count in role_analysis.head(10).items():
            answer_parts.append(f"- **{role}**: {count} employees")
    
    # Termination analysis
    terminated = risk_df[risk_df[status_col].str.contains('Terminated', case=False, na=False)]
    if len(terminated) > 0:
        answer_parts.append(
            f"\n### Termination Status:\n"
            f"- **{len(terminated)} employees** ({len(terminated)/len(risk_df)*100:.1f}%) "
            f"in this risk cluster have been terminated."
        )
    
    evidence_facts = [
        {
            "subject": "Risk Cluster Analysis",
            "predicate": "has_count",
            "object": str(len(risk_df)),
            "source": ["strategic_query"]
        }
    ]
    
    return answer_parts, evidence_facts


def process_s1_2(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    S1.2: Active high performers at risk
    Detect active employees with high performance but declining engagement.
    """
    perf_col = columns.get("PerformanceScore")
    eng_col = columns.get("EngagementSurvey")
    status_col = columns.get("EmploymentStatus")
    
    if not all([perf_col, eng_col, status_col]):
        return ["Missing required columns"], []
    
    # Ensure numeric columns are numeric
    if df[perf_col].dtype == 'object':
        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
    if df[eng_col].dtype == 'object':
        df[eng_col] = pd.to_numeric(df[eng_col], errors='coerce')
    
    # Remove rows with NaN in required columns
    df_clean = df.dropna(subset=[perf_col, eng_col])
    
    # Filter for active employees only
    active_df = df_clean[df_clean[status_col].str.contains('Active', case=False, na=False)].copy()
    
    if len(active_df) == 0:
        return ["No active employees found in the dataset."], []
    
    # High performers (top 25%)
    perf_threshold = active_df[perf_col].quantile(0.75)
    high_performers = active_df[active_df[perf_col] >= perf_threshold]
    
    # Low engagement (bottom 25%)
    eng_threshold = active_df[eng_col].quantile(0.25)
    at_risk = high_performers[high_performers[eng_col] <= eng_threshold]
    
    if len(at_risk) == 0:
        return ["No active high performers with low engagement found."], []
    
    answer_parts = [
        f"## Active High Performers at Risk\n\n"
        f"Found **{len(at_risk)} active employees** ({len(at_risk)/len(active_df)*100:.1f}% of active workforce) "
        f"with high performance (≥{perf_threshold:.1f}) but low engagement (≤{eng_threshold:.1f})."
    ]
    
    # Characterize by common attributes
    if "Department" in df_clean.columns:
        dept_counts = at_risk["Department"].value_counts()
        answer_parts.append("\n### By Department:")
        for dept, count in dept_counts.head(10).items():
            answer_parts.append(f"- **{dept}**: {count} employees")
    
    if "ManagerName" in df_clean.columns:
        mgr_counts = at_risk["ManagerName"].value_counts()
        answer_parts.append("\n### By Manager (Top 10):")
        for mgr, count in mgr_counts.head(10).items():
            answer_parts.append(f"- **{mgr}**: {count} employees")
    
    if "Position" in df_clean.columns:
        role_counts = at_risk["Position"].value_counts()
        answer_parts.append("\n### By Role:")
        for role, count in role_counts.head(10).items():
            answer_parts.append(f"- **{role}**: {count} employees")
    
    evidence_facts = [
        {
            "subject": "At-Risk Active Employees",
            "predicate": "has_count",
            "object": str(len(at_risk)),
            "source": ["strategic_query"]
        }
    ]
    
    return answer_parts, evidence_facts


def process_s2_1(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    S2.1: Strategic ranking of recruitment channels
    Rank recruitment sources by performance and retention.
    """
    source_col = columns.get("RecruitmentSource")
    perf_col = columns.get("PerformanceScore")
    status_col = columns.get("EmploymentStatus")
    
    if not all([source_col, perf_col, status_col]):
        return ["Missing required columns"], []
    
    # Ensure performance column is numeric
    if df[perf_col].dtype == 'object':
        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
    
    # Remove rows with NaN in performance column
    df_clean = df.dropna(subset=[perf_col])
    
    if len(df_clean) == 0:
        return ["No valid performance data found"], []
    
    # Compute metrics per recruitment source
    source_analysis = df_clean.groupby(source_col).agg({
        perf_col: ['mean', 'count'],
        status_col: lambda x: (x.str.contains('Active', case=False, na=False).sum() / len(x) * 100)
    })
    source_analysis.columns = ['Avg_Performance', 'Count', 'Active_Rate_%']
    source_analysis = source_analysis.sort_values(['Avg_Performance', 'Active_Rate_%'], ascending=[False, False])
    
    answer_parts = [
        "## Strategic Ranking of Recruitment Channels\n\n"
        "Ranked by average performance score and active employment rate:\n"
    ]
    
    rank = 1
    for source, row in source_analysis.iterrows():
        answer_parts.append(
            f"{rank}. **{source}**: "
            f"Avg Performance: {row['Avg_Performance']:.2f}, "
            f"Active Rate: {row['Active_Rate_%']:.1f}%, "
            f"Total Hires: {int(row['Count'])}"
        )
        rank += 1
    
    evidence_facts = [
        {
            "subject": "Recruitment Source Ranking",
            "predicate": "has_sources",
            "object": str(len(source_analysis)),
            "source": ["strategic_query"]
        }
    ]
    
    return answer_parts, evidence_facts


def process_s2_2(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    S2.2: Underperforming recruitment sources
    Identify sources with low performance and early turnover.
    """
    source_col = columns.get("RecruitmentSource")
    perf_col = columns.get("PerformanceScore")
    status_col = columns.get("EmploymentStatus")
    
    if not all([source_col, perf_col, status_col]):
        return ["Missing required columns"], []
    
    # Ensure performance column is numeric
    if df[perf_col].dtype == 'object':
        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
    
    # Remove rows with NaN in performance column
    df_clean = df.dropna(subset=[perf_col])
    
    if len(df_clean) == 0:
        return ["No valid performance data found"], []
    
    # Low performance threshold (bottom 25%)
    perf_threshold = df_clean[perf_col].quantile(0.25)
    
    # Analyze each source
    source_stats = df_clean.groupby(source_col).agg({
        perf_col: ['mean', 'count'],
        status_col: lambda x: (x.str.contains('Terminated', case=False, na=False).sum() / len(x) * 100)
    })
    source_stats.columns = ['Avg_Performance', 'Count', 'Termination_Rate_%']
    
    # Filter for underperformers
    underperformers = source_stats[
        (source_stats['Avg_Performance'] < perf_threshold) | 
        (source_stats['Termination_Rate_%'] > 30)  # >30% termination rate
    ].sort_values(['Termination_Rate_%', 'Avg_Performance'], ascending=[False, True])
    
    if len(underperformers) == 0:
        return ["No underperforming recruitment sources identified."], []
    
    answer_parts = [
        "## Underperforming Recruitment Sources\n\n"
        f"Identified **{len(underperformers)} recruitment sources** with low performance "
        f"(<{perf_threshold:.1f}) or high termination rates (>30%):\n"
    ]
    
    for source, row in underperformers.iterrows():
        answer_parts.append(
            f"- **{source}**: "
            f"Avg Performance: {row['Avg_Performance']:.2f}, "
            f"Termination Rate: {row['Termination_Rate_%']:.1f}%, "
            f"Hires: {int(row['Count'])}"
        )
    
    answer_parts.append(
        "\n### Recommendation:\n"
        "Consider revising the sourcing mix by reducing reliance on these channels "
        "or implementing additional screening/onboarding processes."
    )
    
    evidence_facts = [
        {
            "subject": "Underperforming Sources",
            "predicate": "has_count",
            "object": str(len(underperformers)),
            "source": ["strategic_query"]
        }
    ]
    
    return answer_parts, evidence_facts


def process_o1_1(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    O1.1: Departmental performance monitoring
    Monitor performance score distribution by department.
    Highlight departments whose performance trends decline or fall below organisation-wide averages.
    """
    perf_col = columns.get("PerformanceScore")
    dept_col = columns.get("Department")
    
    if perf_col is None:
        # Try to find Department column
        if "Department" in df.columns:
            dept_col = "Department"
        elif "DeptID" in df.columns:
            dept_col = "DeptID"
        else:
            return ["Missing required columns. Need PerformanceScore and Department columns."], []
    else:
        if dept_col is None:
            if "Department" in df.columns:
                dept_col = "Department"
            elif "DeptID" in df.columns:
                dept_col = "DeptID"
            else:
                return ["Missing Department column."], []
    
    # Ensure performance column is numeric
    if df[perf_col].dtype == 'object':
        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
    
    # Remove rows with NaN in required columns
    df_clean = df.dropna(subset=[perf_col, dept_col])
    
    if len(df_clean) == 0:
        return ["No valid performance data found. Please ensure PerformanceScore and Department columns contain data."], []
    
    # Compute comprehensive statistics by department
    dept_stats = df_clean.groupby(dept_col)[perf_col].agg(['mean', 'median', 'std', 'count', 'min', 'max']).round(2)
    dept_stats = dept_stats.sort_values('mean', ascending=False)  # Highest first
    
    # Calculate overall organizational statistics
    overall_mean = df_clean[perf_col].mean()
    overall_std = df_clean[perf_col].std()
    overall_median = df_clean[perf_col].median()
    
    answer_parts = [
        "## Departmental Performance Monitoring\n\n"
        f"**Overall Organizational Benchmark:**\n"
        f"- Average performance score: **{overall_mean:.2f}**\n"
        f"- Median performance score: **{overall_median:.2f}**\n"
        f"- Standard deviation: **{overall_std:.2f}**\n\n"
        f"**Performance Analysis by Department:**\n\n"
    ]
    
    # Create table header
    answer_parts.append("| Department | Mean Score | vs Org Avg | Deviation | Median | Std Dev | Range | Employees | Status |\n")
    answer_parts.append("|-----------|------------|------------|-----------|--------|---------|-------|-----------|--------|\n")
    
    below_benchmark_depts = []
    department_details = []
    
    for dept, row in dept_stats.iterrows():
        dept_mean = row['mean']
        dept_median = row['median']
        dept_std = row['std']
        dept_count = int(row['count'])
        dept_min = row['min']
        dept_max = row['max']
        
        # Convert department to string to ensure proper display
        dept_name = str(dept).strip()
        # Check if department column contains actual names (not just numeric IDs)
        # Try to find a 'Department' column with text values
        if dept_name.replace('.', '').replace('-', '').isdigit():
            # If department is numeric, try to get actual department name from other columns
            dept_rows = df_clean[df_clean[dept_col] == dept]
            if len(dept_rows) > 0:
                # Check all columns for potential department names
                for col in df_clean.columns:
                    if col != dept_col and col.lower() in ['department', 'dept', 'division', 'unit']:
                        dept_names = dept_rows[col].dropna().unique()
                        if len(dept_names) > 0 and not str(dept_names[0]).replace('.', '').replace('-', '').isdigit():
                            dept_name = str(dept_names[0])
                            break
                # If still numeric, try to use the dept_col value as-is but format it better
                if dept_name.replace('.', '').replace('-', '').isdigit():
                    # Keep numeric but format as "Department {number}"
                    dept_name = f"Department {dept_name}"
        
        # Calculate deviation from organizational benchmark
        deviation = dept_mean - overall_mean
        deviation_pct = (deviation / overall_mean * 100) if overall_mean > 0 else 0
        
        # Identify departments below benchmark
        is_below = dept_mean < overall_mean
        is_significantly_below = dept_mean < (overall_mean - overall_std)
        
        if is_below:
            below_benchmark_depts.append({
                'department': dept_name,
                'mean_score': dept_mean,
                'deviation': abs(deviation),
                'deviation_pct': abs(deviation_pct),
                'employee_count': dept_count,
                'is_significant': is_significantly_below
            })
        
        status_icon = "🚨" if is_significantly_below else "⚠️" if is_below else "✅"
        status_text = "**SIGNIFICANTLY BELOW**" if is_significantly_below else "**BELOW**" if is_below else "ABOVE"
        
        department_details.append({
            'department': dept_name,
            'mean': dept_mean,
            'median': dept_median,
            'std': dept_std,
            'count': dept_count,
            'min': dept_min,
            'max': dept_max,
            'deviation': deviation,
            'is_below': is_below
        })
        
        # Format mean score with bold if below benchmark
        if is_significantly_below:
            mean_str = f"**{dept_mean:.2f}**"
        elif is_below:
            mean_str = f"**{dept_mean:.2f}**"
        else:
            mean_str = f"{dept_mean:.2f}"
        
        answer_parts.append(
            f"| {dept_name} | {mean_str} | {overall_mean:.2f} | {deviation:.2f} ({deviation_pct:.2f}%) | "
            f"{dept_median:.2f} | {dept_std:.2f} | {dept_min:.2f}-{dept_max:.2f} | "
            f"{dept_count} | {status_icon} {status_text} |\n"
        )
    
    # Summary of departments below benchmark
    if below_benchmark_depts:
        # Sort by severity (worst first)
        below_benchmark_depts.sort(key=lambda x: (x['is_significant'], x['deviation']), reverse=True)
        
        significant_below = [d for d in below_benchmark_depts if d['is_significant']]
        
        answer_parts.append(
            f"\n## ⚠️ Departments Requiring Attention\n\n"
            f"**{len(below_benchmark_depts)} department(s) below organizational benchmark:**\n\n"
        )
        
        if significant_below:
            answer_parts.append(
                f"**🚨 Critical ({len(significant_below)} department(s) significantly below):**\n"
            )
            for dept_info in significant_below:
                answer_parts.append(
                    f"- **{dept_info['department']}**: "
                    f"Average {dept_info['mean_score']:.2f} "
                    f"({dept_info['deviation']:.2f} below benchmark, {dept_info['deviation_pct']:.1f}%), "
                    f"{dept_info['employee_count']} employees"
                )
            
            if len(below_benchmark_depts) > len(significant_below):
                answer_parts.append(f"\n**⚠️ Below Benchmark:**\n")
                for dept_info in [d for d in below_benchmark_depts if not d['is_significant']]:
                    answer_parts.append(
                        f"- **{dept_info['department']}**: "
                        f"Average {dept_info['mean_score']:.2f} "
                        f"({dept_info['deviation']:.2f} below benchmark), "
                        f"{dept_info['employee_count']} employees"
                    )
        else:
            for dept_info in below_benchmark_depts:
                answer_parts.append(
                    f"- **{dept_info['department']}**: "
                    f"Average {dept_info['mean_score']:.2f} "
                    f"({dept_info['deviation']:.2f} below benchmark, {dept_info['deviation_pct']:.1f}%), "
                    f"{dept_info['employee_count']} employees"
                )
        
        answer_parts.append(
            f"\n**Recommended Actions:**\n"
            f"- Review department management practices\n"
            f"- Assess training and development needs\n"
            f"- Investigate root causes of performance gaps\n"
            f"- Implement targeted performance improvement plans"
        )
    else:
        answer_parts.append(
            f"\n✅ **All departments meet or exceed the organizational benchmark.** "
            f"No immediate intervention required."
        )
    
    # Create evidence facts
    evidence_facts = [
        {
            "subject": "Department Performance Analysis",
            "predicate": "has_departments_below_benchmark",
            "object": str(len(below_benchmark_depts)),
            "source": ["operational_query"]
        }
    ]
    
    # Add individual department facts
    for dept_info in below_benchmark_depts[:10]:  # Limit to top 10
        evidence_facts.append({
            "subject": dept_info['department'],
            "predicate": "has_performance_below_benchmark",
            "object": str(dept_info['mean_score']),
            "source": ["operational_query"]
        })
    
    return answer_parts, evidence_facts


def process_o1_2(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    O1.2: Tracking low-performance concentrations
    Track the proportion of employees rated as 'Needs Improvement' or on PIP within each department.
    """
    perf_col = columns.get("PerformanceScore")
    dept_col = columns.get("Department")
    
    if perf_col is None:
        return ["Missing required columns. Need PerformanceScore column."], []
    
    if dept_col is None:
        if "Department" in df.columns:
            dept_col = "Department"
        elif "DeptID" in df.columns:
            dept_col = "DeptID"
        else:
            return ["Missing Department column."], []
    
    # Handle both numeric and categorical performance scores
    df_clean = df.dropna(subset=[perf_col, dept_col]).copy()
    
    if len(df_clean) == 0:
        return ["No valid performance data found. Please ensure PerformanceScore and Department columns contain data."], []
    
    # Check if performance column is categorical (text) or numeric
    is_categorical = df_clean[perf_col].dtype == 'object'
    
    if is_categorical:
        # Handle categorical performance scores (e.g., "Needs Improvement", "Fully Meets", "Exceeds")
        # Identify low performance categories
        low_perf_keywords = ['needs improvement', 'needs', 'improvement', 'pip', 'below', 'low', 'poor', 'unsatisfactory']
        
        # Convert to lowercase for comparison
        df_clean['perf_lower'] = df_clean[perf_col].astype(str).str.lower()
        
        # Identify low performers
        df_clean['is_low_perf'] = df_clean['perf_lower'].apply(
            lambda x: any(keyword in x for keyword in low_perf_keywords)
        )
        
        # Calculate proportions by department
        dept_total = df_clean.groupby(dept_col).size()
        dept_low_perf = df_clean[df_clean['is_low_perf']].groupby(dept_col).size()
        dept_low_pct = (dept_low_perf / dept_total * 100).fillna(0).round(2)
        dept_low_pct = dept_low_pct.sort_values(ascending=False)
        
        # Get unique low performance categories
        low_perf_categories = df_clean[df_clean['is_low_perf']][perf_col].unique()
        
    else:
        # Handle numeric performance scores
        if df_clean[perf_col].dtype != 'float64' and df_clean[perf_col].dtype != 'int64':
            df_clean[perf_col] = pd.to_numeric(df_clean[perf_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[perf_col])
        
        # Define low performance threshold (bottom 25% or scores <= 2 if scale is 1-4)
        overall_min = df_clean[perf_col].min()
        overall_max = df_clean[perf_col].max()
        
        # Use bottom quartile or score <= 2 (assuming 1-4 scale)
        if overall_max <= 4:
            low_threshold = 2.0  # "Needs Improvement" typically corresponds to score of 2
        else:
            low_threshold = df_clean[perf_col].quantile(0.25)
        
        df_clean['is_low_perf'] = df_clean[perf_col] <= low_threshold
        
        # Calculate proportions by department
        dept_total = df_clean.groupby(dept_col).size()
        dept_low_perf = df_clean[df_clean['is_low_perf']].groupby(dept_col).size()
        dept_low_pct = (dept_low_perf / dept_total * 100).fillna(0).round(2)
        dept_low_pct = dept_low_pct.sort_values(ascending=False)
        
        low_perf_categories = None
    
    # Calculate overall low performance rate
    overall_low_pct = (df_clean['is_low_perf'].sum() / len(df_clean) * 100).round(2)
    
    answer_parts = [
        "## Low-Performance Concentration Tracking\n\n"
        f"**Overall Statistics:**\n"
        f"- Overall low performance rate: **{overall_low_pct:.1f}%**\n"
    ]
    
    if is_categorical and low_perf_categories is not None:
        answer_parts.append(
            f"- Low performance categories identified: {', '.join([str(c) for c in low_perf_categories[:5]])}\n"
        )
    else:
        answer_parts.append(
            f"- Low performance threshold: **≤{low_threshold:.2f}** (bottom quartile or score ≤ 2)\n"
        )
    
    answer_parts.append(f"\n**Low Performance by Department:**\n\n")
    
    # Alert thresholds
    critical_threshold = 30  # Critical if >30% low performers
    warning_threshold = 20   # Warning if >20% low performers
    
    critical_depts = []
    warning_depts = []
    department_details = []
    
    # Create table with department names and performance metrics
    answer_parts.append("| Department | Low Performance % | Low Performers | Total Employees | Status |\n")
    answer_parts.append("|------------|-------------------|----------------|----------------|--------|\n")
    
    for dept, pct in dept_low_pct.items():
        count = int(dept_low_perf.get(dept, 0))
        total = int(dept_total.get(dept, 0))
        
        # Convert department to string to ensure proper display
        dept_name = str(dept).strip()
        # Check if department column contains actual names (not just numeric IDs)
        if dept_name.replace('.', '').replace('-', '').isdigit():
            # If department is numeric, try to get actual department name from other columns
            dept_rows = df_clean[df_clean[dept_col] == dept]
            if len(dept_rows) > 0:
                # Check all columns for potential department names
                for col in df_clean.columns:
                    if col != dept_col and col.lower() in ['department', 'dept', 'division', 'unit']:
                        dept_names = dept_rows[col].dropna().unique()
                        if len(dept_names) > 0 and not str(dept_names[0]).replace('.', '').replace('-', '').isdigit():
                            dept_name = str(dept_names[0])
                            break
                # If still numeric, format it better
                if dept_name.replace('.', '').replace('-', '').isdigit():
                    dept_name = f"Department {dept_name}"
        
        # Determine alert level
        if pct > critical_threshold:
            critical_depts.append({
                'department': dept_name,
                'percentage': pct,
                'count': count,
                'total': total
            })
            status = "🚨 **CRITICAL**"
        elif pct > warning_threshold:
            warning_depts.append({
                'department': dept_name,
                'percentage': pct,
                'count': count,
                'total': total
            })
            status = "⚠️ **WARNING**"
        else:
            status = "✅ OK"
        
        department_details.append({
            'department': dept_name,
            'percentage': pct,
            'count': count,
            'total': total,
            'is_critical': pct > critical_threshold,
            'is_warning': warning_threshold < pct <= critical_threshold
        })
        
        # Format percentage with bold if critical/warning
        if pct > critical_threshold:
            pct_str = f"**{pct:.1f}%**"
        elif pct > warning_threshold:
            pct_str = f"**{pct:.1f}%**"
        else:
            pct_str = f"{pct:.1f}%"
        
        answer_parts.append(
            f"| {dept_name} | {pct_str} | {count} | {total} | {status} |\n"
        )
        
        # Remove old bullet point code if it exists - table format is used instead
    
    # Summary of alerts
    if critical_depts or warning_depts:
        answer_parts.append(f"\n## ⚠️ Alert Summary\n\n")
        
        if critical_depts:
            answer_parts.append(
                f"**🚨 CRITICAL ({len(critical_depts)} department(s) >{critical_threshold}% low performers):**\n\n"
            )
            for dept_info in critical_depts:
                answer_parts.append(
                    f"- **{dept_info['department']}**: "
                    f"{dept_info['percentage']:.1f}% ({dept_info['count']}/{dept_info['total']} employees) "
                    f"require immediate attention"
                )
            answer_parts.append("")
        
        if warning_depts:
            answer_parts.append(
                f"**⚠️ WARNING ({len(warning_depts)} department(s) >{warning_threshold}% low performers):**\n\n"
            )
            for dept_info in warning_depts:
                answer_parts.append(
                    f"- **{dept_info['department']}**: "
                    f"{dept_info['percentage']:.1f}% ({dept_info['count']}/{dept_info['total']} employees) "
                    f"require monitoring"
                )
            answer_parts.append("")
        
        answer_parts.append(
            f"**Recommended Actions:**\n"
            f"- Review department-specific performance management practices\n"
            f"- Assess training and development programs\n"
            f"- Implement Performance Improvement Plans (PIPs) where appropriate\n"
            f"- Conduct root cause analysis for high concentrations\n"
            f"- Provide additional support and resources to affected departments"
        )
    else:
        answer_parts.append(
            f"\n✅ **All departments are within acceptable thresholds.** "
            f"No immediate intervention required."
        )
    
    # Create evidence facts
    evidence_facts = [
        {
            "subject": "Low Performance Tracking",
            "predicate": "has_overall_low_performance_rate",
            "object": f"{overall_low_pct:.1f}%",
            "source": ["operational_query"]
        },
        {
            "subject": "Low Performance Tracking",
            "predicate": "has_critical_departments",
            "object": str(len(critical_depts)),
            "source": ["operational_query"]
        }
    ]
    
    # Add individual department facts
    for dept_info in (critical_depts + warning_depts)[:10]:  # Limit to top 10
        evidence_facts.append({
            "subject": dept_info['department'],
            "predicate": "has_low_performance_rate",
            "object": f"{dept_info['percentage']:.1f}%",
            "source": ["operational_query"]
        })
    
    return answer_parts, evidence_facts


def process_o2_1(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    O2.1: Absence patterns by employment status
    Compare average absence levels between active vs. terminated employees, broken down by Department and Role.
    Identify absence patterns statistically associated with future termination.
    """
    abs_col = columns.get("Absences")
    status_col = columns.get("EmploymentStatus")
    dept_col = columns.get("Department")
    pos_col = columns.get("Position")
    
    # Try to find columns if not provided
    if abs_col is None:
        for col in df.columns:
            if col.lower() in ['absences', 'absence', 'days absent']:
                abs_col = col
                print(f"✅ Found absences column: {col}")
                break
        if abs_col is None:
            from strategic_query_agent import normalize_column_name
            abs_col = normalize_column_name(df, "Absences")
            if abs_col:
                print(f"✅ Found absences column via normalize: {abs_col}")
    
    if status_col is None:
        for col in df.columns:
            if col.lower() in ['employmentstatus', 'employment status', 'status']:
                status_col = col
                print(f"✅ Found employment status column: {col}")
                break
        if status_col is None:
            from strategic_query_agent import normalize_column_name
            status_col = normalize_column_name(df, "EmploymentStatus")
            if status_col:
                print(f"✅ Found employment status column via normalize: {status_col}")
    
    if abs_col is None or status_col is None:
        available_cols = list(df.columns)[:30]
        return [f"Missing required columns. Need Absences (found: {abs_col}) and EmploymentStatus (found: {status_col}). Available columns: {', '.join(available_cols)}..."], []
    
    # Find department and position columns if not provided
    if dept_col is None:
        for col in df.columns:
            if col.lower() in ['department', 'deptid', 'dept']:
                dept_col = col
                break
        if dept_col is None:
            if "Department" in df.columns:
                dept_col = "Department"
            elif "DeptID" in df.columns:
                dept_col = "DeptID"
    
    if pos_col is None:
        for col in df.columns:
            if col.lower() in ['position', 'role', 'positionid']:
                pos_col = col
                break
        if pos_col is None:
            if "Position" in df.columns:
                pos_col = "Position"
            elif "Role" in df.columns:
                pos_col = "Role"
    
    # Ensure absences column is numeric
    if df[abs_col].dtype == 'object':
        df[abs_col] = pd.to_numeric(df[abs_col], errors='coerce')
    
    # Remove rows with NaN in required columns
    df_clean = df.dropna(subset=[abs_col, status_col]).copy()
    
    if len(df_clean) == 0:
        return ["No valid absence data found. Please ensure Absences and EmploymentStatus columns contain data."], []
    
    # Identify active and terminated employees - check for various status formats
    status_str = df_clean[status_col].astype(str).str.lower().str.strip()
    
    # Get unique status values for debugging
    unique_statuses = df_clean[status_col].astype(str).unique()
    print(f"🔍 Found {len(unique_statuses)} unique employment statuses: {list(unique_statuses)[:10]}")
    
    # Terminated status variations - check FIRST to avoid false positives
    # "Voluntarily Terminated", "Terminated for Cause" should match
    terminated_keywords = ['terminated', 'resigned', 'left', 'inactive', 'former', 'voluntarily', 'cause', 'termd', 'term']
    terminated_mask = status_str.apply(lambda x: any(keyword in x for keyword in terminated_keywords))
    terminated_df = df_clean[terminated_mask]
    
    # Active status variations - "Active" should match, but also check for non-terminated
    active_keywords = ['active', 'employed', 'current', 'working']
    active_mask = status_str.apply(lambda x: any(keyword in x for keyword in active_keywords))
    active_df = df_clean[active_mask]
    
    print(f"   Active employees (by keyword): {len(active_df)}, Terminated: {len(terminated_df)}")
    print(f"   Unique status values: {list(unique_statuses)[:10]}")
    
    # If no active found by keyword but we have data, try different approach
    if len(active_df) == 0 and len(df_clean) > 0:
        # Maybe all employees are active (no status column distinction)
        # Or status values are different - try using all non-terminated as active
        if len(terminated_df) > 0:
            # We have terminated, so active = not terminated
            active_df = df_clean[~terminated_mask]
            print(f"   Using non-terminated as active: {len(active_df)} employees")
        else:
            # No terminated found - maybe all are active?
            # Check if there are any status values that might indicate active
            # If status column has values but none match "terminated", assume all are active
            active_df = df_clean
            print(f"   No terminated found, treating all as active: {len(active_df)} employees")
    
    # Final check: if still no active, but we have data, use all non-terminated
    if len(active_df) == 0 and len(df_clean) > 0:
        active_df = df_clean[~terminated_mask] if len(terminated_df) > 0 else df_clean
        print(f"   Final fallback: using {len(active_df)} as active employees")
    
    if len(active_df) == 0:
        # Try to provide helpful error message
        return [f"No active employees found. Available status values: {', '.join([str(s) for s in unique_statuses[:10]])}. Please check EmploymentStatus column values."], []
    
    if len(terminated_df) == 0:
        # If no terminated, we can still analyze active employees only
        print(f"⚠️  No terminated employees found, analyzing active employees only")
        # Continue with analysis using only active employees
    
    # Overall comparison
    active_avg = active_df[abs_col].mean()
    active_median = active_df[abs_col].median()
    active_std = active_df[abs_col].std()
    active_count = len(active_df)
    
    terminated_avg = terminated_df[abs_col].mean()
    terminated_median = terminated_df[abs_col].median()
    terminated_std = terminated_df[abs_col].std()
    terminated_count = len(terminated_df)
    
    difference = terminated_avg - active_avg
    difference_pct = (difference / active_avg * 100) if active_avg > 0 else 0
    
    # Statistical significance test (simple comparison)
    is_significant = abs(difference) > (active_std + terminated_std) / 2
    
    answer_parts = [
        "## Absence Patterns by Employment Status\n\n"
        f"**Overall Comparison:**\n\n"
        f"**Active Employees:**\n"
        f"- Average absences: **{active_avg:.2f}** days\n"
        f"- Median: {active_median:.2f} days\n"
        f"- Standard deviation: {active_std:.2f} days\n"
        f"- Employee count: {active_count}\n\n"
        f"**Terminated Employees:**\n"
        f"- Average absences: **{terminated_avg:.2f}** days\n"
        f"- Median: {terminated_median:.2f} days\n"
        f"- Standard deviation: {terminated_std:.2f} days\n"
        f"- Employee count: {terminated_count}\n\n"
        f"**Key Finding:**\n"
        f"- Difference: **{difference:+.2f}** days ({difference_pct:+.1f}% {'higher' if difference > 0 else 'lower'}) "
        f"for terminated employees\n"
    ]
    
    if is_significant:
        answer_parts.append(
            f"- ⚠️ **Statistically significant difference** - absence patterns are associated with termination\n"
        )
    
    # Analysis by Department
    if dept_col and dept_col in df_clean.columns:
        answer_parts.append(f"\n**Analysis by Department:**\n\n")
        answer_parts.append("| Department | Active Avg | Terminated Avg | Difference | Active Count | Terminated Count | Status |\n")
        answer_parts.append("|-----------|------------|----------------|------------|--------------|------------------|--------|\n")
        
        dept_comparison = df_clean.groupby([dept_col, status_col])[abs_col].agg(['mean', 'count']).round(2)
        dept_pivot = dept_comparison.unstack(fill_value=0)
        
        dept_patterns = []
        
        for dept in dept_pivot.index:
            active_mean = dept_pivot.loc[dept, ('mean', 'Active')] if ('mean', 'Active') in dept_pivot.columns else 0
            term_mean = dept_pivot.loc[dept, ('mean', 'Terminated')] if ('mean', 'Terminated') in dept_pivot.columns else 0
            active_cnt = dept_pivot.loc[dept, ('count', 'Active')] if ('count', 'Active') in dept_pivot.columns else 0
            term_cnt = dept_pivot.loc[dept, ('count', 'Terminated')] if ('count', 'Terminated') in dept_pivot.columns else 0
            
            if active_cnt > 0 and term_cnt > 0:
                dept_diff = term_mean - active_mean
                is_high_risk = dept_diff > (active_avg * 0.5)  # 50% higher than overall active average
                
                # Convert department to string
                dept_name = str(dept).strip()
                if dept_name.replace('.', '').replace('-', '').isdigit():
                    dept_rows = df_clean[df_clean[dept_col] == dept]
                    if len(dept_rows) > 0 and 'Department' in df_clean.columns and df_clean['Department'].dtype == 'object':
                        dept_names = dept_rows['Department'].dropna().unique()
                        if len(dept_names) > 0:
                            dept_name = str(dept_names[0])
                
                dept_patterns.append({
                    'department': dept_name,
                    'active_avg': active_mean,
                    'terminated_avg': term_mean,
                    'difference': dept_diff,
                    'active_count': int(active_cnt),
                    'terminated_count': int(term_cnt),
                    'is_high_risk': is_high_risk
                })
        
        # Sort by difference (highest risk first)
        dept_patterns.sort(key=lambda x: x['difference'], reverse=True)
        
        for dept_info in dept_patterns:
            status = "🚨 **HIGH RISK**" if dept_info['is_high_risk'] else "⚠️ WARNING" if dept_info['difference'] > 0 else "✅ OK"
            diff_str = f"**{dept_info['difference']:+.2f}**" if dept_info['is_high_risk'] else f"{dept_info['difference']:+.2f}"
            answer_parts.append(
                f"| {dept_info['department']} | {dept_info['active_avg']:.2f} | {dept_info['terminated_avg']:.2f} | "
                f"{diff_str} | {dept_info['active_count']} | {dept_info['terminated_count']} | {status} |\n"
            )
        
        # Identify high-risk departments
        high_risk_depts = [d for d in dept_patterns if d['is_high_risk']]
        if high_risk_depts:
            answer_parts.append(
                f"\n⚠️ **High-Risk Departments** (terminated employees have >50% higher absences):\n"
            )
            for dept_info in high_risk_depts:
                answer_parts.append(
                    f"- **{dept_info['department']}**: "
                    f"{dept_info['difference']:.2f} days higher absences for terminated employees"
                )
    
    # Analysis by Role/Position
    if pos_col and pos_col in df_clean.columns:
        answer_parts.append(f"\n**Analysis by Role/Position:**\n\n")
        
        role_comparison = df_clean.groupby([pos_col, status_col])[abs_col].agg(['mean', 'count']).round(2)
        role_pivot = role_comparison.unstack(fill_value=0)
        
        role_patterns = []
        
        for role in role_pivot.index[:15]:  # Top 15 roles
            active_mean = role_pivot.loc[role, ('mean', 'Active')] if ('mean', 'Active') in role_pivot.columns else 0
            term_mean = role_pivot.loc[role, ('mean', 'Terminated')] if ('mean', 'Terminated') in role_pivot.columns else 0
            active_cnt = role_pivot.loc[role, ('count', 'Active')] if ('count', 'Active') in role_pivot.columns else 0
            term_cnt = role_pivot.loc[role, ('count', 'Terminated')] if ('count', 'Terminated') in role_pivot.columns else 0
            
            if active_cnt > 0 and term_cnt > 0:
                role_diff = term_mean - active_mean
                role_patterns.append({
                    'role': role,
                    'active_avg': active_mean,
                    'terminated_avg': term_mean,
                    'difference': role_diff,
                    'active_count': int(active_cnt),
                    'terminated_count': int(term_cnt)
                })
        
        # Sort by difference
        role_patterns.sort(key=lambda x: x['difference'], reverse=True)
        
        for role_info in role_patterns[:10]:  # Top 10
            answer_parts.append(
                f"- **{role_info['role']}**:\n"
                f"  - Active: {role_info['active_avg']:.2f} days (n={role_info['active_count']})\n"
                f"  - Terminated: {role_info['terminated_avg']:.2f} days (n={role_info['terminated_count']})\n"
                f"  - Difference: {role_info['difference']:+.2f} days\n"
            )
    
    # Summary and recommendations
    answer_parts.append(
        f"\n## Key Insights\n\n"
        f"**Absence Pattern Association with Termination:**\n"
    )
    
    if is_significant and difference > 0:
        answer_parts.append(
            f"- ✅ **Confirmed**: Terminated employees show significantly higher absence rates "
            f"({difference:.2f} days, {difference_pct:.1f}% higher)\n"
            f"- This suggests absence patterns may be an early indicator of termination risk\n\n"
            f"**Recommended Actions:**\n"
            f"- Monitor employees with above-average absences as potential retention risks\n"
            f"- Implement early intervention programs for high-absence employees\n"
            f"- Review absence policies and support mechanisms\n"
            f"- Conduct exit interviews to understand absence-related termination causes"
        )
    elif difference > 0:
        answer_parts.append(
            f"- ⚠️ **Moderate association**: Terminated employees show higher absence rates "
            f"({difference:.2f} days), but difference is not statistically significant\n"
            f"- Monitor trends and consider additional factors\n"
        )
    else:
        answer_parts.append(
            f"- ℹ️ **No clear association**: Absence patterns do not show significant correlation with termination\n"
        )
    
    # Create evidence facts
    evidence_facts = [
        {
            "subject": "Absence Analysis",
            "predicate": "has_active_avg_absences",
            "object": f"{active_avg:.2f}",
            "source": ["operational_query"]
        },
        {
            "subject": "Absence Analysis",
            "predicate": "has_terminated_avg_absences",
            "object": f"{terminated_avg:.2f}",
            "source": ["operational_query"]
        },
        {
            "subject": "Absence Analysis",
            "predicate": "has_absence_difference",
            "object": f"{difference:.2f}",
            "source": ["operational_query"]
        }
    ]
    
    # Add department facts if available
    if dept_col and dept_col in df_clean.columns and 'high_risk_depts' in locals():
        for dept_info in high_risk_depts[:5]:
            evidence_facts.append({
                "subject": dept_info['department'],
                "predicate": "has_high_absence_termination_risk",
                "object": f"{dept_info['difference']:.2f}",
                "source": ["operational_query"]
            })
    
    return answer_parts, evidence_facts


def process_o3_1(df: pd.DataFrame, columns: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    O3.1: Team-level engagement monitoring
    Analyze engagement survey scores by manager.
    For each manager, compute average engagement score and identify managers with low engagement.
    """
    eng_col = columns.get("EngagementSurvey")
    mgr_col = columns.get("ManagerName")
    
    # Try to find columns if not provided
    if eng_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'engagement' in col_lower and ('survey' in col_lower or 'score' in col_lower):
                eng_col = col
                print(f"✅ Found engagement survey column: {col}")
                break
        if eng_col is None:
            from strategic_query_agent import normalize_column_name
            eng_col = normalize_column_name(df, "EngagementSurvey")
            if eng_col:
                print(f"✅ Found engagement survey column via normalize: {eng_col}")
    
    if mgr_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'manager' in col_lower and 'name' in col_lower:
                mgr_col = col
                print(f"✅ Found manager name column: {col}")
                break
        if mgr_col is None:
            from strategic_query_agent import normalize_column_name
            mgr_col = normalize_column_name(df, "ManagerName")
            if mgr_col:
                print(f"✅ Found manager name column via normalize: {mgr_col}")
    
    if eng_col is None or mgr_col is None:
        available_cols = list(df.columns)[:30]
        return [f"Missing required columns. Need EngagementSurvey (found: {eng_col}) and ManagerName (found: {mgr_col}). Available columns: {', '.join(available_cols)}..."], []
    
    # Ensure engagement column is numeric
    if df[eng_col].dtype == 'object':
        df[eng_col] = pd.to_numeric(df[eng_col], errors='coerce')
    
    # Remove rows with NaN in required columns
    df_clean = df.dropna(subset=[eng_col, mgr_col])
    
    if len(df_clean) == 0:
        return ["No valid engagement data found. Please ensure EngagementSurvey and ManagerName columns contain data."], []
    
    # Group by manager and compute statistics
    mgr_stats = df_clean.groupby(mgr_col)[eng_col].agg(['mean', 'count', 'std', 'min', 'max']).round(2)
    mgr_stats = mgr_stats.sort_values('mean', ascending=True)  # Lowest first
    
    # Calculate overall statistics
    overall_mean = df_clean[eng_col].mean()
    overall_std = df_clean[eng_col].std()
    low_threshold = overall_mean - overall_std  # 1 std dev below mean
    
    answer_parts = [
        "## Team-Level Engagement Monitoring\n\n"
        f"**Overall Statistics:**\n"
        f"- Average engagement score: **{overall_mean:.2f}**\n"
        f"- Standard deviation: **{overall_std:.2f}**\n"
        f"- Low engagement threshold (1 std dev below mean): **{low_threshold:.2f}**\n\n"
        f"**Engagement Analysis by Manager:**\n\n"
    ]
    
    # Create table header
    answer_parts.append("| Manager | Avg Score | vs Org Avg | Deviation | Team Size | Score Range | Std Dev | Status |\n")
    answer_parts.append("|---------|-----------|------------|-----------|-----------|-------------|---------|--------|\n")
    
    low_engagement_managers = []
    manager_details = []
    
    for mgr, row in mgr_stats.iterrows():
        avg_score = row['mean']
        team_size = int(row['count'])
        std_dev = row['std']
        min_score = row['min']
        max_score = row['max']
        
        # Convert manager to string
        mgr_name = str(mgr).strip()
        
        # Identify managers with low engagement (below threshold)
        is_low = avg_score < low_threshold
        deviation = avg_score - overall_mean
        
        if is_low:
            low_engagement_managers.append({
                'manager': mgr_name,
                'avg_score': avg_score,
                'team_size': team_size,
                'below_threshold': overall_mean - avg_score
            })
            status = "🚨 **LOW ENGAGEMENT**"
        elif avg_score < overall_mean:
            status = "⚠️ **BELOW AVERAGE**"
        else:
            status = "✅ OK"
        
        manager_details.append({
            'manager': mgr_name,
            'avg_score': avg_score,
            'team_size': team_size,
            'std_dev': std_dev,
            'min_score': min_score,
            'max_score': max_score,
            'is_low': is_low
        })
        
        # Format score with bold if low
        if is_low:
            score_str = f"**{avg_score:.2f}**"
        elif avg_score < overall_mean:
            score_str = f"**{avg_score:.2f}**"
        else:
            score_str = f"{avg_score:.2f}"
        
        answer_parts.append(
            f"| {mgr_name} | {score_str} | {overall_mean:.2f} | {deviation:+.2f} | {team_size} | "
            f"{min_score:.2f}-{max_score:.2f} | {std_dev:.2f} | {status} |\n"
        )
    
    # Summary of low engagement managers
    if low_engagement_managers:
        answer_parts.append(
            f"\n## ⚠️ Priority Follow-up Required\n\n"
            f"**{len(low_engagement_managers)} manager(s) with persistently low engagement** "
            f"(below threshold of {low_threshold:.2f}):\n\n"
        )
        
        # Sort by how far below threshold (worst first)
        low_engagement_managers.sort(key=lambda x: x['avg_score'])
        
        for mgr_info in low_engagement_managers:
            answer_parts.append(
                f"- **{mgr_info['manager']}**: "
                f"Average score {mgr_info['avg_score']:.2f} "
                f"({mgr_info['below_threshold']:.2f} below overall mean), "
                f"Team size: {mgr_info['team_size']}"
            )
        
        answer_parts.append(
            f"\n**Recommended Actions:**\n"
            f"- One-on-one coaching sessions\n"
            f"- Team climate assessments\n"
            f"- Feedback sessions with team members\n"
            f"- Review management practices and support needs"
        )
    else:
        answer_parts.append(
            f"\n✅ **All managers meet the engagement threshold.** "
            f"No immediate intervention required."
        )
    
    # Create evidence facts for low engagement managers
    evidence_facts = [
        {
            "subject": "Manager Engagement Analysis",
            "predicate": "has_low_engagement_count",
            "object": str(len(low_engagement_managers)),
            "source": ["operational_query"]
        }
    ]
    
    # Add individual manager facts
    for mgr_info in low_engagement_managers[:10]:  # Limit to top 10
        evidence_facts.append({
            "subject": mgr_info['manager'],
            "predicate": "has_low_engagement_score",
            "object": str(mgr_info['avg_score']),
            "source": ["operational_query"]
        })
    
    return answer_parts, evidence_facts


def process_o4_1(df: pd.DataFrame, columns: Dict[str, str], query_info: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    O4.1: Average per group with min/max
    Examples:
    - "get the average performance score per manager and bring me the min"
    - "average over absences per department and return the department with max"
    """
    metric_col = columns.get("metric")
    group_by_col = columns.get("group_by")
    aggregate_op = query_info.get("aggregate_op", "min")  # "min" or "max"
    
    if not metric_col or not group_by_col:
        return [f"Missing required columns. Need metric column and group_by column."], []
    
    # Ensure metric column is numeric
    if df[metric_col].dtype == 'object':
        df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
    
    # Remove rows with NaN in required columns
    df_clean = df.dropna(subset=[metric_col, group_by_col])
    
    if len(df_clean) == 0:
        return [f"No valid data found. Please ensure {metric_col} and {group_by_col} columns contain data."], []
    
    # Group by and calculate average
    group_avg = df_clean.groupby(group_by_col)[metric_col].mean().round(2)
    group_avg = group_avg.sort_values(ascending=(aggregate_op == "min"))
    
    # Get the min or max
    if aggregate_op == "min":
        result_value = group_avg.iloc[0]
        result_group = group_avg.index[0]
        op_label = "minimum"
    else:
        result_value = group_avg.iloc[-1]
        result_group = group_avg.index[-1]
        op_label = "maximum"
    
    # Build answer
    metric_name = metric_col.replace('_', ' ').title()
    group_name = group_by_col.replace('_', ' ').title()
    
    answer_parts = [
        f"## Average {metric_name} per {group_name} - {op_label.title()}\n\n"
        f"**Result:**\n"
        f"- {group_name}: **{result_group}**\n"
        f"- Average {metric_name}: **{result_value:.2f}**\n\n"
        f"**All Averages:**\n\n"
    ]
    
    # Create table
    answer_parts.append(f"| {group_name} | Average {metric_name} |\n")
    answer_parts.append(f"|{'---' * 2}|\n")
    
    for group_val, avg_val in group_avg.items():
        is_result = (group_val == result_group)
        group_str = f"**{group_val}**" if is_result else str(group_val)
        avg_str = f"**{avg_val:.2f}**" if is_result else f"{avg_val:.2f}"
        answer_parts.append(f"| {group_str} | {avg_str} |\n")
    
    # Evidence facts
    evidence_facts = [
        {
            "subject": f"{group_name} Analysis",
            "predicate": f"has_{op_label}_average_{metric_name.lower().replace(' ', '_')}",
            "object": str(result_value),
            "source": ["operational_query"]
        },
        {
            "subject": result_group,
            "predicate": f"has_average_{metric_name.lower().replace(' ', '_')}",
            "object": str(result_value),
            "source": ["operational_query"]
        }
    ]
    
    return answer_parts, evidence_facts

