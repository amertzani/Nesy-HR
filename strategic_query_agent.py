"""
Strategic Query Agent - Processes multi-variable analytical queries
Uses statistics from Statistics Agent and facts from Knowledge Graph
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from agent_system import document_agents, agents_store, STATISTICS_AGENT_ID
from knowledge import graph as kb_graph
import re


STRATEGIC_QUERY_AGENT_ID = "strategic_query_agent"


def get_all_statistics() -> List[Dict[str, Any]]:
    """Get statistics from all document agents"""
    all_stats = []
    for agent_id, agent in document_agents.items():
        if hasattr(agent, 'metadata') and agent.metadata:
            stats = agent.metadata.get("statistics")
            if stats:
                stats["document_id"] = agent.document_id
                stats["document_name"] = agent.document_name
                all_stats.append(stats)
    return all_stats


def reconstruct_dataframe_from_facts() -> Optional[pd.DataFrame]:
    """
    Reconstruct a DataFrame from knowledge graph facts.
    This allows us to do multi-variable analysis without needing the original CSV file.
    Optimized for large graphs by early skipping of metadata.
    """
    if kb_graph is None or len(kb_graph) == 0:
        print("⚠️  Knowledge graph is empty or None")
        return None
    
    try:
        print(f"🔄 Starting DataFrame reconstruction from {len(kb_graph)} facts...")
        import time
        start_time = time.time()
        
        # Extract all employee facts
        employees = {}
        processed_count = 0
        skipped_metadata = 0
        
        # Pre-compile metadata patterns for faster checking
        metadata_patterns = [
            'fact_subject', 'fact_predicate', 'fact_object', 'has_details',
            'source_document', 'uploaded_at', 'is_inferred', 'confidence', 'agent_id'
        ]
        
        for s, p, o in kb_graph:
            processed_count += 1
            
            # Progress logging for large graphs
            if processed_count % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"   Processed {processed_count}/{len(kb_graph)} facts ({elapsed:.1f}s)...")
            # Skip metadata triples (optimized check)
            predicate_str = str(p)
            predicate_lower = predicate_str.lower()
            if any(meta in predicate_lower for meta in metadata_patterns):
                skipped_metadata += 1
                continue
            
            # Extract subject (employee name)
            subject_str = str(s)
            if 'urn:entity:' in subject_str:
                from urllib.parse import unquote
                subject_clean = unquote(subject_str.split('urn:entity:')[-1]).replace('_', ' ')
            elif 'urn:' in subject_str and 'entity' not in subject_str:
                # Handle other urn: formats - extract the last part
                from urllib.parse import unquote
                parts = subject_str.split('urn:')
                if len(parts) > 1:
                    subject_clean = unquote(parts[-1]).replace('_', ' ')
                else:
                    subject_clean = subject_str.strip()
            else:
                subject_clean = subject_str.strip()
            
            # Extract predicate (attribute)
            # Handle different predicate formats: "has_salary", "urn:predicate:has_salary", etc.
            predicate_clean = predicate_str
            if 'urn:predicate:' in predicate_str:
                from urllib.parse import unquote
                predicate_clean = unquote(predicate_str.split('urn:predicate:')[-1]).replace('_', ' ').strip()
            elif 'urn:' in predicate_str and 'predicate' not in predicate_str:
                # Handle other urn: formats - extract the last part
                from urllib.parse import unquote
                parts = predicate_str.split('urn:')
                if len(parts) > 1:
                    predicate_clean = unquote(parts[-1]).replace('_', ' ').strip()
                else:
                    predicate_clean = predicate_str.replace('has_', '').replace('_', ' ').strip()
            elif '#' in predicate_str:
                predicate_clean = predicate_str.split('#')[-1].replace('_', ' ').strip()
            elif '/' in predicate_str:
                predicate_clean = predicate_str.split('/')[-1].replace('_', ' ').strip()
            else:
                predicate_clean = predicate_str.replace('has_', '').replace('_', ' ').strip()
            
            # Extract object (value)
            object_val = str(o).strip()
            
            # Skip if predicate is empty or object is empty
            if not predicate_clean or not object_val or object_val.lower() in ['none', 'null', 'nan']:
                continue
            
            # Normalize employee name (handle Last, First format)
            if subject_clean not in employees:
                employees[subject_clean] = {}
            
            # Map common predicate variations to standard column names
            # Handle predicates like "has salary", "works in department", etc.
            # Also handle actual CSV column names like "has PerfScoreID", "has PerformanceScore", etc.
            predicate_mapping = {
                # Salary variations
                'has salary': 'Salary',
                'salary': 'Salary',
                # Performance score variations (handle both PerformanceScore and PerfScoreID)
                'has performance score': 'PerformanceScore',
                'has performance': 'PerformanceScore',
                'performance score': 'PerformanceScore',
                'performance': 'PerformanceScore',
                'perfscoreid': 'PerformanceScore',  # Handle PerfScoreID variation
                'perf score id': 'PerformanceScore',
                'has perfscoreid': 'PerformanceScore',  # Actual CSV column name stored as predicate
                'has performancescore': 'PerformanceScore',
                'performancescore': 'PerformanceScore',
                'perf': 'PerformanceScore',  # Short form
                # Engagement variations - CRITICAL: handle "has engagement survey" from CSV processing
                'has engagement survey': 'EngagementSurvey',  # This is what agent_system.py creates
                'has engagement': 'EngagementSurvey',
                'engagement survey': 'EngagementSurvey',
                'engagement': 'EngagementSurvey',
                'has engagementsurvey': 'EngagementSurvey',
                'engagementsurvey': 'EngagementSurvey',
                'engagement_survey': 'EngagementSurvey',
                # Also handle if CSV column name is stored directly as predicate (exact match)
                'has engagementsurvey': 'EngagementSurvey',  # Direct column name from CSV (no space)
                'engagementsurvey': 'EngagementSurvey',  # No prefix, exact column name
                # Handle case where column name is stored exactly as "EngagementSurvey"
                'engagementsurvey': 'EngagementSurvey',  # Exact match for column name
                # Employment status variations - CRITICAL: handle "has employment status" from CSV processing
                'has employment status': 'EmploymentStatus',  # This is what agent_system.py creates
                'employment status': 'EmploymentStatus',
                'status': 'EmploymentStatus',
                'has employmentstatus': 'EmploymentStatus',
                'employmentstatus': 'EmploymentStatus',
                'employment_status': 'EmploymentStatus',
                # Department variations
                'works in department': 'Department',
                'has department': 'Department',
                'department': 'Department',
                'has dept': 'Department',
                'dept': 'Department',
                # Position variations
                'has position': 'Position',
                'position': 'Position',
                # Absences variations
                'has absences': 'Absences',
                'absences': 'Absences',
                'absence': 'Absences',
                # Manager variations - CRITICAL: handle both ManagerName and ManagerID
                'has manager name': 'ManagerName',
                'has manager id': 'ManagerID',  # CRITICAL: This was missing!
                'has manager': 'ManagerName',
                'manager name': 'ManagerName',
                'manager id': 'ManagerID',  # CRITICAL: Handle ManagerID separately
                'manager': 'ManagerName',
                'has managername': 'ManagerName',
                'has managerid': 'ManagerID',  # CRITICAL: Handle ManagerID
                'managername': 'ManagerName',
                'managerid': 'ManagerID',  # CRITICAL: Handle ManagerID
                'manager_id': 'ManagerID',  # Handle underscore variation
                'manager_name': 'ManagerName',  # Handle underscore variation
                # Recruitment source variations
                'has recruitment source': 'RecruitmentSource',
                'recruitment source': 'RecruitmentSource',
                'recruitment': 'RecruitmentSource',
                'has recruitmentsource': 'RecruitmentSource',
                'recruitmentsource': 'RecruitmentSource',
                # Date of hire variations
                'dateofhire': 'DateofHire',
                'date of hire': 'DateofHire',
                'has dateofhire': 'DateofHire',
                'has date of hire': 'DateofHire',
                # Special projects count variations
                'specialprojectscount': 'SpecialProjectsCount',
                'special projects count': 'SpecialProjectsCount',
                'has specialprojectscount': 'SpecialProjectsCount',
                'has special projects count': 'SpecialProjectsCount',
            }
            
            # Try to find a match in the mapping (check full predicate first, then partial)
            predicate_normalized = predicate_clean.lower().strip()
            
            # First try exact match
            if predicate_normalized in predicate_mapping:
                predicate_clean = predicate_mapping[predicate_normalized]
            else:
                # Then try partial matches - be more aggressive
                matched = False
                for key, standard_name in predicate_mapping.items():
                    # Check if key is contained in predicate or vice versa
                    if key in predicate_normalized or predicate_normalized in key:
                        predicate_clean = standard_name
                        matched = True
                        break
                    # Also check word-by-word matching for multi-word predicates
                    key_words = key.split()
                    pred_words = predicate_normalized.split()
                    if len(key_words) > 1 and any(kw in pred_words for kw in key_words):
                        # If at least one word matches, try the mapping
                        if any(kw in predicate_normalized for kw in key_words):
                            predicate_clean = standard_name
                            matched = True
                            break
                
                # If no match found, try to use the original column name from predicate
                # Handle cases like "has PerfScoreID" -> "PerfScoreID" or "PerformanceScore"
                if not matched:
                    # Remove "has " prefix if present
                    if predicate_clean.lower().startswith('has '):
                        original_col = predicate_clean[4:].strip()
                        # Check if this looks like a column name (has capital letters or is all caps)
                        if original_col and (original_col[0].isupper() or original_col.isupper()):
                            # Try to map common column name variations
                            col_lower = original_col.lower()
                            if 'perf' in col_lower and ('score' in col_lower or 'id' in col_lower):
                                predicate_clean = 'PerformanceScore'
                            elif 'engagement' in col_lower:
                                predicate_clean = 'EngagementSurvey'
                            elif 'employment' in col_lower and 'status' in col_lower:
                                predicate_clean = 'EmploymentStatus'
                            elif 'department' in col_lower or 'dept' in col_lower:
                                predicate_clean = 'Department'
                            elif 'absence' in col_lower:
                                predicate_clean = 'Absences'
                            elif 'manager' in col_lower:
                                # Distinguish between ManagerName and ManagerID
                                if 'id' in col_lower or 'managerid' in col_lower:
                                    predicate_clean = 'ManagerID'
                                else:
                                    predicate_clean = 'ManagerName'
                            elif 'recruitment' in col_lower or 'source' in col_lower:
                                predicate_clean = 'RecruitmentSource'
                            else:
                                # Use the original column name as-is (capitalize properly)
                                predicate_clean = ' '.join(word.capitalize() for word in original_col.split())
                        else:
                            # Capitalize first letter of each word to match CSV column names
                            predicate_clean = ' '.join(word.capitalize() for word in predicate_clean.split())
            
            # Use cleaned predicate name (store as-is if not in mapping, might be actual column name)
            # Filter out text extraction artifacts - only keep predicates that look like CSV columns
            predicate_lower = predicate_clean.lower().strip()
            
            # Skip common text extraction verbs/relations that aren't CSV columns
            text_artifacts = ['is', 'has', 'was', 'were', 'found', 'causes', 'relates', 'contains', 
                            'discovered', 'affected', 'refers', 'associated', 'related']
            
            # Exception: "has X" patterns where X is a proper noun (likely CSV column)
            is_csv_column = False
            if predicate_lower.startswith('has ') and len(predicate_clean.split()) >= 2:
                # "has engagement survey", "has employment status" etc. are CSV columns
                is_csv_column = True
            elif predicate_clean[0].isupper() or any(word[0].isupper() for word in predicate_clean.split() if len(word) > 1):
                # Capitalized words are likely column names
                is_csv_column = True
            elif predicate_lower in ['salary', 'position', 'department', 'age', 'absences', 'performance score', 
                                    'engagement survey', 'employment status', 'manager name', 'manager id', 
                                    'recruitment source', 'perfscoreid', 'performancescore']:
                # Known CSV column patterns
                is_csv_column = True
            
            # Special handling for EngagementSurvey - check if predicate contains engagement+survey
            is_engagement_survey = False
            if 'engagement' in predicate_lower and 'survey' in predicate_lower:
                predicate_clean = 'EngagementSurvey'
                is_engagement_survey = True
                is_csv_column = True  # Force it to be stored
            
            # Only store if it looks like a CSV column or is in our mapping
            if is_csv_column or predicate_clean in predicate_mapping.values() or predicate_normalized in predicate_mapping or is_engagement_survey:
                # If it's an engagement survey variant, normalize to EngagementSurvey
                if is_engagement_survey:
                    predicate_clean = 'EngagementSurvey'
                employees[subject_clean][predicate_clean] = object_val
        
        if not employees:
            return None
        
        # Convert to DataFrame
        rows = []
        for emp_name, attrs in employees.items():
            row = {"Employee_Name": emp_name}
            row.update(attrs)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Convert numeric columns - be more aggressive about conversion
        numeric_keywords = ['score', 'salary', 'age', 'absence', 'absences', 'count', 'number', 
                          'performance', 'engagement', 'rating', 'percent', 'percentage', 'rate',
                          'tenure', 'years', 'days', 'hours', 'amount', 'value', 'total', 'sum',
                          'id', 'satisfaction', 'projects']
        
        for col in df.columns:
            if col != "Employee_Name":
                # Try direct numeric conversion first
                try:
                    # Convert to string first, then try numeric
                    df[col] = df[col].astype(str)
                    # Remove any non-numeric characters except decimal point and minus sign
                    df[col] = df[col].str.replace(r'[^\d\.\-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
                
                # If still not numeric, check if column name suggests it should be numeric
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in numeric_keywords):
                    try:
                        # Force conversion, replacing non-numeric with NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
        
        # Debug: Print DataFrame info
        elapsed_total = time.time() - start_time
        if len(df) > 0:
            print(f"✅ Reconstructed DataFrame: {len(df)} rows, {len(df.columns)} columns in {elapsed_total:.2f}s")
            print(f"   Processed {processed_count} facts, skipped {skipped_metadata} metadata triples")
            print(f"   Columns: {list(df.columns)[:10]}...")
        else:
            print(f"⚠️  Reconstructed DataFrame is empty after {elapsed_total:.2f}s")
            print(f"   Processed {processed_count} facts, skipped {skipped_metadata} metadata triples")
        
        return df
    
    except Exception as e:
        print(f"⚠️  Error reconstructing DataFrame: {e}")
        import traceback
        traceback.print_exc()
        return None


def normalize_column_name(df: pd.DataFrame, target_name: str) -> Optional[str]:
    """Find the actual column name in the dataframe that matches the target name."""
    target_lower = target_name.lower().replace(' ', '').replace('_', '')
    
    for col in df.columns:
        col_normalized = col.lower().replace(' ', '').replace('_', '')
        if target_lower in col_normalized or col_normalized in target_lower:
            return col
    
    # Try partial matches
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in target_name.lower().split()):
            return col
    
    return None


def process_strategic_query_with_agent(query_info: Dict[str, Any], question: str) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process strategic query using Strategic Query Agent.
    Uses statistics and reconstructed DataFrame from knowledge graph.
    """
    strategic_type = query_info.get("strategic_type")
    subtype = query_info.get("subtype")
    operation = query_info.get("operation")
    variables = query_info.get("variables", [])
    
    # Try to reconstruct DataFrame from knowledge graph
    df = reconstruct_dataframe_from_facts()
    
    if df is None or len(df) == 0:
        # Fallback: check if we have statistics
        all_stats = get_all_statistics()
        if not all_stats:
            return (
                "I couldn't find the dataset. Please ensure a CSV file has been uploaded and processed.",
                [],
                {"strategy": "strategic_query_agent", "reason": "No data available"}
            )
        
        # We have statistics but no DataFrame - return helpful message
        stats = all_stats[0]
        return (
            f"I found statistics for {stats.get('total_rows', 0)} rows, but need to reconstruct the data for multi-variable analysis. "
            f"Please re-upload your CSV file to enable full strategic query processing.",
            [],
            {"strategy": "strategic_query_agent", "reason": "DataFrame reconstruction needed"}
        )
    
    # Normalize column names - be more flexible with matching
    actual_columns = {}
    for var in variables:
        # Special cases for exact column matching
        exact_match_found = False
        
        # Try exact match first (case-insensitive) for all variables
        var_lower = var.lower().replace('_', '').replace(' ', '')
        for col in df.columns:
            col_normalized = col.lower().replace('_', '').replace(' ', '')
            if var_lower == col_normalized:
                actual_columns[var] = col
                print(f"✅ Found {var} (exact match): {col}")
                exact_match_found = True
                break
        
        # If not found, try normalize_column_name
        if not exact_match_found:
            actual_col = normalize_column_name(df, var)
            if actual_col:
                actual_columns[var] = actual_col
                print(f"✅ Found {var} (normalized): {actual_col}")
            else:
                # Try alternative names from mapping
                alternatives = {
                    "PerformanceScore": ["PerformanceScore", "PerfScoreID", "performance score"],
                    "EngagementSurvey": ["EngagementSurvey", "engagement survey"],
                    "EmploymentStatus": ["EmploymentStatus", "employment status"],
                    "Department": ["Department", "DeptID"],
                    "Position": ["Position", "Role"],
                    "Absences": ["Absences", "absence"],
                    "ManagerName": ["ManagerName", "Manager", "ManagerID"]
                }
                
                if var in alternatives:
                    for alt_name in alternatives[var]:
                        alt_col = normalize_column_name(df, alt_name)
                        if alt_col:
                            actual_columns[var] = alt_col
                            print(f"✅ Found {var} via alternative '{alt_name}': {alt_col}")
                            break
                
                if var not in actual_columns:
                    print(f"⚠️  Could not find column for '{var}'")
                    available_cols = ', '.join(df.columns[:15].tolist())
                    return (
                        f"I couldn't find the '{var}' column (or its variations) in the dataset. "
                        f"Available columns: {available_cols}... "
                        f"Please ensure the dataset contains columns matching: {var}",
                        [],
                        {"strategy": "strategic_query_agent", "reason": f"Column '{var}' not found"}
                    )
            # Try alternative names with more variations
            alternatives = {
                "PerformanceScore": ["performance", "performance score", "perf", "perfscore", "perfscoreid", "performance score id"],
                "EngagementSurvey": ["engagement", "engagement survey", "engagement score", "engagementsurvey", "engagement_survey", "engagement score"],
                "EmploymentStatus": ["status", "employment status", "employment", "employmentstatus", "emp status", "current status"],
                "RecruitmentSource": ["recruitment", "source", "recruitment source", "recruitmentsource", "recruitment_source", "hiring source"],
                "Absences": ["absence", "absences", "days absent", "absence count", "days_absent", "absent days"],
                "ManagerName": ["manager", "manager name", "supervisor", "managername", "manager_name", "supervisor name"],
                "Department": ["department", "dept", "division", "deptid", "department id"]
            }
            
            if var in alternatives:
                for alt in alternatives[var]:
                    actual_col = normalize_column_name(df, alt)
                    if actual_col:
                        actual_columns[var] = actual_col
                        print(f"✅ Found column '{actual_col}' for '{var}' using alternative '{alt}'")
                        break
            
            # If still not found, try direct column name match (case-insensitive)
            if var not in actual_columns:
                for col in df.columns:
                    if var.lower().replace('_', '').replace(' ', '') == col.lower().replace('_', '').replace(' ', ''):
                        actual_columns[var] = col
                        print(f"✅ Found column '{col}' for '{var}' using direct match")
                        break
            
            if var not in actual_columns:
                print(f"⚠️  Could not find column matching '{var}'")
                print(f"   Available columns: {list(df.columns)[:30]}")
                return (
                    f"I couldn't find the '{var}' column in the dataset. Available columns: {', '.join(df.columns[:15])}...",
                    [],
                    {"strategy": "strategic_query_agent", "reason": f"Column '{var}' not found"}
                )
    
    answer_parts = []
    evidence_facts = []
    
    try:
        # Import processing functions from strategic_queries
        from strategic_queries import (
            process_s1_1, process_s1_2, process_s2_1, process_s2_2,
            process_o1_1, process_o1_2, process_o2_1, process_o3_1
        )
        
        if strategic_type == "S1":
            if subtype == "S1.1":
                answer_parts, evidence_facts = process_s1_1(df, actual_columns)
            elif subtype == "S1.2":
                answer_parts, evidence_facts = process_s1_2(df, actual_columns)
        
        elif strategic_type == "S2":
            if subtype == "S2.1":
                answer_parts, evidence_facts = process_s2_1(df, actual_columns)
            elif subtype == "S2.2":
                answer_parts, evidence_facts = process_s2_2(df, actual_columns)
        
        elif strategic_type == "O1":
            if subtype == "O1.1":
                answer_parts, evidence_facts = process_o1_1(df, actual_columns)
            elif subtype == "O1.2":
                answer_parts, evidence_facts = process_o1_2(df, actual_columns)
        
        elif strategic_type == "O2":
            if subtype == "O2.1":
                answer_parts, evidence_facts = process_o2_1(df, actual_columns)
        
        elif strategic_type == "O3":
            if subtype == "O3.1":
                answer_parts, evidence_facts = process_o3_1(df, actual_columns)
        
        if answer_parts:
            answer = "\n\n".join(answer_parts)
            
            # Store insights in knowledge graph so LLM can access them later
            try:
                from knowledge import add_to_graph
                from datetime import datetime
                insight_text = f"Strategic Analysis ({strategic_type}.{subtype}): {answer}"
                add_to_graph(
                    insight_text,
                    source_document="strategic_insights",
                    uploaded_at=datetime.now().isoformat(),
                    agent_id=STRATEGIC_QUERY_AGENT_ID
                )
                # Stored strategic insight in knowledge graph
            except Exception as e:
                # Failed to store strategic insight
                pass
            
            # Extract and store statistics facts (correlations, distributions, min/max) in knowledge graph
            try:
                all_stats = get_all_statistics()
                if all_stats:
                    from agent_system import extract_statistical_facts, KG_AGENT_ID
                    for stats in all_stats:
                        doc_name = stats.get("document_name", "Unknown")
                        doc_id = stats.get("document_id", "")
                        if doc_id:
                            facts_count = extract_statistical_facts(stats, doc_name, doc_id)
                            # Statistics facts extracted and stored
            except Exception as e:
                # Failed to extract statistics facts
                pass
            
            return answer, evidence_facts, {
                "strategy": "strategic_query_agent",
                "reason": f"Processed {strategic_type} query using Strategic Query Agent",
                "query_type": strategic_type,
                "subtype": subtype,
                "data_source": "knowledge_graph"
            }
        else:
            return (
                "I processed the query but couldn't generate a complete answer. Please check if the required columns exist in the dataset.",
                evidence_facts,
                {"strategy": "strategic_query_agent", "reason": "Incomplete analysis"}
            )
    
    except Exception as e:
        print(f"⚠️  Error processing strategic query: {e}")
        import traceback
        traceback.print_exc()
        return (
            f"I encountered an error processing this strategic query: {str(e)}",
            [],
            {"strategy": "strategic_query_agent", "reason": f"Error: {str(e)}"}
        )

