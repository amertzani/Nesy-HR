"""
Enhanced query processor for structured queries with KG evidence.
Handles queries like "what is the employee name with the max salary" or 
"what is the position of Beak, Kimberly" with traceable evidence.
"""

try:
    from knowledge import graph, get_fact_source_document
except ImportError:
    # Handle case where knowledge module isn't available
    graph = None
    def get_fact_source_document(*args, **kwargs):
        return []
from urllib.parse import unquote
import re
from typing import List, Dict, Tuple, Optional, Any


def detect_query_type(question: str) -> Dict[str, Any]:
    """
    Detect the type of query and extract relevant information.
    Returns a dictionary with query_type and extracted parameters.
    """
    question_lower = question.lower()
    
    # Simple keyword-based routing: Check for "operational" or "strategic" keywords first
    # This takes priority over complex pattern matching
    if "operational" in question_lower or "opererational" in question_lower or "operation" in question_lower:  # Handle typo
        return {
            "query_type": "operational",
            "operation": None,
            "attribute": None,
            "entity_name": None,
            "filter_value": None,
        }
    
    if "strategic" in question_lower or "strrategic" in question_lower or "strategy" in question_lower:
        return {
            "query_type": "strategic",
            "operation": None,
            "attribute": None,
            "entity_name": None,
            "filter_value": None,
        }
    
    # If no operational/strategic keywords found, continue with structured query detection
    
    # Check for average/statistics queries FIRST (before filter patterns to avoid false matches)
    # "what is the average salary in the department of Admin Offices?"
    # "what is the average performance score per manager?"
    average_patterns = [
        r'(?:what|which|how).*?(?:is|are).*?(?:the|a).*?(?:average|avg|mean).*?(?:salary|performance|score|absence|engagement|satisfaction).*?(?:in|of|for|per|by).*?(?:department|manager|position|role)',
        r'(?:average|avg|mean).*?(?:salary|performance|score|absence|engagement|satisfaction).*?(?:in|of|for|per|by).*?(?:department|manager|position|role)',
        r'(?:what|which|how).*?(?:is|are).*?(?:the|a).*?(?:average|avg|mean).*?(?:of|for).*?(?:department|manager|position|role)',
    ]
    
    for pattern in average_patterns:
        if re.search(pattern, question_lower):
            return {
                "query_type": "statistics",
                "operation": "average",
                "attribute": None,
                "entity_name": None,
                "filter_value": None,
            }
    
    # Pattern 1: Max/Min queries
    # "what is the employee name with the max salary"
    # "who has the highest salary"
    # "which employee has the minimum age"
    max_patterns = [
        r'(?:what|which|who).*?(?:employee|person|name).*?(?:with|has|having).*?(?:max|maximum|highest|top).*?(salary|age|tenure|performance)',
        r'(?:max|maximum|highest|top).*?(salary|age|tenure|performance).*?(?:employee|person|name)',
        r'(?:employee|person|name).*?(?:with|has|having).*?(?:max|maximum|highest|top).*?(salary|age|tenure|performance)',
        r'(?:what|which|who).*?(?:is|are).*?(?:the|a).*?(?:name|employee|person).*?(?:with|has|having).*?(?:max|maximum|highest|top).*?(salary|age|tenure|performance)',
        r'(?:name|employee|person).*?(?:with|has|having).*?(?:max|maximum|highest|top).*?(salary|age|tenure|performance)',
    ]
    
    min_patterns = [
        r'(?:what|which|who).*?(?:employee|person|name).*?(?:with|has|having).*?(?:min|minimum|lowest|bottom).*?(salary|age|tenure|performance)',
        r'(?:min|minimum|lowest|bottom).*?(salary|age|tenure|performance).*?(?:employee|person|name)',
        r'(?:employee|person|name).*?(?:with|has|having).*?(?:min|minimum|lowest|bottom).*?(salary|age|tenure|performance)',
        r'(?:what|which|who).*?(?:is|are).*?(?:the|a).*?(?:name|employee|person).*?(?:with|has|having).*?(?:min|minimum|lowest|bottom).*?(salary|age|tenure|performance)',
        r'(?:name|employee|person).*?(?:with|has|having).*?(?:min|minimum|lowest|bottom).*?(salary|age|tenure|performance)',
    ]
    
    # Pattern 2: Filter queries (find attribute of specific entity)
    # "what is the position of Beak, Kimberly"
    # "what is the salary of Becker, Renee"
    # "who is the manager of Boutwell, Bonalyn"
    filter_patterns = [
        r'(?:what|which|who).*?(?:is|are).*?(?:the|a).*?(position|position\s+id|positionid|salary|department|age|name|status|absences|marital|gender|state|city|zip|phone|email|manager|manager\s+id|managerid).*?(?:of|for).*?([A-Z][a-z]+(?:,\s*[A-Z][a-z\.]+)?)',
        r'([A-Z][a-z]+(?:,\s*[A-Z][a-z\.]+)?).*?(?:has|have).*?(?:a|an|the).*?(position|position\s+id|positionid|salary|department|age|status|absences|marital|gender|state|city|zip|phone|email|manager|manager\s+id|managerid)',
        r'(?:what|which|who).*?(?:is|are).*?([A-Z][a-z]+(?:,\s*[A-Z][a-z\.]+)?).*?(?:salary|position|position\s+id|positionid|department|age|status|absences|manager|manager\s+id|managerid)',
        r'(?:who|what).*?(?:is|are).*?(?:the|a).*?(?:manager|manager\s+id|managerid).*?(?:of|for).*?([A-Z][a-z]+(?:,\s*[A-Z][a-z\.]+)?)',
        # Direct pattern for "X of Y" format (e.g., "position id of Becker, Scott")
        r'(position\s+id|positionid|manager\s+id|managerid|salary|position|department|age|status|absences|marital|gender|state|city|zip|phone|email|manager)\s+of\s+([A-Z][a-z]+(?:,\s*[A-Z][a-z\.]+)?)',
    ]
    
    # Pattern 3: Count/Aggregation queries
    # "how many employees are in Sales"
    # "how many employees have salary > 50000"
    count_patterns = [
        r'how many.*?(?:employee|person).*?(?:are|have|has).*?(?:in|with|having)',
    ]
    
    # Pattern 4: List queries
    # "list all employees in Sales"
    # "show me all employees with salary > 100000"
    list_patterns = [
        r'(?:list|show|give).*?(?:all|every).*?(?:employee|person)',
    ]
    
    query_info = {
        "query_type": "general",
        "operation": None,  # "max", "min", "filter", "count", "list"
        "attribute": None,  # "salary", "position", "age", etc.
        "entity_name": None,  # "Beak, Kimberly", "Becker, Renee"
        "filter_value": None,  # For filtering queries
    }
    
    # Check for max queries
    for pattern in max_patterns:
        match = re.search(pattern, question_lower)
        if match:
            query_info["query_type"] = "structured"
            query_info["operation"] = "max"
            query_info["attribute"] = match.group(1) if match.lastindex >= 1 else None
            return query_info
    
    # Check for min queries
    for pattern in min_patterns:
        match = re.search(pattern, question_lower)
        if match:
            query_info["query_type"] = "structured"
            query_info["operation"] = "min"
            query_info["attribute"] = match.group(1) if match.lastindex >= 1 else None
            return query_info
    
    # Check for filter queries (find attribute of specific entity)
    for pattern in filter_patterns:
        match = re.search(pattern, question)
        if match:
            query_info["query_type"] = "structured"
            query_info["operation"] = "filter"
            if match.lastindex >= 2:
                query_info["attribute"] = match.group(1)
                query_info["entity_name"] = match.group(2)
            elif match.lastindex >= 1:
                # Try to extract entity name and attribute from the match
                groups = match.groups()
                if len(groups) >= 2:
                    query_info["attribute"] = groups[0]
                    query_info["entity_name"] = groups[1]
            return query_info
    
    # Check for count queries
    for pattern in count_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "structured"
            query_info["operation"] = "count"
            return query_info
    
    # Check for list queries
    for pattern in list_patterns:
        if re.search(pattern, question_lower):
            query_info["query_type"] = "structured"
            query_info["operation"] = "list"
            return query_info
    
    return query_info


def extract_employee_facts() -> List[Dict[str, Any]]:
    """
    Extract all employee-related facts from the knowledge graph.
    Returns a list of structured employee records.
    Optimized for large graphs by skipping metadata triples early.
    """
    if graph is None:
        print("⚠️  Knowledge graph is None")
        return []
    
    employees = {}  # employee_name -> {attributes}
    fact_count = 0
    skipped_metadata = 0
    
    for s, p, o in graph:
        fact_count += 1
        
        # Early skip for metadata triples (most common, skip before processing)
        predicate_str = str(p)
        if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
            'fact_object' in predicate_str or 'has_details' in predicate_str or 
            'source_document' in predicate_str or 'uploaded_at' in predicate_str or
            'is_inferred' in predicate_str or 'confidence' in predicate_str or
            'agent_id' in predicate_str):
            skipped_metadata += 1
            continue
        
        # Extract subject
        subject_uri_str = str(s)
        if 'urn:entity:' in subject_uri_str:
            subject = subject_uri_str.split('urn:entity:')[-1]
        elif 'urn:' in subject_uri_str:
            subject = subject_uri_str.split('urn:')[-1]
        else:
            subject = subject_uri_str
        subject = unquote(subject).replace('_', ' ')
        
        # Extract predicate
        predicate_uri_str = str(p)
        if 'urn:predicate:' in predicate_uri_str:
            predicate = predicate_uri_str.split('urn:predicate:')[-1]
        elif 'urn:' in predicate_uri_str:
            predicate = predicate_uri_str.split('urn:')[-1]
        else:
            predicate = predicate_uri_str
        predicate = unquote(predicate).replace('_', ' ')
        
        # Object is already a literal
        object_val = str(o)
        
        # Check if this is an employee-related fact
        # Look for employee names (capitalized, comma-separated names)
        # Or facts where subject contains employee-related keywords
        employee_name = None
        
        # Pattern 1: Subject is an employee name (e.g., "Becker, Renee")
        if re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', subject):
            employee_name = subject
        # Pattern 2: Object is an employee name
        elif re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', object_val):
            employee_name = object_val
        # Pattern 3: Subject contains "Employee" or "employee"
        elif 'employee' in subject.lower() and len(subject.split()) <= 5:
            # Try to extract name from predicate or object
            if re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', object_val):
                employee_name = object_val
            elif re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', predicate):
                employee_name = predicate
        
        # Pattern 4: Facts like "Employee Name has salary X" or "Name is employee"
        if not employee_name:
            # Check if predicate suggests employee attribute
            employee_attributes = ['salary', 'position', 'department', 'age', 'name', 'status', 
                                  'employeeid', 'empid', 'maritalstatus', 'gender', 'dept']
            if any(attr in predicate.lower() for attr in employee_attributes):
                # Subject might be employee name
                if re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', subject):
                    employee_name = subject
                # Or object might be employee name
                elif re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', object_val):
                    employee_name = object_val
        
        # Pattern 5: Check if subject contains employee name pattern anywhere
        if not employee_name:
            name_match = re.search(r'([A-Z][a-z]+,\s*[A-Z][a-z]+)', subject)
            if name_match:
                employee_name = name_match.group(1)
        
        # Pattern 6: Check object for employee name pattern
        if not employee_name:
            name_match = re.search(r'([A-Z][a-z]+,\s*[A-Z][a-z]+)', object_val)
            if name_match:
                employee_name = name_match.group(1)
        
        if employee_name:
            if employee_name not in employees:
                employees[employee_name] = {
                    "name": employee_name,
                    "attributes": {},
                    "facts": []
                }
            
            # Store attribute
            attr_name = predicate.lower()
            
            # CRITICAL FIX: Don't overwrite existing attributes - keep the FIRST value
            # This prevents later facts from overwriting correct manager information
            # Only set if attribute doesn't exist yet, or if current value is invalid/empty
            if attr_name not in employees[employee_name]["attributes"]:
                employees[employee_name]["attributes"][attr_name] = object_val
            elif not employees[employee_name]["attributes"][attr_name] or employees[employee_name]["attributes"][attr_name].strip() == "":
                # Only overwrite if current value is empty/invalid
                employees[employee_name]["attributes"][attr_name] = object_val
            # Otherwise, keep the first (original) value
            
            employees[employee_name]["facts"].append({
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": get_fact_source_document(subject, predicate, object_val)
            })
        
        # Also check for facts where we can infer employee relationships
        # e.g., "Record 1: Employee Becker, Renee | salary is 62506"
        if 'record' in subject.lower() or 'employee' in subject.lower():
            # Try to extract employee name from object or predicate
            if re.search(r'[A-Z][a-z]+,\s*[A-Z][a-z]+', object_val):
                name_match = re.search(r'([A-Z][a-z]+,\s*[A-Z][a-z]+)', object_val)
                if name_match:
                    employee_name = name_match.group(1)
                    if employee_name not in employees:
                        employees[employee_name] = {
                            "name": employee_name,
                            "attributes": {},
                            "facts": []
                        }
                    # Extract attribute value from object
                    if 'salary' in predicate.lower() or 'salary' in object_val.lower():
                        salary_match = re.search(r'(\d+)', object_val)
                        if salary_match:
                            employees[employee_name]["attributes"]["salary"] = salary_match.group(1)
                    if 'position' in predicate.lower() or 'position' in object_val.lower():
                        # Extract position name
                        position_match = re.search(r'position.*?is\s+([^|]+)', object_val, re.IGNORECASE)
                        if position_match:
                            employees[employee_name]["attributes"]["position"] = position_match.group(1).strip()
    
    print(f"📊 Extracted {len(employees)} employees from {fact_count} facts in knowledge graph")
    if employees:
        sample_emp = list(employees.values())[0]
        print(f"   Sample employee: {sample_emp['name']} with {len(sample_emp['attributes'])} attributes: {list(sample_emp['attributes'].keys())[:5]}")
    
    return list(employees.values())


def extract_single_employee_facts(employee_name: str) -> List[Dict[str, Any]]:
    """
    Extract facts for a single employee only (much faster than extracting all).
    Optimized for filter queries - uses RDFLib's triples() method for efficient lookup.
    """
    if graph is None:
        return []
    
    employee_normalized = employee_name.strip()
    employees = {}
    fact_count = 0
    skipped_metadata = 0
    
    # Normalize name parts for matching
    name_parts = [p.strip() for p in employee_normalized.split(',')]
    
    # Pre-compute normalized patterns for faster matching
    query_normalized = employee_normalized.lower().replace(' ', '').replace(',', ',')
    last_name_lower = name_parts[0].lower().strip() if len(name_parts) >= 2 else ''
    first_name_lower = name_parts[1].lower().strip() if len(name_parts) >= 2 else ''
    
    # Try to use RDFLib's efficient triples() method with subject pattern
    # Try multiple formats to find the employee
    from rdflib import Literal, URIRef
    from urllib.parse import quote, unquote
    
    matching_triples = []
    
    # Format 1: Literal with exact name
    subject_literal = Literal(employee_normalized)
    matching_triples = list(graph.triples((subject_literal, None, None)))
    
    # Format 2: URI with spaces replaced by underscores
    if not matching_triples:
        subject_clean = employee_normalized.replace(' ', '_')
        subject_uri = URIRef(f"urn:entity:{quote(subject_clean, safe='')}")
        matching_triples = list(graph.triples((subject_uri, None, None)))
    
    # Format 3: URI without quote encoding
    if not matching_triples:
        subject_uri2 = URIRef(f"urn:entity:{subject_clean}")
        matching_triples = list(graph.triples((subject_uri2, None, None)))
    
    # Format 4: Try with quotes removed (e.g., "Costello, Frank" -> Costello, Frank)
    if not matching_triples:
        # Try removing quotes if present
        unquoted_name = employee_normalized
        if employee_normalized.startswith('"') and employee_normalized.endswith('"'):
            unquoted_name = employee_normalized[1:-1].strip()
        elif employee_normalized.startswith("'") and employee_normalized.endswith("'"):
            unquoted_name = employee_normalized[1:-1].strip()
        
        if unquoted_name != employee_normalized:
            subject_literal2 = Literal(unquoted_name)
            matching_triples = list(graph.triples((subject_literal2, None, None)))
        
        # Also try with quotes added (in case KG has quotes but query doesn't)
        if not matching_triples:
            quoted_name = f'"{employee_normalized}"'
            subject_literal3 = Literal(quoted_name)
            matching_triples = list(graph.triples((subject_literal3, None, None)))
    
    # Format 5: Try with different case
    if not matching_triples:
        # Try with first letter of each name capitalized differently
        parts = employee_normalized.split(',')
        if len(parts) == 2:
            last_name = parts[0].strip()
            first_name = parts[1].strip()
            # Try various capitalizations
            variations = [
                f"{last_name}, {first_name}",
                f"{last_name.title()}, {first_name.title()}",
                f"{last_name.upper()}, {first_name.upper()}",
            ]
            for variant in variations:
                variant_literal = Literal(variant)
                variant_triples = list(graph.triples((variant_literal, None, None)))
                if variant_triples:
                    matching_triples = variant_triples
                    break
    
    # If still no matches, use LIMITED iteration with early exit (max 10,000 facts checked)
    if not matching_triples:
        print(f"⚠️  No direct triples() match, using LIMITED iteration (max 10k facts)...")
        max_iterations = 10000  # Limit to prevent timeout
        for s, p, o in graph:
            fact_count += 1
            
            # Early skip for metadata triples
            predicate_str = str(p)
            if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
                'fact_object' in predicate_str or 'has_details' in predicate_str or 
                'source_document' in predicate_str or 'uploaded_at' in predicate_str or
                'is_inferred' in predicate_str or 'confidence' in predicate_str or
                'agent_id' in predicate_str):
                skipped_metadata += 1
                continue
            
            # Fast subject match check
            subject_uri_str = str(s)
            subject_uri_lower = subject_uri_str.lower()
            
            # Quick literal match check
            if len(name_parts) >= 2:
                subject_normalized = subject_uri_lower.replace(' ', '').replace(',', ',')
                if query_normalized == subject_normalized:
                    matching_triples.append((s, p, o))
                elif last_name_lower in subject_uri_lower and first_name_lower in subject_uri_lower:
                    comma_idx = subject_uri_lower.find(',')
                    if comma_idx > 0:
                        subject_last = subject_uri_lower[:comma_idx].strip()
                        subject_first = subject_uri_lower[comma_idx+1:].strip()
                        if last_name_lower in subject_last and first_name_lower in subject_first:
                            matching_triples.append((s, p, o))
            elif employee_normalized.lower() in subject_uri_lower:
                matching_triples.append((s, p, o))
            
            # Early exit: if we've found enough facts, stop iterating
            if len(matching_triples) > 50:  # Reduced from 100 for faster response
                break
            
            # Hard limit: don't iterate more than max_iterations
            if fact_count >= max_iterations:
                print(f"⚠️  Reached iteration limit ({max_iterations}), stopping search")
                break
    
    # Process matching triples
    for s, p, o in matching_triples:
        fact_count += 1
        
        # Skip metadata triples
        predicate_str = str(p)
        if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
            'fact_object' in predicate_str or 'has_details' in predicate_str or 
            'source_document' in predicate_str or 'uploaded_at' in predicate_str or
            'is_inferred' in predicate_str or 'confidence' in predicate_str or
            'agent_id' in predicate_str):
            skipped_metadata += 1
            continue
        
        # Extract subject, predicate, object
        subject_uri_str = str(s)
        if 'urn:entity:' in subject_uri_str:
            subject = subject_uri_str.split('urn:entity:')[-1]
        elif 'urn:' in subject_uri_str:
            subject = subject_uri_str.split('urn:')[-1]
        else:
            subject = subject_uri_str
        subject = unquote(subject).replace('_', ' ')
        
        predicate_uri_str = str(p)
        if 'urn:predicate:' in predicate_uri_str:
            predicate = predicate_uri_str.split('urn:predicate:')[-1]
        elif 'urn:' in predicate_uri_str:
            predicate = predicate_uri_str.split('urn:')[-1]
        else:
            predicate = predicate_uri_str
        predicate = unquote(predicate).replace('_', ' ')
        
        object_val = str(o)
        
        # Found matching employee fact
        if employee_normalized not in employees:
            employees[employee_normalized] = {
                "name": employee_normalized,
                "attributes": {},
                "facts": []
            }
        
        attr_name = predicate.lower()
        
        # CRITICAL FIX: Don't overwrite existing attributes - keep the FIRST value
        # This prevents later facts from overwriting correct manager information
        if attr_name not in employees[employee_normalized]["attributes"]:
            employees[employee_normalized]["attributes"][attr_name] = object_val
        elif not employees[employee_normalized]["attributes"][attr_name] or employees[employee_normalized]["attributes"][attr_name].strip() == "":
            # Only overwrite if current value is empty/invalid
            employees[employee_normalized]["attributes"][attr_name] = object_val
        # Otherwise, keep the first (original) value
        
        employees[employee_normalized]["facts"].append({
            "subject": subject,
            "predicate": predicate,
            "object": object_val,
            "source": []  # Skip expensive source lookup for filter queries
        })
    
    print(f"📊 Extracted facts for {employee_normalized}: {len(employees.get(employee_normalized, {}).get('attributes', {}))} attributes from {len(matching_triples)} triples (skipped {skipped_metadata} metadata)")
    return list(employees.values())


def process_structured_query(question: str, query_info: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Process structured queries using KG facts.
    Returns (answer, evidence_facts)
    Optimized: For filter queries, only extract the specific employee needed.
    """
    operation = query_info.get("operation")
    entity_name = query_info.get("entity_name")
    
    # For filter queries, extract only the specific employee (much faster)
    if operation == "filter" and entity_name:
        print(f"🔍 Filter query: extracting only {entity_name} (optimized)")
        employees = extract_single_employee_facts(entity_name)
    else:
        # For max/min queries, need all employees
        print(f"🔍 {operation.upper()} query: extracting all employees...")
        employees = extract_employee_facts()
    
    if not employees:
        print("⚠️  No employees found in knowledge graph")
        return None, []
    
    print(f"🔍 Processing {query_info.get('operation')} query on {len(employees)} employees")
    
    operation = query_info.get("operation")
    attribute = query_info.get("attribute")
    entity_name = query_info.get("entity_name")
    
    evidence_facts = []
    
    if operation == "max":
        # Find employee with maximum value of attribute
        if not attribute:
            print("⚠️  No attribute specified for max query")
            return None, []
        
        print(f"🔍 Looking for employee with maximum {attribute}")
        max_value = None
        max_employee = None
        
        for emp in employees:
            # Try different attribute name variations
            attr_variations = [
                attribute.lower(),
                f"has_{attribute.lower()}",
                f"{attribute.lower()}_is",
                attribute.lower().replace(' ', '_'),
                f"has {attribute.lower()}",
                f"{attribute.lower()} is",
            ]
            
            emp_value = None
            matched_attr = None
            for attr_var in attr_variations:
                # Check exact match
                if attr_var in emp["attributes"]:
                    matched_attr = attr_var
                    try:
                        emp_value = float(emp["attributes"][attr_var])
                        break
                    except (ValueError, TypeError):
                        # Try to extract number from string
                        value_str = str(emp["attributes"][attr_var])
                        num_match = re.search(r'(\d+\.?\d*)', value_str)
                        if num_match:
                            emp_value = float(num_match.group(1))
                            break
                # Check partial match (e.g., "salary" in "has salary")
                for attr_key in emp["attributes"].keys():
                    if attribute.lower() in attr_key.lower() or attr_key.lower() in attribute.lower():
                        matched_attr = attr_key
                        try:
                            emp_value = float(emp["attributes"][attr_key])
                            break
                        except (ValueError, TypeError):
                            value_str = str(emp["attributes"][attr_key])
                            num_match = re.search(r'(\d+\.?\d*)', value_str)
                            if num_match:
                                emp_value = float(num_match.group(1))
                                break
                if emp_value is not None:
                    break
            
            if emp_value is not None:
                if max_value is None or emp_value > max_value:
                    max_value = emp_value
                    max_employee = emp
                    # Collect evidence facts
                    evidence_facts = [f for f in emp["facts"] if attribute.lower() in f["predicate"].lower() or (matched_attr and matched_attr in f["predicate"].lower())]
                    print(f"   Found candidate: {emp['name']} with {attribute}={emp_value} (matched attr: {matched_attr})")
        
        if max_employee:
            answer = f"{max_employee['name']}"
            print(f"✅ Max {attribute} employee: {answer} (value: {max_value})")
            return answer, evidence_facts
        else:
            print(f"⚠️  No employee found with {attribute} attribute")
    
    elif operation == "min":
        # Find employee with minimum value of attribute
        if not attribute:
            print("⚠️  No attribute specified for min query")
            return None, []
        
        print(f"🔍 Looking for employee with minimum {attribute}")
        min_value = None
        min_employee = None
        
        for emp in employees:
            attr_variations = [
                attribute.lower(),
                f"has_{attribute.lower()}",
                f"{attribute.lower()}_is",
                attribute.lower().replace(' ', '_'),
                f"has {attribute.lower()}",
                f"{attribute.lower()} is",
            ]
            
            emp_value = None
            matched_attr = None
            for attr_var in attr_variations:
                # Check exact match
                if attr_var in emp["attributes"]:
                    matched_attr = attr_var
                    try:
                        emp_value = float(emp["attributes"][attr_var])
                        break
                    except (ValueError, TypeError):
                        value_str = str(emp["attributes"][attr_var])
                        num_match = re.search(r'(\d+\.?\d*)', value_str)
                        if num_match:
                            emp_value = float(num_match.group(1))
                            break
                # Check partial match (e.g., "salary" in "has salary")
                for attr_key in emp["attributes"].keys():
                    if attribute.lower() in attr_key.lower() or attr_key.lower() in attribute.lower():
                        matched_attr = attr_key
                        try:
                            emp_value = float(emp["attributes"][attr_key])
                            break
                        except (ValueError, TypeError):
                            value_str = str(emp["attributes"][attr_key])
                            num_match = re.search(r'(\d+\.?\d*)', value_str)
                            if num_match:
                                emp_value = float(num_match.group(1))
                                break
                if emp_value is not None:
                    break
            
            if emp_value is not None:
                if min_value is None or emp_value < min_value:
                    min_value = emp_value
                    min_employee = emp
                    # Collect evidence facts
                    evidence_facts = [f for f in emp["facts"] if attribute.lower() in f["predicate"].lower() or (matched_attr and matched_attr in f["predicate"].lower())]
                    print(f"   Found candidate: {emp['name']} with {attribute}={emp_value} (matched attr: {matched_attr})")
        
        if min_employee:
            answer = f"{min_employee['name']}"
            print(f"✅ Min {attribute} employee: {answer} (value: {min_value})")
            return answer, evidence_facts
        else:
            print(f"⚠️  No employee found with {attribute} attribute")
    
    elif operation == "filter":
        # Find attribute value for specific employee
        if not entity_name or not attribute:
            return None, []
        
        # Normalize entity name (handle variations)
        entity_name_normalized = entity_name.strip()
        
        # Find matching employee (case-insensitive, handle variations)
        matching_employee = None
        for emp in employees:
            emp_name = emp["name"]
            # Exact match
            if emp_name.lower() == entity_name_normalized.lower():
                matching_employee = emp
                break
            # Partial match (last name, first name)
            emp_parts = [p.strip() for p in emp_name.split(',')]
            query_parts = [p.strip() for p in entity_name_normalized.split(',')]
            if len(emp_parts) >= 2 and len(query_parts) >= 2:
                if emp_parts[0].lower() == query_parts[0].lower() and emp_parts[1].lower() == query_parts[1].lower():
                    matching_employee = emp
                    break
        
        if matching_employee:
            # Find attribute value
            # Special handling for manager queries - map to correct predicate
            if attribute.lower() in ['manager', 'managerid']:
                attr_variations = [
                    "has manager name",  # Primary predicate for ManagerName
                    "has manager id",    # Primary predicate for ManagerID
                    "has managername",
                    "has managerid",
                    "manager name",
                    "manager id",
                    "managername",
                    "managerid",
                    "manager",
                    "has manager",
                ]
            else:
                attr_variations = [
                    attribute.lower(),
                    f"has_{attribute.lower()}",
                    f"{attribute.lower()}_is",
                    attribute.lower().replace(' ', '_'),
                    f"has {attribute.lower()}",  # Add space-separated version
                ]
            
            attr_value = None
            matched_attr = None
            for attr_var in attr_variations:
                if attr_var in matching_employee["attributes"]:
                    matched_attr = attr_var
                    attr_value = matching_employee["attributes"][attr_var]
                    # Collect evidence facts
                    evidence_facts = [f for f in matching_employee["facts"] 
                                    if attribute.lower() in f["predicate"].lower() or 
                                       attr_var in f["predicate"].lower() or
                                       "manager" in f["predicate"].lower()]
                    break
            
            # Try partial match if exact match failed
            if not attr_value:
                for attr_key in matching_employee["attributes"].keys():
                    # For manager queries, check for any manager-related key
                    if attribute.lower() in ['manager', 'managerid']:
                        if 'manager' in attr_key.lower():
                            matched_attr = attr_key
                            attr_value = matching_employee["attributes"][attr_key]
                            evidence_facts = [f for f in matching_employee["facts"] 
                                            if "manager" in f["predicate"].lower()]
                            break
                    elif attribute.lower() in attr_key.lower() or attr_key.lower() in attribute.lower():
                        matched_attr = attr_key
                        attr_value = matching_employee["attributes"][attr_key]
                        evidence_facts = [f for f in matching_employee["facts"] 
                                        if attribute.lower() in f["predicate"].lower() or 
                                           attr_key.lower() in f["predicate"].lower()]
                        break
            
            if attr_value:
                answer = str(attr_value)
                print(f"✅ Found {attribute} for {entity_name}: {answer} (matched attr: {matched_attr})")
                return answer, evidence_facts
            else:
                print(f"⚠️  Could not find {attribute} for {entity_name}. Available attributes: {list(matching_employee['attributes'].keys())[:10]}")
    
    return None, []


def build_evidence_context(evidence_facts: List[Dict[str, Any]], question: str) -> str:
    """
    Build a traceable evidence context from facts.
    """
    if not evidence_facts:
        return ""
    
    context_lines = ["**Evidence from Knowledge Graph:**"]
    
    for i, fact in enumerate(evidence_facts, 1):
        subj = fact.get("subject", "")
        pred = fact.get("predicate", "")
        obj = fact.get("object", "")
        sources = fact.get("source", [])
        
        fact_line = f"{i}. {subj} → {pred} → {obj}"
        
        if sources:
            source_list = []
            # Handle different source formats:
            # 1. List of strings: ["operational_query", "strategic_query"]
            # 2. List of tuples: [("document.csv", "2024-01-01"), ...]
            # 3. Mixed or other formats
            for source_item in sources:
                if isinstance(source_item, tuple):
                    # Tuple format: (source_doc, uploaded_at) or more elements
                    if len(source_item) >= 1:
                        source_doc = source_item[0]
                        if source_doc:
                            source_list.append(str(source_doc))
                elif isinstance(source_item, (list, tuple)):
                    # Nested list/tuple
                    if len(source_item) >= 1:
                        source_list.append(str(source_item[0]))
                else:
                    # String or other format
                    source_list.append(str(source_item))
            
            if source_list:
                # Remove duplicates and limit to 2
                unique_sources = list(dict.fromkeys(source_list))[:2]
                fact_line += f" [Source: {', '.join(unique_sources)}]"
        
        context_lines.append(fact_line)
    
    return "\n".join(context_lines)



