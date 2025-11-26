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
    filter_patterns = [
        r'(?:what|which).*?(?:is|are).*?(?:the|a).*?(position|salary|department|age|name|status).*?(?:of|for).*?([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?)',
        r'([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?).*?(?:has|have).*?(?:a|an|the).*?(position|salary|department|age|status)',
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
    """
    if graph is None:
        print("⚠️  Knowledge graph is None")
        return []
    
    employees = {}  # employee_name -> {attributes}
    fact_count = 0
    
    for s, p, o in graph:
        fact_count += 1
        # Skip metadata triples
        predicate_str = str(p)
        if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
            'fact_object' in predicate_str or 'has_details' in predicate_str or 
            'source_document' in predicate_str or 'uploaded_at' in predicate_str or
            'is_inferred' in predicate_str or 'confidence' in predicate_str or
            'agent_id' in predicate_str):
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
            employees[employee_name]["attributes"][attr_name] = object_val
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


def process_structured_query(question: str, query_info: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Process structured queries using KG facts.
    Returns (answer, evidence_facts)
    """
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
            attr_variations = [
                attribute.lower(),
                f"has_{attribute.lower()}",
                f"{attribute.lower()}_is",
                attribute.lower().replace(' ', '_'),
            ]
            
            attr_value = None
            for attr_var in attr_variations:
                if attr_var in matching_employee["attributes"]:
                    attr_value = matching_employee["attributes"][attr_var]
                    # Collect evidence facts
                    evidence_facts = [f for f in matching_employee["facts"] 
                                    if attribute.lower() in f["predicate"].lower() or 
                                       attr_var in f["predicate"].lower()]
                    break
            
            if attr_value:
                answer = str(attr_value)
                return answer, evidence_facts
    
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
            for source_doc, uploaded_at in sources:
                if source_doc:
                    source_list.append(source_doc)
            if source_list:
                fact_line += f" [Source: {', '.join(source_list[:2])}]"
        
        context_lines.append(fact_line)
    
    return "\n".join(context_lines)



