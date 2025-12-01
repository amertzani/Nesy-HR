"""
Orchestrator Agent - Coordinates queries and delegates to appropriate agents
"""

from typing import List, Dict, Any, Optional, Tuple
from agent_system import document_agents, agents_store, ORCHESTRATOR_AGENT_ID
import re


def find_agent_for_employee(employee_name: str) -> List[str]:
    """
    Find which document/worker agents have information about a specific employee.
    Returns list of agent IDs that processed this employee.
    """
    matching_agents = []
    
    # Normalize employee name for matching
    employee_normalized = employee_name.strip()
    
    for agent_id, agent in document_agents.items():
        # Check if agent has this employee in its tracked list
        if hasattr(agent, 'employee_names') and agent.employee_names:
            for tracked_name in agent.employee_names:
                # Exact match
                if tracked_name.lower() == employee_normalized.lower():
                    matching_agents.append(agent_id)
                    break
                # Partial match (handle variations)
                tracked_parts = [p.strip() for p in tracked_name.split(',')]
                query_parts = [p.strip() for p in employee_normalized.split(',')]
                if len(tracked_parts) >= 2 and len(query_parts) >= 2:
                    if tracked_parts[0].lower() == query_parts[0].lower() and \
                       tracked_parts[1].lower() == query_parts[1].lower():
                        matching_agents.append(agent_id)
                        break
    
    return matching_agents


def find_agents_for_query(query: str, query_type: str, attribute: Optional[str] = None, query_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Determine which agents should handle a query.
    Returns routing information for the orchestrator.
    """
    routing_info = {
        "query_type": query_type,
        "target_agents": [],
        "strategy": "all_agents",  # "all_agents", "specific_agent", "statistics_only", etc.
        "reason": ""
    }
    
    query_lower = query.lower()
    
    # For employee-specific queries, find the relevant agent
    if query_type == "filter" and attribute:
        # Extract employee name from query
        name_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',
            r'(?:of|for)\s+([A-Z][a-z]+,\s*[A-Z][a-z]+)',
        ]
        
        employee_name = None
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                employee_name = match.group(1) if match.lastindex >= 1 else match.group(0)
                break
        
        if employee_name:
            matching_agents = find_agent_for_employee(employee_name)
            if matching_agents:
                routing_info["target_agents"] = matching_agents
                routing_info["strategy"] = "specific_agents"
                routing_info["reason"] = f"Found employee '{employee_name}' in {len(matching_agents)} agent(s)"
            else:
                routing_info["strategy"] = "all_agents"
                routing_info["reason"] = f"Employee '{employee_name}' not found in tracked agents, querying all document workers"
        else:
            routing_info["strategy"] = "all_agents"
            routing_info["reason"] = "Could not extract employee name from query, querying all document workers"
    
    # For min/max queries, need to query all document workers to find global min/max
    elif query_type == "structured" and query_type in ["max", "min"]:
        routing_info["strategy"] = "all_agents"
        routing_info["reason"] = f"{query_type.upper()} query requires checking all document workers for global comparison"
        routing_info["target_agents"] = [aid for aid in document_agents.keys() 
                                         if document_agents[aid].type in ["document", "document_worker"]]
    
    # For statistics queries, route to statistics agent
    # Check for correlation, distribution, min/max, and statistical analysis keywords
    statistics_keywords = [
        "correlation", "correlate", "correlated", "correlates",
        "distribution", "distribute", "distributed", "distributions",
        "min", "minimum", "max", "maximum", "range",
        "statistic", "statistics", "statistical", "statistically",
        "average", "mean", "median", "mode", "standard deviation", "std",
        "quartile", "percentile", "variance", "spread",
        "relationship between", "relationship of", "related to",
        "how are", "how do", "connection between", "connection of"
    ]
    
    if any(keyword in query_lower for keyword in statistics_keywords):
        routing_info["strategy"] = "statistics_agent"
        routing_info["target_agents"] = ["statistics_agent"]
        routing_info["reason"] = "Query requires statistical analysis (correlations, distributions, or min/max values)"
    
    # For visualization queries, route to visualization agent
    elif "visual" in query_lower or "chart" in query_lower or "graph" in query_lower or "plot" in query_lower:
        routing_info["strategy"] = "visualization_agent"
        routing_info["target_agents"] = ["visualization_agent"]
        routing_info["reason"] = "Query requires visualization"
    
    # Default: query all document agents
    else:
        routing_info["target_agents"] = [aid for aid in document_agents.keys() 
                                         if document_agents[aid].type in ["document", "document_worker"]]
        routing_info["reason"] = "Default: query all document agents"
    
    return routing_info


def orchestrate_query(query: str, query_info: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Orchestrate a query by routing it to appropriate agents.
    Returns (answer, evidence_facts, routing_info)
    """
    orchestrator = agents_store.get(ORCHESTRATOR_AGENT_ID)
    if not orchestrator:
        print("⚠️  Orchestrator agent not found, falling back to direct query")
        return None, [], {}
    
    orchestrator.status = "processing"
    
    try:
        # Determine routing strategy (pass query_info for operation detection)
        routing_info = find_agents_for_query(
            query, 
            query_info.get("query_type", "general"),
            query_info.get("attribute"),
            query_info  # Pass full query_info for operation detection
        )
        routing_info["query_info"] = query_info
        
        
        # Simple keyword-based routing: Check for "operational" or "strategic" keywords first
        # This takes priority over complex pattern matching
        query_type = query_info.get("query_type")
        
        # Route to Operational Query Agent if keyword "operational" is detected
        if query_type == "operational":
            try:
                from operational_query_agent import process_operational_query_with_agent, OPERATIONAL_QUERY_AGENT_ID
                operational_agent = agents_store.get(OPERATIONAL_QUERY_AGENT_ID)
                if operational_agent:
                    operational_agent.status = "processing"
                    answer, evidence_facts, op_routing = process_operational_query_with_agent(query_info, query)
                    operational_agent.status = "active"
                    routing_info.update(op_routing)
                    routing_info["target_agents"] = [OPERATIONAL_QUERY_AGENT_ID]
                    routing_info["strategy"] = "operational_agent"
                    routing_info["reason"] = "Routed to operational agent based on keyword detection"
                    orchestrator.status = "active"
                    return answer, evidence_facts, routing_info
                else:
                    print("⚠️  Operational Query Agent not found")
            except ImportError as e:
                print(f"⚠️  Operational query agent module not available: {e}")
            except Exception as e:
                print(f"⚠️  Error processing operational query: {e}")
                import traceback
                traceback.print_exc()
        
        # Route to Strategic Query Agent if keyword "strategic" is detected
        elif query_type == "strategic":
            # Keyword-based routing for strategic queries
            try:
                from strategic_query_agent import process_strategic_query_with_agent, STRATEGIC_QUERY_AGENT_ID
                strategic_agent = agents_store.get(STRATEGIC_QUERY_AGENT_ID)
                if strategic_agent:
                    strategic_agent.status = "processing"
                    answer, evidence_facts, strat_routing = process_strategic_query_with_agent(query_info, query)
                    strategic_agent.status = "active"
                    routing_info.update(strat_routing)
                    routing_info["target_agents"] = [STRATEGIC_QUERY_AGENT_ID]
                    routing_info["strategy"] = "strategic_agent"
                    routing_info["reason"] = "Routed to strategic agent based on keyword detection"
                    orchestrator.status = "active"
                    return answer, evidence_facts, routing_info
                else:
                    print("⚠️  Strategic Query Agent not found")
            except ImportError as e:
                print(f"⚠️  Strategic query agent module not available: {e}")
            except Exception as e:
                print(f"⚠️  Error processing strategic query: {e}")
                import traceback
                traceback.print_exc()
        
        # For structured queries, use query processor but with agent-aware extraction
        if query_info.get("query_type") == "structured":
            from query_processor import process_structured_query
            # extract_employee_facts_from_agents is defined in this file (orchestrator.py)
            
            # Extract facts from specific agents if routing found them
            if routing_info["strategy"] == "specific_agents" and routing_info["target_agents"]:
                employees = extract_employee_facts_from_agents(routing_info["target_agents"])
            else:
                # Query all document workers
                employees = extract_employee_facts_from_agents(list(document_agents.keys()))
            
            if employees:
                # Process query with agent-filtered data
                answer, evidence_facts = process_structured_query_with_employees(query, query_info, employees)
                orchestrator.status = "active"
                return answer, evidence_facts, routing_info
        
        # For other queries, delegate to appropriate agent
        if routing_info["strategy"] == "statistics_agent":
            # Get statistics context for LLM
            try:
                from agent_system import format_statistics_context_for_llm
                from strategic_query_agent import get_all_statistics
                all_stats = get_all_statistics()
                if all_stats:
                    stats_context = format_statistics_context_for_llm(query, all_stats)
                    routing_info["statistics_context"] = stats_context
                    routing_info["has_statistics"] = True
            except Exception as e:
                print(f"⚠️  Error getting statistics context: {e}")
                routing_info["has_statistics"] = False
            
            # Delegate to statistics agent - LLM will handle with statistics context
            orchestrator.status = "active"
            return None, [], routing_info
        
        orchestrator.status = "active"
        return None, [], routing_info
        
    except Exception as e:
        print(f"⚠️  Orchestrator error: {e}")
        import traceback
        traceback.print_exc()
        orchestrator.status = "active"
        return None, [], {}


def extract_employee_facts_from_agents(agent_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Extract employee facts from specific agents' data ranges.
    More efficient than querying entire graph.
    """
    from knowledge import graph, get_fact_source_document
    from urllib.parse import unquote
    import re
    
    if graph is None:
        return []
    
    employees = {}
    
    # Get source documents for these agents
    agent_documents = {}
    for agent_id in agent_ids:
        if agent_id in document_agents:
            agent = document_agents[agent_id]
            doc_name = agent.document_name
            if doc_name:
                agent_documents[doc_name] = agent_id
    
    # Extract facts from graph, filtering by source document
    for s, p, o in graph:
        # Skip metadata triples
        predicate_str = str(p)
        if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
            'fact_object' in predicate_str or 'has_details' in predicate_str or 
            'source_document' in predicate_str or 'uploaded_at' in predicate_str or
            'is_inferred' in predicate_str or 'confidence' in predicate_str or
            'agent_id' in predicate_str):
            continue
        
        # Check if fact belongs to one of our target agents
        sources = get_fact_source_document(
            unquote(str(s).split(':')[-1] if ':' in str(s) else str(s)).replace('_', ' '),
            unquote(str(p).split(':')[-1] if ':' in str(p) else str(p)).replace('_', ' '),
            str(o)
        )
        
        fact_belongs_to_agent = False
        for source_doc, _ in sources:
            if source_doc in agent_documents:
                fact_belongs_to_agent = True
                break
        
        if not fact_belongs_to_agent:
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
        
        # Check if this is an employee-related fact
        employee_name = None
        if re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', subject):
            employee_name = subject
        elif re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', object_val):
            employee_name = object_val
        else:
            name_match = re.search(r'([A-Z][a-z]+,\s*[A-Z][a-z]+)', subject)
            if name_match:
                employee_name = name_match.group(1)
            else:
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
            
            attr_name = predicate.lower()
            employees[employee_name]["attributes"][attr_name] = object_val
            employees[employee_name]["facts"].append({
                "subject": subject,
                "predicate": predicate,
                "object": object_val,
                "source": sources
            })
    
    return list(employees.values())


def process_structured_query_with_employees(query: str, query_info: Dict[str, Any], 
                                           employees: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Process structured query with pre-filtered employee list.
    Same logic as process_structured_query but uses provided employees list.
    """
    from query_processor import process_structured_query
    
    # Temporarily replace extract_employee_facts to use our filtered list
    # This is a workaround - in production, we'd refactor process_structured_query
    # to accept employees as parameter
    
    if not employees:
        return None, []
    
    operation = query_info.get("operation")
    attribute = query_info.get("attribute")
    entity_name = query_info.get("entity_name")
    
    evidence_facts = []
    
    if operation == "min" and attribute:
        min_value = None
        min_employee = None
        
        for emp in employees:
            # Try different attribute variations
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
                
                # Check partial match
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
                    evidence_facts = [f for f in emp["facts"] 
                                    if attribute.lower() in f["predicate"].lower() or 
                                       (matched_attr and matched_attr in f["predicate"].lower())]
        
        if min_employee:
            return min_employee['name'], evidence_facts
    
    elif operation == "max" and attribute:
        max_value = None
        max_employee = None
        
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
                    evidence_facts = [f for f in emp["facts"] 
                                    if attribute.lower() in f["predicate"].lower() or 
                                       (matched_attr and matched_attr in f["predicate"].lower())]
        
        if max_employee:
            return max_employee['name'], evidence_facts
    
    elif operation == "filter" and entity_name and attribute:
        # Find employee
        matching_employee = None
        for emp in employees:
            emp_name = emp["name"]
            if emp_name.lower() == entity_name.lower():
                matching_employee = emp
                break
            
            emp_parts = [p.strip() for p in emp_name.split(',')]
            query_parts = [p.strip() for p in entity_name.split(',')]
            if len(emp_parts) >= 2 and len(query_parts) >= 2:
                if emp_parts[0].lower() == query_parts[0].lower() and emp_parts[1].lower() == query_parts[1].lower():
                    matching_employee = emp
                    break
        
        if matching_employee:
            attr_variations = [
                attribute.lower(),
                f"has_{attribute.lower()}",
                f"{attribute.lower()}_is",
                attribute.lower().replace(' ', '_'),
            ]
            
            for attr_var in attr_variations:
                if attr_var in matching_employee["attributes"]:
                    attr_value = matching_employee["attributes"][attr_var]
                    evidence_facts = [f for f in matching_employee["facts"] 
                                    if attribute.lower() in f["predicate"].lower() or 
                                       attr_var in f["predicate"].lower()]
                    return str(attr_value), evidence_facts
            
            # Try partial match
            for attr_key in matching_employee["attributes"].keys():
                if attribute.lower() in attr_key.lower() or attr_key.lower() in attribute.lower():
                    attr_value = matching_employee["attributes"][attr_key]
                    evidence_facts = [f for f in matching_employee["facts"] 
                                    if attribute.lower() in f["predicate"].lower()]
                    return str(attr_value), evidence_facts
    
    return None, []

