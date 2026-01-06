"""
Run Evaluation Tests
====================

This script tests the system with the generated scenarios and compares
results with ground truth from the dataset.

Author: Research Brain Team
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Add the current directory to path to import system modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from responses import respond
    from query_processor import detect_query_type
    from orchestrator import orchestrate_query
except ImportError as e:
    print(f"⚠️  Warning: Could not import system modules: {e}")
    print("   This script requires the system to be running or modules to be available")
    respond = None


def load_test_scenarios() -> List[Dict[str, Any]]:
    """Load test scenarios from JSON file."""
    scenarios_file = "test_scenarios.json"
    if not os.path.exists(scenarios_file):
        print(f"❌ Test scenarios file not found: {scenarios_file}")
        print("   Please run evaluation_test_scenarios.py first")
        return []
    
    with open(scenarios_file, 'r') as f:
        data = json.load(f)
        return data.get('scenarios', [])


def test_query(query: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a single query through the system.
    Returns the system's response and metadata.
    """
    result = {
        "query": query,
        "scenario_id": scenario.get('id'),
        "variables": scenario.get('variables', []),
        "response": None,
        "response_time": None,
        "error": None,
        "query_type": None,
        "routing_info": None
    }
    
    if respond is None:
        result["error"] = "System modules not available"
        return result
    
    try:
        import time
        start_time = time.time()
        
        # Detect query type
        query_info = detect_query_type(query)
        result["query_type"] = query_info.get("query_type")
        
        # Get response from system
        response, evidence, routing_info = orchestrate_query(query, query_info)
        
        result["response"] = response
        result["routing_info"] = routing_info
        result["response_time"] = time.time() - start_time
        
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


def evaluate_results(test_results: List[Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate test results and compute metrics.
    """
    evaluation = {
        "total_queries": len(test_results),
        "successful_queries": 0,
        "failed_queries": 0,
        "average_response_time": 0,
        "by_scenario": {},
        "by_type": {"operational": 0, "strategic": 0}
    }
    
    response_times = []
    
    for result in test_results:
        scenario_id = result.get("scenario_id")
        
        if result.get("error"):
            evaluation["failed_queries"] += 1
        else:
            evaluation["successful_queries"] += 1
            if result.get("response_time"):
                response_times.append(result["response_time"])
        
        # Group by scenario
        if scenario_id not in evaluation["by_scenario"]:
            evaluation["by_scenario"][scenario_id] = {
                "total": 0,
                "successful": 0,
                "failed": 0
            }
        
        evaluation["by_scenario"][scenario_id]["total"] += 1
        if result.get("error"):
            evaluation["by_scenario"][scenario_id]["failed"] += 1
        else:
            evaluation["by_scenario"][scenario_id]["successful"] += 1
    
    if response_times:
        evaluation["average_response_time"] = sum(response_times) / len(response_times)
        evaluation["min_response_time"] = min(response_times)
        evaluation["max_response_time"] = max(response_times)
    
    return evaluation


def generate_evaluation_report(test_results: List[Dict[str, Any]], 
                               scenarios: List[Dict[str, Any]],
                               evaluation: Dict[str, Any],
                               output_file: str = "evaluation_results.txt"):
    """Generate a comprehensive evaluation report."""
    report = []
    report.append("=" * 80)
    report.append("SYSTEM EVALUATION RESULTS")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total queries tested: {evaluation['total_queries']}")
    report.append(f"Successful: {evaluation['successful_queries']} ({evaluation['successful_queries']/evaluation['total_queries']*100:.1f}%)")
    report.append(f"Failed: {evaluation['failed_queries']} ({evaluation['failed_queries']/evaluation['total_queries']*100:.1f}%)")
    
    if evaluation.get('average_response_time'):
        report.append(f"Average response time: {evaluation['average_response_time']:.2f}s")
        report.append(f"Min response time: {evaluation['min_response_time']:.2f}s")
        report.append(f"Max response time: {evaluation['max_response_time']:.2f}s")
    report.append("")
    
    # Results by scenario
    report.append("RESULTS BY SCENARIO")
    report.append("-" * 80)
    for scenario_id, stats in evaluation['by_scenario'].items():
        scenario = next((s for s in scenarios if s['id'] == scenario_id), None)
        scenario_name = scenario['name'] if scenario else scenario_id
        success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        report.append(f"\n{scenario_id}: {scenario_name}")
        report.append(f"  Total queries: {stats['total']}")
        report.append(f"  Successful: {stats['successful']} ({success_rate:.1f}%)")
        report.append(f"  Failed: {stats['failed']}")
    report.append("")
    
    # Detailed results
    report.append("DETAILED RESULTS")
    report.append("-" * 80)
    for result in test_results:
        scenario_id = result.get("scenario_id", "Unknown")
        query = result.get("query", "")
        
        report.append(f"\n[Scenario {scenario_id}]")
        report.append(f"Query: {query}")
        report.append(f"Query Type: {result.get('query_type', 'Unknown')}")
        
        if result.get("error"):
            report.append(f"❌ Error: {result['error']}")
        else:
            report.append(f"✅ Response received")
            if result.get("response_time"):
                report.append(f"   Response time: {result['response_time']:.2f}s")
            if result.get("routing_info"):
                routing = result['routing_info']
                report.append(f"   Routing: {routing.get('strategy', 'Unknown')}")
                if routing.get('reason'):
                    report.append(f"   Reason: {routing['reason']}")
            
            # Show first 200 chars of response
            response = result.get("response", "")
            if response:
                preview = response[:200] + "..." if len(response) > 200 else response
                report.append(f"   Response preview: {preview}")
    
    report_str = "\n".join(report)
    
    with open(output_file, 'w') as f:
        f.write(report_str)
    
    print(f"✅ Evaluation report saved to {output_file}")
    return report_str


def main():
    """Main function to run evaluation tests."""
    print("🧪 Running system evaluation tests...")
    print()
    
    # Load scenarios
    scenarios = load_test_scenarios()
    if not scenarios:
        return
    
    print(f"✅ Loaded {len(scenarios)} test scenarios")
    print()
    
    # Collect all queries
    all_queries = []
    for scenario in scenarios:
        for query in scenario.get('queries', []):
            all_queries.append((query, scenario))
    
    print(f"📝 Testing {len(all_queries)} queries...")
    print()
    
    # Test each query
    test_results = []
    for i, (query, scenario) in enumerate(all_queries, 1):
        print(f"[{i}/{len(all_queries)}] Testing: {query[:60]}...")
        result = test_query(query, scenario)
        test_results.append(result)
        
        if result.get("error"):
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Response received ({result.get('response_time', 0):.2f}s)")
    
    print()
    
    # Evaluate results
    evaluation = evaluate_results(test_results, scenarios)
    
    # Generate report
    report = generate_evaluation_report(test_results, scenarios, evaluation)
    print()
    print(report)
    
    # Save results as JSON
    results_json = {
        "test_results": test_results,
        "evaluation": evaluation,
        "tested_at": datetime.now().isoformat()
    }
    
    with open("evaluation_results.json", 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print("✅ Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()

