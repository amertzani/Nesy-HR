"""
LLM Comparison Tool
===================

This tool compares your knowledge graph-based system with LLM baselines (GPT-4, etc.)
to demonstrate your system's advantages:
- Better accuracy (grounded in facts)
- Traceability (can show evidence)
- Speed (direct KG access)
- Consistency (same query = same answer)

Usage:
    python compare_with_llm.py --scenario O1
    python compare_with_llm.py --all
    python compare_with_llm.py --llm gpt-4 --scenario O1
"""

import json
import os
import sys
import time
import argparse
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import system modules
try:
    from answer_query_terminal import answer_query
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False
    print("⚠️  Could not import answer_query_terminal. Will use subprocess instead.")


def load_test_scenarios() -> List[Dict[str, Any]]:
    """Load test scenarios with ground truth."""
    scenarios_file = "test_scenarios.json"
    if not os.path.exists(scenarios_file):
        print(f"❌ Test scenarios file not found: {scenarios_file}")
        print("   Please run: python evaluation_test_scenarios.py")
        return []
    
    with open(scenarios_file, 'r') as f:
        data = json.load(f)
        return data.get('scenarios', [])


def query_your_system(query: str) -> Dict[str, Any]:
    """
    Query your knowledge graph-based system.
    Returns response, response time, and evidence.
    """
    result = {
        "query": query,
        "response": None,
        "response_time": None,
        "evidence": [],
        "error": None,
        "method": None
    }
    
    try:
        start_time = time.time()
        
        if SYSTEM_AVAILABLE:
            # Use direct import
            answer_result = answer_query(query)
            result["response"] = answer_result.get("answer", "")
            result["method"] = answer_result.get("method", "unknown")
            result["evidence"] = answer_result.get("facts_used", [])
        else:
            # Use subprocess to call answer_query_terminal.py
            process = subprocess.run(
                [sys.executable, "answer_query_terminal.py", query],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if process.returncode == 0:
                output = process.stdout
                # Extract answer from output
                if "ANSWER" in output:
                    parts = output.split("ANSWER")
                    if len(parts) > 1:
                        answer_section = parts[1].split("=" * 80)[0]
                        result["response"] = answer_section.strip()
                else:
                    result["response"] = output.strip()
            else:
                result["error"] = process.stderr or "Unknown error"
        
        result["response_time"] = time.time() - start_time
        
    except subprocess.TimeoutExpired:
        result["error"] = "Query timed out (>60s)"
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


def query_llm(query: str, llm_type: str = "gpt-4", dataset_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Query an LLM baseline (GPT-4, Claude, etc.).
    
    Args:
        query: The query to ask
        llm_type: Which LLM to use ("gpt-4", "gpt-3.5-turbo", "claude-3", etc.)
        dataset_context: Optional context about the dataset (for fair comparison)
    """
    result = {
        "query": query,
        "response": None,
        "response_time": None,
        "error": None,
        "llm_type": llm_type
    }
    
    try:
        import openai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            result["error"] = "OPENAI_API_KEY not found in environment. Set it in .env file or export it."
            return result
        
        client = openai.OpenAI(api_key=api_key)
        
        # Build prompt
        system_prompt = """You are a data analyst assistant helping with HR analytics questions.
Answer questions accurately based on the provided context. If you don't have the exact data,
clearly state that you need access to the dataset to provide accurate answers."""
        
        user_prompt = query
        if dataset_context:
            user_prompt = f"Context about the dataset:\n{dataset_context}\n\nQuestion: {query}"
        
        # Map llm_type to model name
        model_map = {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo",
            "gpt-3.5-turbo": "gpt-3.5-turbo"
        }
        model = model_map.get(llm_type, "gpt-4-turbo-preview")
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more accurate responses
            max_tokens=500
        )
        
        result["response"] = response.choices[0].message.content.strip()
        result["response_time"] = time.time() - start_time
        
    except ImportError:
        result["error"] = "openai library not installed. Install with: pip install openai"
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text."""
    # Look for numbers (including decimals)
    patterns = [
        r'(\d+\.?\d*)',  # Simple number
        r'\$(\d+[,\d]*\.?\d*)',  # Dollar amount
        r'(\d+\.?\d*)\s*(?:absences?|employees?|score|engagement|salary)',  # With context
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                # Take the first match, clean it
                value_str = matches[0].replace(',', '')
                return float(value_str)
            except:
                continue
    
    return None


def extract_entity_name(text: str, entity_type: str = "department") -> Optional[str]:
    """Extract entity name (department, manager, etc.) from text."""
    text_lower = text.lower()
    
    if entity_type == "department":
        departments = [
            "production", "sales", "it/is", "it", "is",
            "admin offices", "admin", "executive office", "executive",
            "software engineering", "software"
        ]
        
        for dept in departments:
            if dept in text_lower:
                # Normalize
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
    
    return None


def evaluate_accuracy(
    response: str,
    ground_truth: Dict[str, Any],
    query_intent: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate accuracy of a response against ground truth.
    
    Returns:
        Dictionary with accuracy metrics
    """
    evaluation = {
        "exact_match": False,
        "numeric_accuracy": None,
        "entity_match": False,
        "correct": False,
        "error": None
    }
    
    if not response or not ground_truth:
        evaluation["error"] = "Missing response or ground truth"
        return evaluation
    
    response_lower = response.lower()
    stats = ground_truth.get("statistics", {})
    
    # Check for exact numeric matches
    if "groupby" in stats:
        # Groupby statistics (e.g., mean, median)
        groups = stats.get("groups", {})
        
        # Try to extract the answer from response
        if "highest" in response_lower or "maximum" in response_lower or "max" in response_lower:
            # Looking for max value
            numeric_value = extract_numeric_value(response)
            entity = extract_entity_name(response, "department")
            
            if numeric_value and groups:
                # Find max in ground truth
                max_value = None
                max_entity = None
                for group_name, group_stats in groups.items():
                    group_mean = group_stats.get("mean")
                    if group_mean is not None:
                        if max_value is None or group_mean > max_value:
                            max_value = group_mean
                            max_entity = group_name
                
                if max_value and numeric_value:
                    # Check if within tolerance (5% or absolute difference < 1)
                    tolerance = max(0.05 * abs(max_value), 1.0)
                    if abs(numeric_value - max_value) <= tolerance:
                        evaluation["numeric_accuracy"] = True
                        evaluation["correct"] = True
                    else:
                        evaluation["numeric_accuracy"] = False
                        evaluation["error"] = f"Expected {max_value}, got {numeric_value}"
                
                if entity and max_entity:
                    # Normalize entity names for comparison
                    entity_normalized = entity.lower().strip()
                    max_entity_normalized = max_entity.lower().strip()
                    if entity_normalized in max_entity_normalized or max_entity_normalized in entity_normalized:
                        evaluation["entity_match"] = True
                        evaluation["correct"] = True
    
    elif "crosstab" in stats:
        # Crosstab statistics
        crosstab = stats.get("crosstab", {})
        # For crosstab, we mainly check if the response mentions relevant entities
        # This is a simpler check
        for entity in crosstab.keys():
            if entity.lower() in response_lower:
                evaluation["entity_match"] = True
                break
    
    # Overall correctness
    if evaluation.get("numeric_accuracy") or evaluation.get("entity_match"):
        evaluation["correct"] = True
    
    return evaluation


def compare_responses(
    your_response: Dict[str, Any],
    llm_response: Dict[str, Any],
    ground_truth: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    """
    Compare responses from both systems.
    """
    comparison = {
        "query": query,
        "your_system": {
            "response": your_response.get("response", ""),
            "response_time": your_response.get("response_time", 0),
            "has_evidence": len(your_response.get("evidence", [])) > 0,
            "evidence_count": len(your_response.get("evidence", [])),
            "error": your_response.get("error")
        },
        "llm": {
            "response": llm_response.get("response", ""),
            "response_time": llm_response.get("response_time", 0),
            "has_evidence": False,  # LLMs don't provide traceable evidence
            "error": llm_response.get("error")
        },
        "accuracy": {
            "your_system": None,
            "llm": None
        },
        "winner": None
    }
    
    # Extract query intent (simplified)
    query_lower = query.lower()
    intent = {
        "operation": "max" if any(w in query_lower for w in ["highest", "maximum", "max", "top"]) else "average",
        "metric": None,
        "group_by": "department" if "department" in query_lower else "manager" if "manager" in query_lower else None
    }
    
    # Evaluate accuracy
    if your_response.get("response") and not your_response.get("error"):
        comparison["accuracy"]["your_system"] = evaluate_accuracy(
            your_response["response"],
            ground_truth,
            intent
        )
    
    if llm_response.get("response") and not llm_response.get("error"):
        comparison["accuracy"]["llm"] = evaluate_accuracy(
            llm_response["response"],
            ground_truth,
            intent
        )
    
    # Determine winner
    your_correct = comparison["accuracy"]["your_system"] and comparison["accuracy"]["your_system"].get("correct", False)
    llm_correct = comparison["accuracy"]["llm"] and comparison["accuracy"]["llm"].get("correct", False)
    
    if your_correct and not llm_correct:
        comparison["winner"] = "your_system"
    elif llm_correct and not your_correct:
        comparison["winner"] = "llm"
    elif your_correct and llm_correct:
        # Both correct - compare on other factors
        if comparison["your_system"]["has_evidence"]:
            comparison["winner"] = "your_system"  # Evidence is a tie-breaker
        elif comparison["your_system"]["response_time"] < comparison["llm"]["response_time"]:
            comparison["winner"] = "your_system"  # Faster
        else:
            comparison["winner"] = "tie"
    else:
        comparison["winner"] = "neither"
    
    return comparison


def generate_comparison_report(
    comparisons: List[Dict[str, Any]],
    output_file: str = "llm_comparison_report.txt"
) -> str:
    """Generate a comprehensive comparison report."""
    report = []
    report.append("=" * 80)
    report.append("KNOWLEDGE GRAPH SYSTEM vs LLM BASELINE COMPARISON")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary statistics
    total = len(comparisons)
    your_correct = sum(1 for c in comparisons if c["accuracy"]["your_system"] and c["accuracy"]["your_system"].get("correct", False))
    llm_correct = sum(1 for c in comparisons if c["accuracy"]["llm"] and c["accuracy"]["llm"].get("correct", False))
    your_wins = sum(1 for c in comparisons if c["winner"] == "your_system")
    llm_wins = sum(1 for c in comparisons if c["winner"] == "llm")
    ties = sum(1 for c in comparisons if c["winner"] == "tie")
    
    your_avg_time = sum(c["your_system"]["response_time"] for c in comparisons) / total if total > 0 else 0
    llm_avg_time = sum(c["llm"]["response_time"] for c in comparisons) / total if total > 0 else 0
    
    your_evidence_count = sum(c["your_system"]["evidence_count"] for c in comparisons)
    
    report.append("SUMMARY STATISTICS")
    report.append("-" * 80)
    report.append(f"Total queries compared: {total}")
    report.append("")
    report.append("Accuracy:")
    report.append(f"  Your System: {your_correct}/{total} ({your_correct/total*100:.1f}%)")
    report.append(f"  LLM Baseline: {llm_correct}/{total} ({llm_correct/total*100:.1f}%)")
    report.append("")
    report.append("Wins:")
    report.append(f"  Your System: {your_wins}")
    report.append(f"  LLM Baseline: {llm_wins}")
    report.append(f"  Ties: {ties}")
    report.append("")
    report.append("Response Times:")
    report.append(f"  Your System: {your_avg_time:.2f}s average")
    report.append(f"  LLM Baseline: {llm_avg_time:.2f}s average")
    report.append(f"  Speedup: {llm_avg_time/your_avg_time:.2f}x faster" if your_avg_time > 0 else "")
    report.append("")
    report.append("Evidence/Traceability:")
    report.append(f"  Your System: {your_evidence_count} facts provided across all queries")
    report.append(f"  LLM Baseline: 0 (no traceable evidence)")
    report.append("")
    
    # Key Advantages
    report.append("KEY ADVANTAGES OF YOUR SYSTEM")
    report.append("-" * 80)
    advantages = []
    if your_correct > llm_correct:
        advantages.append(f"✓ Higher accuracy ({your_correct/total*100:.1f}% vs {llm_correct/total*100:.1f}%)")
    if your_avg_time < llm_avg_time:
        advantages.append(f"✓ Faster responses ({your_avg_time:.2f}s vs {llm_avg_time:.2f}s)")
    if your_evidence_count > 0:
        advantages.append(f"✓ Provides traceable evidence ({your_evidence_count} facts)")
    if your_wins > llm_wins:
        advantages.append(f"✓ Wins more comparisons ({your_wins} vs {llm_wins})")
    
    if advantages:
        for adv in advantages:
            report.append(adv)
    else:
        report.append("(Run more comparisons to identify advantages)")
    report.append("")
    
    # Detailed comparisons
    report.append("DETAILED COMPARISONS")
    report.append("-" * 80)
    for i, comp in enumerate(comparisons, 1):
        report.append(f"\n[{i}] Query: {comp['query']}")
        report.append("")
        
        # Your system
        report.append("Your System:")
        if comp["your_system"]["error"]:
            report.append(f"  ❌ Error: {comp['your_system']['error']}")
        else:
            report.append(f"  ✅ Response ({comp['your_system']['response_time']:.2f}s):")
            response_preview = comp["your_system"]["response"][:200] + "..." if len(comp["your_system"]["response"]) > 200 else comp["your_system"]["response"]
            report.append(f"     {response_preview}")
            if comp["your_system"]["has_evidence"]:
                report.append(f"  📊 Evidence: {comp['your_system']['evidence_count']} facts")
            if comp["accuracy"]["your_system"]:
                acc = comp["accuracy"]["your_system"]
                if acc.get("correct"):
                    report.append("  ✓ Correct")
                else:
                    report.append(f"  ✗ Incorrect: {acc.get('error', 'Unknown error')}")
        
        report.append("")
        
        # LLM
        report.append("LLM Baseline:")
        if comp["llm"]["error"]:
            report.append(f"  ❌ Error: {comp['llm']['error']}")
        else:
            report.append(f"  ✅ Response ({comp['llm']['response_time']:.2f}s):")
            response_preview = comp["llm"]["response"][:200] + "..." if len(comp["llm"]["response"]) > 200 else comp["llm"]["response"]
            report.append(f"     {response_preview}")
            if comp["accuracy"]["llm"]:
                acc = comp["accuracy"]["llm"]
                if acc.get("correct"):
                    report.append("  ✓ Correct")
                else:
                    report.append(f"  ✗ Incorrect: {acc.get('error', 'Unknown error')}")
        
        report.append("")
        report.append(f"Winner: {comp['winner'].upper()}")
        report.append("-" * 80)
    
    report_str = "\n".join(report)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(report_str)
    
    print(f"✅ Comparison report saved to {output_file}")
    return report_str


def main():
    parser = argparse.ArgumentParser(
        description="Compare your knowledge graph system with LLM baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare a specific scenario
  python compare_with_llm.py --scenario O1
  
  # Compare all scenarios
  python compare_with_llm.py --all
  
  # Use specific LLM
  python compare_with_llm.py --llm gpt-4 --scenario O1
  
  # Compare with limited queries (faster)
  python compare_with_llm.py --scenario O1 --max-queries 2
        """
    )
    
    parser.add_argument("--scenario", help="Test specific scenario (e.g., O1, S1)")
    parser.add_argument("--all", action="store_true", help="Test all scenarios")
    parser.add_argument("--llm", default="gpt-4", choices=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"], help="LLM to compare with")
    parser.add_argument("--max-queries", type=int, help="Maximum number of queries to test per scenario")
    parser.add_argument("--output", default="llm_comparison_report.txt", help="Output file for report")
    
    args = parser.parse_args()
    
    # Load scenarios
    scenarios = load_test_scenarios()
    if not scenarios:
        return
    
    # Filter scenarios
    if args.scenario:
        scenarios = [s for s in scenarios if s.get('id') == args.scenario]
        if not scenarios:
            print(f"❌ Scenario '{args.scenario}' not found")
            return
    
    if not args.scenario and not args.all:
        print("ℹ️  No scenario specified. Use --scenario <ID> or --all")
        print("\nAvailable scenarios:")
        for s in scenarios:
            print(f"  {s['id']}: {s['name']}")
        return
    
    print(f"🔬 Comparing your system with {args.llm}")
    print(f"📊 Testing {len(scenarios)} scenario(s)")
    print()
    
    all_comparisons = []
    
    # Test each scenario
    for scenario in scenarios:
        scenario_id = scenario.get('id')
        scenario_name = scenario.get('name')
        queries = scenario.get('queries', [])
        ground_truth = scenario.get('ground_truth', {})
        
        print(f"\n{'='*80}")
        print(f"Scenario {scenario_id}: {scenario_name}")
        print(f"{'='*80}")
        
        if args.max_queries:
            queries = queries[:args.max_queries]
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query}")
            
            # Query your system
            print("  🔍 Querying your system...")
            your_response = query_your_system(query)
            
            if your_response.get("error"):
                print(f"  ❌ Error: {your_response['error']}")
            else:
                print(f"  ✅ Response received ({your_response.get('response_time', 0):.2f}s)")
            
            # Query LLM
            print(f"  🤖 Querying {args.llm}...")
            llm_response = query_llm(query, llm_type=args.llm)
            
            if llm_response.get("error"):
                print(f"  ❌ Error: {llm_response['error']}")
            else:
                print(f"  ✅ Response received ({llm_response.get('response_time', 0):.2f}s)")
            
            # Compare
            comparison = compare_responses(your_response, llm_response, ground_truth, query)
            all_comparisons.append(comparison)
            
            # Show winner
            winner = comparison.get("winner", "unknown")
            if winner == "your_system":
                print(f"  🏆 Winner: Your System")
            elif winner == "llm":
                print(f"  🏆 Winner: {args.llm}")
            elif winner == "tie":
                print(f"  🤝 Tie")
            else:
                print(f"  ❓ Neither system got it correct")
            
            # Small delay to avoid rate limits
            time.sleep(1)
    
    # Generate report
    print("\n" + "=" * 80)
    print("Generating comparison report...")
    report = generate_comparison_report(all_comparisons, args.output)
    
    # Save JSON results
    json_file = args.output.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        json.dump({
            "comparisons": all_comparisons,
            "generated_at": datetime.now().isoformat(),
            "llm_type": args.llm
        }, f, indent=2, default=str)
    
    print(f"✅ JSON results saved to {json_file}")
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nView the full report: {args.output}")


if __name__ == "__main__":
    main()

