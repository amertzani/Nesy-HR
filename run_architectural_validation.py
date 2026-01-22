#!/usr/bin/env python3
"""
Example script to run architectural validation of trustworthiness.

Usage:
    python run_architectural_validation.py
    python run_architectural_validation.py --queries evaluation/test_scenarios.json
"""

import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architectural_validation_framework import ArchitecturalTrustworthinessValidator, load_test_queries


def load_from_test_scenarios(scenarios_file: str) -> list:
    """Load queries from test_scenarios.json format."""
    with open(scenarios_file, 'r') as f:
        data = json.load(f)
    
    queries = []
    for scenario in data.get('scenarios', []):
        for query_data in scenario.get('queries', []):
            queries.append({
                'query': query_data.get('query', ''),
                'expected_answer': query_data.get('expected_answer')
            })
    
    return queries


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Architectural Validation')
    parser.add_argument('--queries', type=str, 
                       default='evaluation/test_scenarios.json',
                       help='Path to test scenarios JSON file')
    parser.add_argument('--output', type=str,
                       default='architectural_validation_report.txt',
                       help='Output report file')
    parser.add_argument('--json', type=str,
                       default='architectural_validation_results.json',
                       help='JSON output file')
    
    args = parser.parse_args()
    
    # Load queries
    if os.path.exists(args.queries):
        print(f"📂 Loading queries from {args.queries}...")
        queries = load_from_test_scenarios(args.queries)
    else:
        print(f"⚠️  File not found: {args.queries}")
        print("📝 Using default test queries...")
        queries = load_test_queries()
    
    print(f"🔍 Validating {len(queries)} queries...")
    print("")
    
    # Create validator
    validator = ArchitecturalTrustworthinessValidator(queries)
    
    # Run validation
    print("Running architectural validation...")
    results = validator.validate_all()
    
    # Generate report
    print("\nGenerating report...")
    report = validator.generate_report(args.output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if 'error' not in results:
        tc = results['traceability_completeness']
        auc = results['absence_of_unsupported_claims']
        correct = results['task_correctness']
        lat = results['latency']
        
        print(f"\n✅ Traceability Completeness: {tc['mean']:.3f} ± {tc['std']:.3f}")
        print(f"✅ Absence of Unsupported Claims: {auc['mean']:.3f} ± {auc['std']:.3f}")
        print(f"✅ Task Correctness: {correct['mean']:.3f} ({correct['correct_count']}/{len(correct['scores'])} correct)")
        print(f"✅ Average Latency: {lat['mean']:.3f}s")
        
        print(f"\n📄 Full report saved to: {args.output}")
    
    # Save JSON
    with open(args.json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📊 JSON results saved to: {args.json}")
    
    print("\n" + "=" * 80)
    print("Architectural validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

