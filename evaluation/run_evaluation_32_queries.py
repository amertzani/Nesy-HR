#!/usr/bin/env python3
"""
Run evaluation for exactly 32 queries as specified in ALL_TESTED_QUERIES.md
This ensures we only test the queries that are documented.
"""

import json
import os
import sys
import subprocess

# Load test scenarios
with open('test_scenarios.json', 'r') as f:
    data = json.load(f)
    scenarios = data.get('scenarios', [])

# Filter to only the scenarios we want (O1-O5, S1-S3, and specific evidence scenarios)
# Based on ALL_TESTED_QUERIES.md:
# - O1: Performance Score by Department (4 queries)
# - O2: Special Projects Count by Department (4 queries) 
# - O3: Engagement by Manager (3 queries)
# - O4: Salary by Department (3 queries)
# - O5: Performance by Recruitment Source (3 queries)
# - S1: Multi-Criteria Employee Search (4 queries)
# - S2: Department-Salary-Performance Analysis (3 queries)
# - S3: (if exists)
# - E1-E4: Evidence scenarios (8 queries total)

allowed_scenario_ids = ['O1', 'O2', 'O3', 'O4', 'O5', 'S1', 'S2', 'S3']
filtered_scenarios = [s for s in scenarios if s.get('id') in allowed_scenario_ids]

# Count queries
total_queries = sum(len(s.get('queries', [])) for s in filtered_scenarios)
print(f"📊 Found {len(filtered_scenarios)} scenarios with {total_queries} queries")

# Check if we need to add evidence scenarios
# According to ALL_TESTED_QUERIES.md, we need 8 evidence queries (25-32)
# Let's check what evidence scenarios exist
evidence_scenarios = [s for s in scenarios if s.get('id', '').startswith('E')]
print(f"📊 Found {len(evidence_scenarios)} evidence scenarios")

# Add evidence scenarios if they exist and we need them
if len(evidence_scenarios) > 0:
    # Take first 4 evidence scenarios (E1-E4) to get 8 queries
    # But we need exactly 8 queries, so let's be selective
    evidence_queries_needed = 8
    evidence_queries_found = 0
    evidence_to_add = []
    
    for ev_scenario in evidence_scenarios[:4]:  # Take first 4 evidence scenarios
        queries = ev_scenario.get('queries', [])
        if evidence_queries_found + len(queries) <= evidence_queries_needed:
            evidence_to_add.append(ev_scenario)
            evidence_queries_found += len(queries)
        elif evidence_queries_found < evidence_queries_needed:
            # Take only some queries from this scenario
            needed = evidence_queries_needed - evidence_queries_found
            ev_scenario_copy = ev_scenario.copy()
            ev_scenario_copy['queries'] = queries[:needed]
            evidence_to_add.append(ev_scenario_copy)
            evidence_queries_found += needed
            break
    
    filtered_scenarios.extend(evidence_to_add)
    print(f"📊 Added {len(evidence_to_add)} evidence scenarios with {evidence_queries_found} queries")

total_queries = sum(len(s.get('queries', [])) for s in filtered_scenarios)
print(f"✅ Total queries to test: {total_queries}")

# Now run evaluate_offline.py with these specific scenarios
# We'll modify the approach: create a temporary scenarios file with only these scenarios
temp_scenarios_file = 'test_scenarios_32_queries.json'
with open(temp_scenarios_file, 'w') as f:
    json.dump({
        'dataset': data.get('dataset', {}),
        'scenarios': filtered_scenarios
    }, f, indent=2)

print(f"✅ Created temporary scenarios file: {temp_scenarios_file}")
print(f"📝 Running evaluation for {total_queries} queries...")

# Run evaluate_offline.py with --all flag, but it will use our filtered scenarios
# Actually, we need to modify evaluate_offline.py to accept a scenarios file
# For now, let's just run it and it should work if test_scenarios.json only has these

# Backup original
import shutil
shutil.copy('test_scenarios.json', 'test_scenarios.json.backup')

# Replace with filtered
shutil.copy(temp_scenarios_file, 'test_scenarios.json')

try:
    # Run evaluation
    result = subprocess.run(
        [sys.executable, 'evaluate_offline.py', '--all'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📄 Report saved to: offline_evaluation_report.txt")
    
finally:
    # Restore original
    shutil.copy('test_scenarios.json.backup', 'test_scenarios.json')
    os.remove('test_scenarios.json.backup')
    if os.path.exists(temp_scenarios_file):
        os.remove(temp_scenarios_file)

