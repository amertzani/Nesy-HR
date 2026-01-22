#!/usr/bin/env python3
"""
Architectural Property Validation
==================================

Validates the four key architectural properties:
1. Evidence-Boundedness
2. Conservative Failure Behavior
3. Traceability & Provenance Completeness
4. Determinism & Reproducibility
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from answer_query_terminal import answer_query
    from knowledge import load_knowledge_graph, graph
    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    print(f"⚠️  Import error: {e}")


class ArchitecturalPropertyValidator:
    """Validates architectural properties of the system."""
    
    def __init__(self, test_queries: List[str]):
        self.test_queries = test_queries
        self.results = []
        
        if SYSTEM_AVAILABLE:
            try:
                if graph is None or len(graph) == 0:
                    load_knowledge_graph()
            except Exception as e:
                print(f"⚠️  Could not load knowledge graph: {e}")
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("🔍 Running architectural property validation...")
        print("=" * 80)
        
        # 1. Evidence-Boundedness
        print("\n1️⃣  VALIDATING EVIDENCE-BOUNDEDNESS...")
        evidence_results = self.validate_evidence_boundedness()
        
        # 2. Conservative Failure
        print("\n2️⃣  VALIDATING CONSERVATIVE FAILURE BEHAVIOR...")
        failure_results = self.validate_conservative_failure()
        
        # 3. Provenance Completeness
        print("\n3️⃣  VALIDATING TRACEABILITY & PROVENANCE...")
        provenance_results = self.validate_provenance_completeness()
        
        # 4. Determinism
        print("\n4️⃣  VALIDATING DETERMINISM & REPRODUCIBILITY...")
        determinism_results = self.validate_determinism()
        
        return {
            "evidence_boundedness": evidence_results,
            "conservative_failure": failure_results,
            "provenance_completeness": provenance_results,
            "determinism": determinism_results,
            "summary": self.generate_summary(
                evidence_results, failure_results, 
                provenance_results, determinism_results
            )
        }
    
    def validate_evidence_boundedness(self) -> Dict[str, Any]:
        """
        Pillar 1: Evidence-Boundedness
        
        Measures:
        - % of responses with complete evidence sets
        - % of responses with empty evidence sets (conservative failure)
        - Examples: evidence exists → answer, evidence missing → no answer
        """
        results = {
            "total_queries": len(self.test_queries),
            "responses_with_evidence": 0,
            "responses_without_evidence": 0,
            "answers_with_evidence": 0,
            "answers_without_evidence": 0,
            "examples": []
        }
        
        for query in self.test_queries:
            response = answer_query(query)
            
            facts_used = response.get("facts_used", [])
            method = response.get("method", "unknown")
            answer = response.get("answer", "")
            
            # Distinguish between KG evidence and direct computation
            has_kg_evidence = len(facts_used) > 0
            has_direct_computation = method in [
                "operational_insights_api", 
                "csv_computation",
                "operational_query"
            ]
            has_evidence = has_kg_evidence or has_direct_computation
            
            has_answer = answer and answer.strip() and answer.lower() not in [
                "none", "null", "no answer", "could not", "error"
            ]
            
            # Count evidence availability (KG evidence OR direct computation)
            if has_kg_evidence:
                results["responses_with_evidence"] += 1
            elif has_direct_computation:
                results["responses_with_evidence"] += 1  # Direct computation is also evidence-bounded
            else:
                results["responses_without_evidence"] += 1
            
            # Count answer patterns
            if has_answer and has_evidence:
                results["answers_with_evidence"] += 1
            elif has_answer and not has_evidence:
                results["answers_without_evidence"] += 1
            
            # Collect examples
            if len(results["examples"]) < 2:
                if has_evidence and has_answer:
                    results["examples"].append({
                        "type": "evidence_exists_answer_produced",
                        "query": query,
                        "answer": answer[:100] + "..." if len(answer) > 100 else answer,
                        "evidence_count": len(facts_used)
                    })
                elif not has_evidence and not has_answer:
                    results["examples"].append({
                        "type": "evidence_missing_no_answer",
                        "query": query,
                        "answer": answer or "No answer",
                        "evidence_count": 0
                    })
        
        # Calculate percentages
        total = results["total_queries"]
        results["pct_with_evidence"] = (results["responses_with_evidence"] / total * 100) if total > 0 else 0
        results["pct_without_evidence"] = (results["responses_without_evidence"] / total * 100) if total > 0 else 0
        results["pct_answers_with_evidence"] = (results["answers_with_evidence"] / total * 100) if total > 0 else 0
        
        # Print results
        print(f"   Total queries: {total}")
        print(f"   Responses with evidence: {results['responses_with_evidence']} ({results['pct_with_evidence']:.1f}%)")
        print(f"   Responses without evidence: {results['responses_without_evidence']} ({results['pct_without_evidence']:.1f}%)")
        print(f"   Answers with evidence: {results['answers_with_evidence']} ({results['pct_answers_with_evidence']:.1f}%)")
        if results["answers_without_evidence"] > 0:
            print(f"   ⚠️  Answers without evidence: {results['answers_without_evidence']} (PROBLEM!)")
        
        print("\n   Examples:")
        for ex in results["examples"]:
            print(f"   - {ex['type']}:")
            print(f"     Query: {ex['query']}")
            print(f"     Answer: {ex['answer']}")
            print(f"     Evidence facts: {ex['evidence_count']}")
        
        return results
    
    def validate_conservative_failure(self) -> Dict[str, Any]:
        """
        Pillar 2: Conservative Failure Behavior
        
        Categorizes queries into:
        - fully supported
        - partially supported
        - unsupported
        
        Shows how each category is handled.
        """
        categories = {
            "fully_supported": 0,
            "partially_supported": 0,
            "unsupported": 0
        }
        
        handling = {
            "fully_supported": {"full_answer": 0, "partial_answer": 0, "no_answer": 0},
            "partially_supported": {"full_answer": 0, "partial_answer": 0, "no_answer": 0},
            "unsupported": {"full_answer": 0, "partial_answer": 0, "no_answer": 0}
        }
        
        for query in self.test_queries:
            response = answer_query(query)
            
            facts_used = response.get("facts_used", [])
            method = response.get("method", "unknown")
            answer = response.get("answer", "")
            has_answer = answer and answer.strip() and answer.lower() not in [
                "none", "null", "no answer", "could not", "error"
            ]
            
            evidence_count = len(facts_used)
            
            # Distinguish between KG evidence and direct computation
            has_direct_computation = method in [
                "operational_insights_api", 
                "csv_computation",
                "operational_query"
            ]
            
            # Categorize
            if evidence_count >= 3 and has_answer:  # Threshold: at least 3 KG facts
                category = "fully_supported"
            elif has_direct_computation and has_answer:
                # Direct computation is also evidence-bounded (computed from source data)
                category = "fully_supported"
            elif evidence_count > 0:
                category = "partially_supported"
            else:
                category = "unsupported"
            
            categories[category] += 1
            
            # Track handling
            if has_answer:
                if len(answer) > 50 and "insufficient" not in answer.lower():
                    handling[category]["full_answer"] += 1
                else:
                    handling[category]["partial_answer"] += 1
            else:
                handling[category]["no_answer"] += 1
        
        # Print table
        print("\n   Category Distribution:")
        print(f"   {'Category':<25} {'Count':<10} {'Full Answer':<15} {'Partial':<15} {'No Answer':<15}")
        print("   " + "-" * 75)
        for cat in ["fully_supported", "partially_supported", "unsupported"]:
            count = categories[cat]
            h = handling[cat]
            print(f"   {cat:<25} {count:<10} {h['full_answer']:<15} {h['partial_answer']:<15} {h['no_answer']:<15}")
        
        # Add note about direct computation
        direct_comp_count = sum(1 for q in self.test_queries 
                                if answer_query(q).get("method") in 
                                ["operational_insights_api", "csv_computation", "operational_query"])
        if direct_comp_count > 0:
            print(f"\n   Note: {direct_comp_count} queries use direct CSV computation (evidence-bounded but not KG facts)")
        
        return {
            "categories": categories,
            "handling": handling,
            "total": len(self.test_queries)
        }
    
    def validate_provenance_completeness(self) -> Dict[str, Any]:
        """
        Pillar 3: Traceability & Provenance Completeness
        
        Measures:
        - % of facts with complete provenance (source, timestamp, pathway)
        - Example provenance chain
        """
        total_facts = 0
        facts_with_source = 0
        facts_with_timestamp = 0
        facts_with_pathway = 0
        facts_with_complete = 0
        
        example_chain = None
        
        for query in self.test_queries:
            response = answer_query(query)
            facts_used = response.get("facts_used", [])
            
            for fact in facts_used:
                total_facts += 1
                
                # Check provenance components
                has_source = self._has_source_document(fact)
                has_timestamp = self._has_timestamp(fact)
                has_pathway = self._has_pathway(fact)
                
                if has_source:
                    facts_with_source += 1
                if has_timestamp:
                    facts_with_timestamp += 1
                if has_pathway:
                    facts_with_pathway += 1
                if has_source and has_timestamp and has_pathway:
                    facts_with_complete += 1
                    # Save first complete example
                    if example_chain is None:
                        example_chain = {
                            "query": query,
                            "fact": fact,
                            "source": fact.get("source_document") or fact.get("source", "unknown"),
                            "timestamp": fact.get("timestamp") or fact.get("uploaded_at", "unknown"),
                            "pathway": fact.get("agent_id") or fact.get("source_type") or "unknown"
                        }
        
        # Calculate percentages
        pct_source = (facts_with_source / total_facts * 100) if total_facts > 0 else 0
        pct_timestamp = (facts_with_timestamp / total_facts * 100) if total_facts > 0 else 0
        pct_pathway = (facts_with_pathway / total_facts * 100) if total_facts > 0 else 0
        pct_complete = (facts_with_complete / total_facts * 100) if total_facts > 0 else 0
        
        print(f"   Total facts in responses: {total_facts}")
        print(f"   Facts with source document: {facts_with_source} ({pct_source:.1f}%)")
        print(f"   Facts with timestamp: {facts_with_timestamp} ({pct_timestamp:.1f}%)")
        print(f"   Facts with pathway: {facts_with_pathway} ({pct_pathway:.1f}%)")
        print(f"   Facts with complete provenance: {facts_with_complete} ({pct_complete:.1f}%)")
        
        if example_chain:
            print("\n   Example Provenance Chain:")
            print(f"   Query: {example_chain['query']}")
            print(f"   Fact: {example_chain['fact'].get('subject', 'N/A')} → {example_chain['fact'].get('predicate', 'N/A')} → {example_chain['fact'].get('object', 'N/A')}")
            print(f"   Source Document: {example_chain['source']}")
            print(f"   Timestamp: {example_chain['timestamp']}")
            print(f"   Processing Pathway: {example_chain['pathway']}")
        
        return {
            "total_facts": total_facts,
            "facts_with_source": facts_with_source,
            "facts_with_timestamp": facts_with_timestamp,
            "facts_with_pathway": facts_with_pathway,
            "facts_with_complete": facts_with_complete,
            "pct_complete": pct_complete,
            "example_chain": example_chain
        }
    
    def _has_source_document(self, fact: Dict) -> bool:
        """Check if fact has source document."""
        # Check multiple possible field names
        source = (fact.get("source_document") or 
                 (fact.get("source")[0][0] if isinstance(fact.get("source"), list) and len(fact.get("source", [])) > 0 else None) or
                 (fact.get("sources")[0] if isinstance(fact.get("sources"), list) and len(fact.get("sources", [])) > 0 else None))
        return source and source not in [None, "unknown", "", "manual"]
    
    def _has_timestamp(self, fact: Dict) -> bool:
        """Check if fact has timestamp."""
        timestamp = fact.get("timestamp") or fact.get("uploaded_at")
        return timestamp is not None and timestamp != "" and timestamp != "None"
    
    def _has_pathway(self, fact: Dict) -> bool:
        """Check if fact has processing pathway."""
        pathway = fact.get("agent_id") or fact.get("source_type")
        return pathway is not None and pathway != "" and pathway != "unknown"
    
    def validate_determinism(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        Pillar 4: Determinism & Reproducibility
        
        Runs each query multiple times and checks for consistency.
        """
        consistent_queries = 0
        inconsistent_queries = 0
        inconsistencies = []
        
        for query in self.test_queries:
            results = []
            for i in range(num_runs):
                response = answer_query(query)
                results.append({
                    "answer": response.get("answer", ""),
                    "facts_count": len(response.get("facts_used", [])),
                    "method": response.get("method", "unknown")
                })
            
            # Check if all results are identical
            first = results[0]
            all_identical = all(
                r["answer"] == first["answer"] and
                r["facts_count"] == first["facts_count"] and
                r["method"] == first["method"]
                for r in results
            )
            
            if all_identical:
                consistent_queries += 1
            else:
                inconsistent_queries += 1
                inconsistencies.append({
                    "query": query,
                    "runs": results
                })
        
        consistency_pct = (consistent_queries / len(self.test_queries) * 100) if self.test_queries else 0
        
        print(f"   Queries tested: {len(self.test_queries)}")
        print(f"   Runs per query: {num_runs}")
        print(f"   Consistent queries: {consistent_queries} ({consistency_pct:.1f}%)")
        print(f"   Inconsistent queries: {inconsistent_queries}")
        
        if inconsistencies:
            print("\n   ⚠️  Inconsistencies found:")
            for inc in inconsistencies[:3]:  # Show first 3
                print(f"   Query: {inc['query']}")
                for i, run in enumerate(inc['runs']):
                    print(f"     Run {i+1}: answer={run['answer'][:50]}..., facts={run['facts_count']}, method={run['method']}")
        
        return {
            "total_queries": len(self.test_queries),
            "num_runs": num_runs,
            "consistent": consistent_queries,
            "inconsistent": inconsistent_queries,
            "consistency_pct": consistency_pct,
            "inconsistencies": inconsistencies
        }
    
    def generate_summary(self, evidence, failure, provenance, determinism) -> Dict[str, Any]:
        """Generate summary of all validations."""
        return {
            "evidence_boundedness": {
                "pct_with_evidence": evidence.get("pct_with_evidence", 0),
                "pct_answers_with_evidence": evidence.get("pct_answers_with_evidence", 0)
            },
            "conservative_failure": {
                "fully_supported": failure["categories"]["fully_supported"],
                "partially_supported": failure["categories"]["partially_supported"],
                "unsupported": failure["categories"]["unsupported"]
            },
            "provenance_completeness": {
                "pct_complete": provenance.get("pct_complete", 0)
            },
            "determinism": {
                "consistency_pct": determinism.get("consistency_pct", 0)
            }
        }


def load_test_queries(query_file: Optional[str] = None) -> List[str]:
    """Load test queries from file or use defaults."""
    if query_file and os.path.exists(query_file):
        with open(query_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [q if isinstance(q, str) else q.get("query", "") for q in data]
            elif isinstance(data, dict):
                # Check for direct "queries" key
                if "queries" in data:
                    queries_list = data["queries"]
                    return [q if isinstance(q, str) else q.get("query", "") for q in queries_list]
                # Check for scenarios structure
                scenarios = data.get("scenarios", [])
                queries = []
                for scenario in scenarios:
                    for q in scenario.get("queries", []):
                        if isinstance(q, str):
                            queries.append(q)
                        elif isinstance(q, dict):
                            queries.append(q.get("query", ""))
                return queries
    
    # Default test queries
    return [
        "What is the average salary by department?",
        "Which department has the highest average salary?",
        "What is the position of Becker, Renee?",
        "Who has the maximum salary?",
    ]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Architectural Properties')
    parser.add_argument('--queries', type=str, 
                       default='evaluation/test_scenarios.json',
                       help='JSON file with test queries')
    parser.add_argument('--output', type=str,
                       default='architectural_validation_results.json',
                       help='JSON output file')
    
    args = parser.parse_args()
    
    # Load queries
    queries = load_test_queries(args.queries)
    print(f"📋 Loaded {len(queries)} test queries")
    
    # Create validator
    validator = ArchitecturalPropertyValidator(queries)
    
    # Run validation
    results = validator.validate_all()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ Validation complete!")
    print(f"📊 Results saved to: {args.output}")
    print("=" * 80)

