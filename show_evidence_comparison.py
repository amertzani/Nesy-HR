"""
Evidence Comparison Tool
========================

This tool demonstrates the key advantage of your system: TRACEABILITY.
It shows how your system can provide evidence (facts from knowledge graph)
while LLMs cannot.

Usage:
    python show_evidence_comparison.py "Which department has the highest average salary?"
"""

import sys
import os
import argparse
import json
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from answer_query_terminal import answer_query
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False
    print("⚠️  Could not import answer_query_terminal")


def show_evidence_comparison(query: str):
    """Show side-by-side comparison with evidence."""
    print("=" * 80)
    print("EVIDENCE COMPARISON: Your System vs LLM")
    print("=" * 80)
    print(f"\nQuery: {query}\n")
    
    # Query your system
    print("🔍 Querying your knowledge graph system...")
    print("-" * 80)
    
    if not SYSTEM_AVAILABLE:
        print("❌ System not available")
        return
    
    result = answer_query(query)
    
    print("\n✅ YOUR SYSTEM RESPONSE:")
    print("=" * 80)
    print(result.get("answer", "No answer"))
    print()
    
    # Show evidence
    evidence = result.get("facts_used", [])
    if evidence:
        print(f"📊 EVIDENCE ({len(evidence)} facts from knowledge graph):")
        print("-" * 80)
        for i, fact in enumerate(evidence[:10], 1):  # Show first 10
            print(f"{i}. {fact.get('fact_text', 'N/A')}")
            if fact.get("is_operational"):
                print("   [Operational Insight]")
        if len(evidence) > 10:
            print(f"\n... and {len(evidence) - 10} more facts")
        print()
    else:
        print("📊 EVIDENCE: No facts shown (may use direct CSV computation)")
        print()
    
    print("\n" + "=" * 80)
    print("❌ LLM BASELINE (GPT-4, Claude, etc.)")
    print("=" * 80)
    print("Response: [Would provide an answer, but...]")
    print()
    print("📊 EVIDENCE: NONE")
    print("   - LLMs cannot show which facts/data they used")
    print("   - No traceability to source data")
    print("   - Cannot verify accuracy")
    print("   - May hallucinate or use outdated training data")
    print()
    
    print("=" * 80)
    print("KEY ADVANTAGE: TRACEABILITY")
    print("=" * 80)
    print("""
Your system provides:
✓ Traceable evidence (facts from knowledge graph)
✓ Verifiable answers (can check against source data)
✓ Consistent results (same query = same answer)
✓ Up-to-date data (from your uploaded dataset)

LLM baselines provide:
✗ No evidence (black box)
✗ Unverifiable answers
✗ Inconsistent results (may vary between calls)
✗ Potentially outdated training data
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Show evidence comparison for a query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python show_evidence_comparison.py "Which department has the highest average salary?"
  python show_evidence_comparison.py "What is the average engagement by manager?"
        """
    )
    
    parser.add_argument("query", nargs="+", help="Query to compare")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more details")
    
    args = parser.parse_args()
    query = " ".join(args.query)
    
    show_evidence_comparison(query)


if __name__ == "__main__":
    main()

