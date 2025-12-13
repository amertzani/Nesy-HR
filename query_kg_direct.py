"""
Direct Knowledge Graph Query Tool
=================================

This tool allows you to query the knowledge graph directly by keywords,
bypassing the LLM. It's perfect for extracting operational insights and statistics.

Usage:
    python query_kg_direct.py "average engagement by manager"
    python query_kg_direct.py "salary department" --limit 20
    python query_kg_direct.py "performance" --source operational_insights
    python query_kg_direct.py --list-operational
"""

import sys
import os
import argparse
from typing import List, Dict, Any, Optional
from urllib.parse import unquote

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from knowledge import graph, get_fact_source_document, get_fact_details, load_knowledge_graph
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    print("❌ Could not import knowledge graph modules")


def search_kg_by_keywords(keywords: List[str], 
                         source_filter: Optional[str] = None,
                         limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search knowledge graph by keywords.
    
    Args:
        keywords: List of keywords to search for
        source_filter: Optional source document filter (e.g., "operational_insights")
        limit: Maximum number of facts to return
    
    Returns:
        List of matching facts with metadata
    """
    if not KG_AVAILABLE or graph is None:
        return []
    
    keywords_lower = [k.lower() for k in keywords]
    matches = []
    
    for s, p, o in graph:
        # Skip metadata triples
        predicate_str = str(p)
        if ('fact_subject' in predicate_str or 'fact_predicate' in predicate_str or 
            'fact_object' in predicate_str or 'has_details' in predicate_str or 
            'source_document' in predicate_str or 'uploaded_at' in predicate_str or
            'is_inferred' in predicate_str or 'confidence' in predicate_str or
            'agent_id' in predicate_str):
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
        
        # Build searchable text (include all variations)
        fact_text = f"{subject} {predicate} {object_val}".lower()
        fact_text_full = f"{subject} {predicate} {object_val} {subject} {predicate}".lower()  # Include reversed for better matching
        
        # Check if any keyword matches (more flexible matching)
        keyword_match = False
        for kw in keywords_lower:
            # Direct match
            if kw in fact_text or kw in fact_text_full:
                keyword_match = True
                break
            # Partial word match (e.g., "manager" matches "managers")
            if any(kw in word or word in kw for word in fact_text.split() if len(word) > 3):
                keyword_match = True
                break
        
        if not keyword_match:
            continue
        
        # Get source document
        sources = get_fact_source_document(subject, predicate, object_val)
        
        # Apply source filter if specified
        if source_filter:
            source_match = any(source_filter.lower() in str(src).lower() for src, _ in sources)
            if not source_match:
                continue
        
        # Get details
        details = get_fact_details(subject, predicate, object_val)
        
        matches.append({
            "subject": subject,
            "predicate": predicate,
            "object": object_val,
            "fact_text": f"{subject} → {predicate} → {object_val}",
            "sources": [str(src) for src, _ in sources],
            "details": details
        })
        
        if len(matches) >= limit:
            break
    
    return matches


def get_operational_insights_direct() -> Dict[str, Any]:
    """
    Get operational insights directly from the knowledge graph.
    Extracts facts with source "operational_insights".
    """
    if not KG_AVAILABLE:
        return {}
    
    # Search for operational insights
    insights = {
        "by_manager": [],
        "by_department": [],
        "by_recruitment_source": [],
        "other": []
    }
    
    # Keywords that indicate operational insights
    insight_keywords = ["manager", "department", "recruitment", "average", "team", 
                       "engagement", "performance", "salary", "absence"]
    
    facts = search_kg_by_keywords(insight_keywords, source_filter="operational_insights", limit=500)
    
    for fact in facts:
        fact_text = fact["fact_text"].lower()
        
        # Categorize by type
        if "manager" in fact_text:
            insights["by_manager"].append(fact)
        elif "department" in fact_text:
            insights["by_department"].append(fact)
        elif "recruitment" in fact_text or "source" in fact_text:
            insights["by_recruitment_source"].append(fact)
        else:
            insights["other"].append(fact)
    
    return insights


def format_fact(fact: Dict[str, Any], show_source: bool = True) -> str:
    """Format a fact for display."""
    lines = [f"  {fact['fact_text']}"]
    
    if show_source and fact.get('sources'):
        sources_str = ", ".join(fact['sources'][:2])  # Show first 2 sources
        lines.append(f"    [Source: {sources_str}]")
    
    if fact.get('details'):
        lines.append(f"    Details: {fact['details'][:100]}...")
    
    return "\n".join(lines)


def print_results(matches: List[Dict[str, Any]], query: str, show_source: bool = True):
    """Print search results in a readable format."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print(f"Found {len(matches)} matching facts")
    print("=" * 80)
    
    if not matches:
        print("\nNo matching facts found.")
        print("\nTips:")
        print("  - Try different keywords")
        print("  - Use --list-operational to see available operational insights")
        print("  - Check if the knowledge graph has been populated")
        return
    
    # Group by source if multiple sources
    by_source = {}
    for fact in matches:
        source = fact.get('sources', ['unknown'])[0] if fact.get('sources') else 'unknown'
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(fact)
    
    # Print grouped results
    for source, facts in by_source.items():
        print(f"\n[{source}] ({len(facts)} facts)")
        print("-" * 80)
        for i, fact in enumerate(facts[:20], 1):  # Limit to 20 per source
            print(f"{i}. {format_fact(fact, show_source=show_source)}")
        
        if len(facts) > 20:
            print(f"\n  ... and {len(facts) - 20} more facts")
    
    print("\n" + "=" * 80)


def list_operational_insights():
    """List all operational insights available in the KG."""
    print("\n" + "=" * 80)
    print("OPERATIONAL INSIGHTS IN KNOWLEDGE GRAPH")
    print("=" * 80)
    
    insights = get_operational_insights_direct()
    
    total = sum(len(v) for v in insights.values())
    print(f"\nTotal operational insight facts: {total}\n")
    
    # By Manager
    if insights["by_manager"]:
        print(f"📊 By Manager ({len(insights['by_manager'])} facts):")
        print("-" * 80)
        for fact in insights["by_manager"][:10]:
            print(f"  • {fact['fact_text']}")
        if len(insights["by_manager"]) > 10:
            print(f"  ... and {len(insights['by_manager']) - 10} more")
        print()
    
    # By Department
    if insights["by_department"]:
        print(f"📊 By Department ({len(insights['by_department'])} facts):")
        print("-" * 80)
        for fact in insights["by_department"][:10]:
            print(f"  • {fact['fact_text']}")
        if len(insights["by_department"]) > 10:
            print(f"  ... and {len(insights['by_department']) - 10} more")
        print()
    
    # By Recruitment Source
    if insights["by_recruitment_source"]:
        print(f"📊 By Recruitment Source ({len(insights['by_recruitment_source'])} facts):")
        print("-" * 80)
        for fact in insights["by_recruitment_source"][:10]:
            print(f"  • {fact['fact_text']}")
        if len(insights["by_recruitment_source"]) > 10:
            print(f"  ... and {len(insights['by_recruitment_source']) - 10} more")
        print()
    
    # Other
    if insights["other"]:
        print(f"📊 Other Operational Insights ({len(insights['other'])} facts):")
        print("-" * 80)
        for fact in insights["other"][:10]:
            print(f"  • {fact['fact_text']}")
        if len(insights["other"]) > 10:
            print(f"  ... and {len(insights['other']) - 10} more")
        print()
    
    print("=" * 80)
    print("\n💡 Use these keywords in your queries:")
    print("   - 'manager engagement' or 'team engagement'")
    print("   - 'department salary' or 'department performance'")
    print("   - 'recruitment source' or 'recruitment performance'")
    print("   - 'average' + any metric (salary, performance, engagement, etc.)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Query knowledge graph directly by keywords",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_kg_direct.py "average engagement by manager"
  python query_kg_direct.py "salary department" --limit 20
  python query_kg_direct.py "performance" --source operational_insights
  python query_kg_direct.py --list-operational
        """
    )
    
    parser.add_argument("query", nargs="*", help="Keywords to search for")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of facts to return")
    parser.add_argument("--source", help="Filter by source document (e.g., 'operational_insights')")
    parser.add_argument("--list-operational", action="store_true", help="List all operational insights")
    parser.add_argument("--no-source", action="store_true", help="Don't show source information")
    
    args = parser.parse_args()
    
    if not KG_AVAILABLE:
        print("❌ Knowledge graph not available. Make sure the system is initialized.")
        return
    
    # CRITICAL: Load knowledge graph from disk
    print("📂 Loading knowledge graph from disk...")
    try:
        load_result = load_knowledge_graph()
        if load_result:
            print(f"✅ {load_result}")
        else:
            print("⚠️  Knowledge graph file not found or empty")
    except Exception as e:
        print(f"⚠️  Error loading knowledge graph: {e}")
    
    # Check graph size
    graph_size = len(graph) if graph else 0
    if graph_size == 0:
        print("\n⚠️  Knowledge graph is empty after loading.")
        print("   Possible reasons:")
        print("   1. No document has been processed yet")
        print("   2. Knowledge graph file doesn't exist: knowledge_graph.pkl")
        print("   3. File exists but is empty")
        print("\n   To populate the graph:")
        print("   1. Start the backend: ./start_backend.sh")
        print("   2. Upload your CSV file through the web interface")
        print("   3. Wait for processing to complete")
        return
    
    print(f"📊 Knowledge graph size: {graph_size} facts")
    
    # List operational insights
    if args.list_operational:
        list_operational_insights()
        return
    
    # Search query
    if not args.query:
        print("❌ No query provided.")
        print("   Usage: python query_kg_direct.py <keywords>")
        print("   Example: python query_kg_direct.py 'average engagement manager'")
        print("   Or use: python query_kg_direct.py --list-operational")
        return
    
    query = " ".join(args.query)
    keywords = args.query
    
    print(f"\n🔍 Searching for: {query}")
    if args.source:
        print(f"   Filter: source = '{args.source}'")
    print(f"   Limit: {args.limit} facts")
    
    # Search
    matches = search_kg_by_keywords(keywords, source_filter=args.source, limit=args.limit)
    
    # Print results
    print_results(matches, query, show_source=not args.no_source)


if __name__ == "__main__":
    main()

