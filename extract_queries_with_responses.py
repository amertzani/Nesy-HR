#!/usr/bin/env python3
"""
Extract queries with responses from the offline evaluation report
and create QUERIES_WITH_RESPONSES.md with the 32 queries.
"""

import re
from typing import List, Dict, Any

def parse_evaluation_report(report_path: str) -> List[Dict[str, Any]]:
    """Parse the offline evaluation report."""
    with open(report_path, 'r') as f:
        content = f.read()
    
    queries = []
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Match query start
        query_match = re.match(r'\[(\d+)\]\s+Query:\s+(.+?)$', line)
        if query_match:
            query_num = int(query_match.group(1))
            query_text = query_match.group(2).strip()
            
            query_data = {
                'query_id': query_num,
                'query': query_text,
                'response': '',
                'response_time': 0.0,
                'evidence_count': 0,
                'evidence_facts': [],
                'correct': False
            }
            
            # Parse response and evidence
            i += 1
            response_lines = []
            collecting_response = False
            
            while i < len(lines):
                line = lines[i]
                
                # Match response time
                if line.startswith('Response'):
                    time_match = re.search(r'\(([\d.]+)s\)', line)
                    if time_match:
                        query_data['response_time'] = float(time_match.group(1))
                    collecting_response = True
                    i += 1
                    # Collect response lines (indented lines with content)
                    while i < len(lines):
                        resp_line = lines[i]
                        # Stop if we hit evidence marker
                        if '📊 Evidence:' in resp_line:
                            collecting_response = False
                            break
                        # Collect lines that are part of the response
                        if resp_line.strip():
                            # Skip empty lines but keep formatted content
                            if resp_line.startswith('  ') or resp_line.startswith('•') or ':' in resp_line:
                                response_lines.append(resp_line.rstrip())
                        i += 1
                    continue
                
                # Stop collecting response when we hit evidence
                if '📊 Evidence:' in line:
                    collecting_response = False
                    evidence_text = line.split('📊 Evidence:')[1].strip()
                    if 'No facts retrieved' in evidence_text:
                        query_data['evidence_count'] = 0
                    else:
                        count_match = re.search(r'(\d+)\s+facts?', evidence_text)
                        if count_match:
                            query_data['evidence_count'] = int(count_match.group(1))
                    
                    # Extract evidence facts
                    i += 1
                    while i < len(lines):
                        fact_line = lines[i]
                        if fact_line.startswith('✓') or fact_line.startswith('✗') or fact_line.startswith('---'):
                            break
                        fact_match = re.match(r'^\s*\d+\.\s+(.+)$', fact_line)
                        if fact_match:
                            query_data['evidence_facts'].append(fact_match.group(1).strip())
                        i += 1
                    continue
                
                # Match correctness
                if line.startswith('✓ Correct'):
                    query_data['correct'] = True
                    break
                elif line.startswith('✗ Incorrect'):
                    query_data['correct'] = False
                    # Try to extract error message
                    error_match = re.search(r'✗ Incorrect:\s*(.+?)$', line)
                    if error_match:
                        query_data['error'] = error_match.group(1).strip()
                    break
                
                # Collect response lines
                if collecting_response and line.strip():
                    # Skip lines that are just formatting
                    if not line.startswith('  Average') and not line.startswith('  •') and not line.startswith('📊'):
                        response_lines.append(line.strip())
                
                i += 1
            
            query_data['response'] = '\n'.join(response_lines)
            queries.append(query_data)
        
        i += 1
    
    return queries

def generate_queries_with_responses_md(queries: List[Dict[str, Any]], output_path: str = 'QUERIES_WITH_RESPONSES.md'):
    """Generate markdown file with queries and responses."""
    
    lines = [
        "# All Tested Queries with System Responses and Extracted Facts",
        "",
        f"**Total Queries**: {len(queries)}",
        "**Source**: Offline Evaluation Report (Latest)",
        "**Dataset**: HRDataset_v14.csv (311 rows, 36 columns)",
        "",
        "---",
        ""
    ]
    
    for query_data in queries:
        query_id = query_data['query_id']
        query = query_data['query']
        response = query_data['response']
        response_time = query_data['response_time']
        evidence_count = query_data['evidence_count']
        evidence_facts = query_data['evidence_facts']
        correct = query_data['correct']
        error = query_data.get('error')
        
        lines.append(f"## Query {query_id}")
        lines.append(f"**Query**: {query}")
        lines.append("")
        lines.append(f"**Response** ({response_time:.2f}s):")
        lines.append("```")
        if response:
            lines.append(response)
        else:
            lines.append("(No response text)")
        lines.append("```")
        lines.append("")
        
        if evidence_count > 0:
            lines.append(f"**Evidence**: {evidence_count} facts retrieved")
            lines.append("")
            lines.append("**Extracted Facts:**")
            lines.append("")
            lines.append("**❓ Other Facts:**")
            for i, fact in enumerate(evidence_facts, 1):
                lines.append(f"{i}. {fact}")
        else:
            lines.append("**Evidence**: No facts retrieved")
        
        lines.append("")
        if correct:
            lines.append("**Result**: ✓ Correct")
        else:
            if error:
                lines.append(f"**Result**: ✗ Incorrect: {error}")
            else:
                lines.append("**Result**: ✗ Incorrect")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Generated {output_path} with {len(queries)} queries")

def main():
    report_path = 'offline_evaluation_report.txt'
    output_path = 'QUERIES_WITH_RESPONSES.md'
    
    print(f"📄 Parsing {report_path}...")
    queries = parse_evaluation_report(report_path)
    print(f"✅ Found {len(queries)} queries")
    
    print(f"📝 Generating {output_path}...")
    generate_queries_with_responses_md(queries, output_path)
    
    print(f"✅ Done! Updated {output_path} with {len(queries)} queries")

if __name__ == '__main__':
    main()

