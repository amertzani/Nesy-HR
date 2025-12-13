#!/bin/bash
# Quick Comparison Demo Script
# This script demonstrates how to compare your system with LLMs

echo "=========================================="
echo "Knowledge Graph System vs LLM Comparison"
echo "=========================================="
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='your-key'"
    echo "   Or add to .env file"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Quick Evidence Demo"
echo "---------------------------"
echo "This shows how your system provides traceable evidence:"
echo ""
python show_evidence_comparison.py "Which department has the highest average salary?"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "Step 2: Full Comparison (Single Scenario)"
echo "-------------------------------------------"
echo "Comparing your system with GPT-4 on scenario O1..."
echo ""
python compare_with_llm.py --scenario O1 --max-queries 2
echo ""
read -p "Press Enter to continue..."

echo ""
echo "Step 3: View Results"
echo "---------------------"
echo "Opening comparison report..."
echo ""
if [ -f "llm_comparison_report.txt" ]; then
    cat llm_comparison_report.txt | head -100
    echo ""
    echo "... (see full report in llm_comparison_report.txt)"
else
    echo "Report not found. Run comparison first."
fi

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review llm_comparison_report.txt"
echo "  2. Run full comparison: python compare_with_llm.py --all"
echo "  3. Use results in your paper/presentation"
echo ""

