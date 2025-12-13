#!/bin/bash
# Quick script to upload CSV and test fact retrieval

CSV_FILE="/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv"
API_URL="http://localhost:8000"

echo "📤 Uploading CSV file and testing fact retrieval..."
echo "="

# Check if server is running
if ! curl -s "$API_URL/api/knowledge/facts?limit=1" > /dev/null 2>&1; then
    echo "❌ API server is not running!"
    echo "   Please start it first: python3 api_server.py"
    exit 1
fi

echo "✅ API server is running"

# Upload CSV file
echo ""
echo "📤 Uploading CSV file: $CSV_FILE"
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ CSV file not found: $CSV_FILE"
    exit 1
fi

UPLOAD_RESPONSE=$(curl -s -X POST "$API_URL/api/knowledge/upload" \
    -F "files=@$CSV_FILE")

echo "$UPLOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$UPLOAD_RESPONSE"

# Wait a bit for processing
echo ""
echo "⏳ Waiting 5 seconds for processing..."
sleep 5

# Check fact count
echo ""
echo "📊 Checking knowledge graph..."
FACTS_RESPONSE=$(curl -s "$API_URL/api/knowledge/facts?limit=1")
FACT_COUNT=$(echo "$FACTS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('facts', [])))" 2>/dev/null || echo "0")

echo "   Facts in knowledge graph: $FACT_COUNT"

# Test a query
echo ""
echo "🧪 Testing fact retrieval with query: 'What is the average engagement survey value per manager name?'"
echo "="

QUERY_RESPONSE=$(curl -s -X POST "$API_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "What is the average engagement survey value per manager name?"}')

echo "$QUERY_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$QUERY_RESPONSE"

echo ""
echo "✅ Test complete!"

