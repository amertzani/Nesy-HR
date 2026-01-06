# Quick Guide: Testing Employee Fact Retrieval

## Step 1: Upload Your CSV File

You have **3 options** to upload:

### Option A: Via Frontend (Easiest)
1. Open http://localhost:3000 in your browser
2. Click "Upload" and select your CSV file
3. Wait for processing to complete (you'll see "Upload complete")

### Option B: Via API (Command Line)
```bash
# Make sure server is running first
python3 api_server.py

# In another terminal, upload:
curl -X POST http://localhost:8000/api/knowledge/upload \
  -F "files=@/Users/s20/Desktop/Gnoses/HR\ Data/HR_S.csv"
```

### Option C: Use the Upload Script
```bash
./upload_and_test.sh
```

## Step 2: Test Fact Retrieval

### Method 1: Test Script (After Upload)
```bash
# Test all employee queries
python3 test_employee_fact_retrieval.py --all

# Test via API (if server is running)
python3 test_employee_fact_retrieval.py --all --api

# Test specific employee
python3 test_employee_fact_retrieval.py --employee "Adinolfi, Wilson K"
```

### Method 2: Direct API Test
```bash
# Test a query directly
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the average engagement survey value per manager name?"}'
```

### Method 3: Check Facts Directly
```bash
# See what facts are in the knowledge graph
curl "http://localhost:8000/api/knowledge/facts?limit=20" | python3 -m json.tool
```

## What to Look For

After uploading, you should see:
- ✅ Facts about managers with precise engagement values (4.43, 4.29, 4.49)
- ✅ Facts about individual employees
- ✅ Facts about departments and teams

## Troubleshooting

**"Knowledge graph is empty (0 facts)"**
- Solution: Upload your CSV file first (see Step 1)

**"No relevant facts found"**
- Wait a few seconds after upload for processing
- Check if facts exist: `curl "http://localhost:8000/api/knowledge/facts?limit=5"`

**"Found rounded values (4) instead of precise values"**
- Re-upload the CSV file - the cleanup code will remove old rounded facts

