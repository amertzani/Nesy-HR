# Testing Employee Fact Retrieval

This guide shows you how to test how the system retrieves facts relevant to employees.

## Prerequisites

1. **Upload a CSV file first** - The knowledge graph needs to have facts before you can test retrieval
   - Upload via frontend: http://localhost:3000
   - Or use the API: `POST /api/knowledge/upload`

## Testing Methods

### Method 1: Direct Testing (Knowledge Graph in Memory)

```bash
# Make sure you've uploaded a CSV file first
python3 test_employee_fact_retrieval.py --all
```

This tests fact retrieval directly from the knowledge graph.

### Method 2: API Testing (Requires Running Server)

```bash
# Start the API server first
python3 api_server.py

# In another terminal, run the test
python3 test_employee_fact_retrieval.py --all --api
```

This tests fact retrieval via the API endpoint, which is closer to how the frontend works.

### Method 3: Test Specific Employee

```bash
# Test facts for a specific employee
python3 test_employee_fact_retrieval.py --employee "Adinolfi, Wilson K"
```

## What Gets Tested

The test script checks:

1. **Manager Engagement Queries**
   - "What is the average engagement survey value per manager name?"
   - Should find: Amy Dunn (4.43), Michael Albert (4.29), Simon Roup (4.49)

2. **Employee-Specific Queries**
   - "What is the salary of Adinolfi?"
   - "Who has the most absences?"
   - "Which employees have the highest performance scores?"

3. **Team-Based Queries**
   - "What is the average engagement for Amy Dunn's team?"
   - "What is Michael Albert's team average engagement?"

4. **Department Queries**
   - "What employees are in the Production department?"

## Expected Results

After uploading HR_S.csv, you should see:

- ✅ Facts found for manager engagement queries with precise values (4.43, 4.29, 4.49)
- ✅ Facts found for employee names
- ✅ Facts found for team metrics
- ✅ Facts found for department information

## Troubleshooting

### "Knowledge graph is empty (0 facts)"

**Solution:** Upload your CSV file first via the frontend or API.

### "No relevant facts found"

**Possible causes:**
1. CSV file hasn't been uploaded yet
2. Facts haven't been stored yet (wait a few seconds after upload)
3. Employee names don't match exactly (try partial names)

### "Found rounded values (4) instead of precise values"

**Solution:** Re-upload the CSV file. The cleanup code will remove old rounded facts and store new precise ones.

## Quick Test via API

If your server is running, you can also test directly:

```bash
# Test a query
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the average engagement survey value per manager name?"}'
```

## Manual Testing in Frontend

1. Go to http://localhost:3000
2. Upload your CSV file
3. Wait for processing to complete
4. Ask questions like:
   - "What is the average engagement survey value per manager name?"
   - "What is Amy Dunn's team average engagement?"
   - "Who has the most absences?"

The system should retrieve relevant facts and provide accurate answers with precise values.

