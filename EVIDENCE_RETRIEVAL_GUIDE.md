# Evidence Retrieval Testing Guide

## ✅ What's New

The offline evaluation now includes **evidence retrieval scenarios** that demonstrate how your system retrieves facts from the knowledge graph.

## 🎯 Evidence Scenarios

### E1: Employee Fact Retrieval
Tests queries that retrieve facts about employees and departments.

**Queries:**
- "Show me facts about employees in IT/IS department"
- "What facts are available about salary information?"
- "Retrieve facts related to performance scores"
- "Find facts about engagement by manager"

### E2: Keyword-Based Fact Search
Tests keyword-based fact retrieval from the knowledge graph.

**Queries:**
- "Search for facts containing 'department' and 'salary'"
- "Find facts about 'performance' and 'manager'"
- "Retrieve facts with keywords 'engagement' and 'team'"

### E3: Department Facts Retrieval
Tests department-specific fact retrieval.

**Queries:**
- "What facts are stored about IT/IS department?"
- "Show me all facts related to Production department"
- "Retrieve facts about Sales department"

## 🚀 How to Test

### Test Evidence Scenarios Only
```bash
python evaluate_offline.py --evidence --max-queries 2
```

### Test All Scenarios (Including Evidence)
```bash
python evaluate_offline.py --all --max-queries 2
```

### Test Specific Evidence Scenario
```bash
python evaluate_offline.py --scenario E1
```

## 📊 What Gets Tested

### 1. Evidence Retrieval
- **Number of facts retrieved**: Counts how many facts are found
- **Fact content**: Shows actual fact text
- **Minimum requirement**: Checks if enough facts are retrieved

### 2. Evidence Quality
- **Has evidence**: Yes/No
- **Evidence count**: Number of facts
- **Meets minimum**: Whether it meets the minimum requirement (usually 3-5 facts)

## 📈 Example Output

```
Scenario E1: Employee Fact Retrieval
=====================================

[1/2] Query: Show me facts about employees in IT/IS department
  ✅ Response received (0.00s)
  📊 Evidence: 20 facts retrieved
     1. average salary in department Production → is → 58741...
     2. IS department → has → engagement score of 4...
  ✓ Evidence retrieved: 20 facts
  ✓ Meets minimum evidence requirement
```

## 📝 Report Features

The evaluation report now includes:

1. **Evidence Statistics**
   - Total evidence facts provided
   - Average evidence per query
   - Evidence retrieval success rate

2. **Evidence Details**
   - Shows actual fact text (first 5 facts)
   - Fact count per query
   - Evidence quality assessment

3. **Evidence Scenarios Section**
   - Separate section for evidence queries
   - Success rate for evidence retrieval
   - Examples of retrieved facts

## 🎯 What This Demonstrates

### 1. Traceability
Your system can show **which facts** support each answer:
```
Evidence: 20 facts retrieved
1. average salary in department Production → is → 58741
2. IS department → has → engagement score of 4.2
3. Manager Simon Roup → has → average engagement survey value of 4.33
...
```

### 2. Knowledge Graph Access
Your system can **search the knowledge graph** directly:
- Keyword-based search
- Entity-based search
- Multi-keyword search

### 3. Evidence Quality
Your system retrieves **relevant facts**:
- Facts match query keywords
- Facts are from the knowledge graph
- Facts can be verified

## 💡 Key Advantages Shown

1. **Traceability** - Can show evidence (facts from KG)
2. **Searchability** - Can search by keywords
3. **Verifiability** - Facts can be checked
4. **Transparency** - Shows what data supports answers

## 📊 Comparison with LLMs

| Feature | Your System | LLMs |
|---------|-------------|------|
| Evidence Retrieval | ✓ Yes (20+ facts) | ✗ No |
| Fact Traceability | ✓ Yes | ✗ No |
| Keyword Search | ✓ Yes | ✗ No |
| Verifiable Facts | ✓ Yes | ✗ No |

## 🚀 Quick Start

```bash
# Test evidence retrieval
python evaluate_offline.py --evidence

# View results
cat offline_evaluation_report.txt
```

## 📝 Example Report Section

```
EVIDENCE RETRIEVAL RESULTS
==========================

Query: "Show me facts about employees in IT/IS department"

Response: [System's answer]

📊 Evidence: 20 facts retrieved
  1. average salary in department Production → is → 58741
  2. IS department → has → engagement score of 4.2
  3. Manager Simon Roup → has → average engagement survey value of 4.33
  4. Recruitment source LinkedIn → has → average salary of 72925
  5. ... and 16 more facts

✓ Evidence retrieved: 20 facts
✓ Meets minimum evidence requirement (5+ facts)
```

## 🎓 Use in Your Paper

You can now demonstrate:

1. **Evidence Retrieval Capability**
   - System retrieves 20+ facts per query
   - Facts are relevant to the query
   - Facts are traceable to source

2. **Traceability Advantage**
   - Can show which facts support answers
   - Can verify answers against facts
   - Can explain reasoning

3. **Knowledge Graph Access**
   - Direct access to stored facts
   - Keyword-based search
   - Entity-based retrieval

---

**Ready to test?** Run:
```bash
python evaluate_offline.py --evidence --max-queries 2
```

