# Validation Approach for Architectural Properties

## Quick Testing Strategy

### 1️⃣ Evidence-Boundedness (Easiest to Start)

**What to Measure:**
- For each query response, check if `facts_used` list is populated
- Count responses with complete evidence vs. empty evidence
- Show examples: evidence exists → answer, evidence missing → no answer

**How to Test:**
```python
# For each query:
response = answer_query(query)
has_evidence = len(response.get("facts_used", [])) > 0
has_answer = response.get("answer") is not None and response.get("answer").strip()

# Categorize:
# - Complete: has_answer AND has_evidence
# - Conservative failure: no_answer AND no_evidence (or partial evidence)
# - Problem: has_answer BUT no_evidence (shouldn't happen)
```

**Metrics:**
- % responses with evidence sets
- % responses with empty evidence (conservative failure)
- Examples showing evidence → answer, no evidence → no answer

---

### 2️⃣ Conservative Failure Behavior

**What to Measure:**
- Categorize each query into: fully supported, partially supported, unsupported
- Show how system handles each category

**How to Test:**
```python
# For each query:
evidence_count = len(response.get("facts_used", []))
answer_quality = check_answer_completeness(response.get("answer"))

# Categorize:
if evidence_count >= expected_facts and answer_quality == "complete":
    category = "fully_supported"
elif evidence_count > 0 and answer_quality in ["partial", "complete"]:
    category = "partially_supported"
else:
    category = "unsupported"

# Track how system handles each:
# - fully_supported → full answer
# - partially_supported → partial answer or "insufficient evidence"
# - unsupported → "No answer available" or empty response
```

**Output:**
- Small table showing counts and handling for each category

---

### 3️⃣ Traceability & Provenance Completeness

**What to Measure:**
- For each fact in responses, check for: source_document, timestamp, processing_pathway
- Calculate % of facts with complete provenance

**How to Test:**
```python
# For each fact in response["facts_used"]:
has_source = "source_document" in fact and fact["source_document"] not in [None, "unknown", ""]
has_timestamp = "timestamp" in fact and fact["timestamp"] is not None
has_pathway = "agent_id" in fact or "source_type" in fact

complete_provenance = has_source and has_timestamp and has_pathway

# Calculate:
total_facts = sum(len(r.get("facts_used", [])) for r in responses)
facts_with_provenance = count_facts_with_complete_provenance(responses)
provenance_completeness = facts_with_provenance / total_facts
```

**Output:**
- % of facts with complete provenance
- One example provenance chain (source → timestamp → pathway → response)

---

### 4️⃣ Determinism & Reproducibility

**What to Measure:**
- Run same queries multiple times, check if results are identical

**How to Test:**
```python
# Run each query 3 times:
results = []
for i in range(3):
    result = answer_query(query)
    results.append({
        "answer": result.get("answer"),
        "facts_count": len(result.get("facts_used", [])),
        "method": result.get("method")
    })

# Check consistency:
all_identical = all(
    r["answer"] == results[0]["answer"] and
    r["facts_count"] == results[0]["facts_count"] and
    r["method"] == results[0]["method"]
    for r in results
)
```

**Output:**
- % queries with 100% consistency across runs
- Report any deviations (should be 0% if truly deterministic)

---

## Implementation Plan

1. **Start with Evidence-Boundedness** - Easiest, just check `facts_used` list
2. **Add Conservative Failure** - Categorize based on evidence availability
3. **Add Provenance** - Check metadata on facts
4. **Add Determinism** - Run queries multiple times

Let's build this step by step!

