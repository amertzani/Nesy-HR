# How to Validate Architectural Properties

## Quick Start

### 1. Run the Validation

```bash
# With default test queries
python validate_architectural_properties.py

# With your test scenarios
python validate_architectural_properties.py --queries evaluation/test_scenarios.json

# Specify output file
python validate_architectural_properties.py --output my_results.json
```

### 2. What Gets Validated

The script validates **4 key architectural properties**:

#### 1️⃣ Evidence-Boundedness
- **Measures**: % of responses with evidence, % with empty evidence
- **Shows**: Examples where evidence exists → answer, evidence missing → no answer
- **Validates**: Section 4.1 (Evidence-Bounded Generation)

#### 2️⃣ Conservative Failure Behavior
- **Categorizes**: Fully supported, partially supported, unsupported queries
- **Shows**: Table of how each category is handled
- **Validates**: Governance over usefulness trade-offs

#### 3️⃣ Traceability & Provenance
- **Measures**: % of facts with complete provenance (source, timestamp, pathway)
- **Shows**: Example provenance chain
- **Validates**: Accountability claims

#### 4️⃣ Determinism & Reproducibility
- **Tests**: Runs each query 3 times
- **Measures**: % consistency across runs
- **Validates**: Testability and auditability

### 3. Understanding the Output

The script prints:
- **Summary statistics** for each pillar
- **Examples** showing evidence-bounded behavior
- **Table** showing conservative failure handling
- **Provenance chain** example
- **Consistency report** for determinism

Results are saved to JSON for further analysis.

### 4. What to Look For

✅ **Good Results:**
- High % of responses with evidence
- Conservative failure: no answers when evidence missing
- High % of facts with complete provenance
- 100% consistency across runs

⚠️ **Problems to Fix:**
- Answers without evidence (shouldn't happen!)
- Low provenance completeness
- Inconsistent results across runs

### 5. Next Steps

After running validation:
1. Review the printed summary
2. Check the JSON output for detailed metrics
3. Use examples in your paper to show architectural properties
4. Fix any issues found (especially answers without evidence)

## Example Output

```
🔍 Running architectural property validation...
================================================================================

1️⃣  VALIDATING EVIDENCE-BOUNDEDNESS...
   Total queries: 10
   Responses with evidence: 8 (80.0%)
   Responses without evidence: 2 (20.0%)
   Answers with evidence: 8 (80.0%)

   Examples:
   - evidence_exists_answer_produced:
     Query: What is the average salary by department?
     Answer: IT/IS has the highest average salary of $92,524.25
     Evidence facts: 15
   - evidence_missing_no_answer:
     Query: What is the salary of Unknown, Person?
     Answer: No answer
     Evidence facts: 0

2️⃣  VALIDATING CONSERVATIVE FAILURE BEHAVIOR...
   Category Distribution:
   Category                  Count      Full Answer     Partial        No Answer      
   ---------------------------------------------------------------------------
   fully_supported           6          6               0              0
   partially_supported        2          1               1              0
   unsupported                2          0               0              2

3️⃣  VALIDATING TRACEABILITY & PROVENANCE...
   Total facts in responses: 45
   Facts with source document: 45 (100.0%)
   Facts with timestamp: 45 (100.0%)
   Facts with pathway: 45 (100.0%)
   Facts with complete provenance: 45 (100.0%)

4️⃣  VALIDATING DETERMINISM & REPRODUCIBILITY...
   Queries tested: 10
   Runs per query: 3
   Consistent queries: 10 (100.0%)
   Inconsistent queries: 0

✅ Validation complete!
```

## Tips

- Start with a small set of queries (5-10) to test
- Focus on queries you know have evidence vs. queries that shouldn't
- Check the examples - they're perfect for your paper!
- The table for conservative failure is ready to use in LaTeX

