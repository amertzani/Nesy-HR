# How Ground Truth Was Computed

## Overview

The ground truth in `test_scenarios.json` is computed **directly from the CSV dataset** using pandas operations. This provides the "correct" answers that your system's responses should match.

## Computation Method

The `compute_ground_truth()` function in `evaluation_test_scenarios.py` uses different methods depending on variable types:

### For 2-Variable Scenarios (k=2)

#### Case 1: Numeric Metric × Categorical Group
**Example**: `Absences` (numeric) × `EmploymentStatus` (categorical)

**Method**: `groupby().agg(['mean', 'median', 'std', 'count', 'min', 'max'])`

```python
grouped = df.groupby('EmploymentStatus')['Absences'].agg(['mean', 'median', 'std', 'count', 'min', 'max'])
```

**Result**: Statistics for each group
- Active: mean=9.83, median=10.0, count=207
- Terminated for Cause: mean=11.56, median=9.5, count=16
- Voluntarily Terminated: mean=10.95, median=11.0, count=88

#### Case 2: Categorical × Categorical
**Example**: `PerformanceScore` (categorical) × `Department` (categorical)

**Method**: `pd.crosstab()` - Cross-tabulation

```python
crosstab = pd.crosstab(df['PerformanceScore'], df['Department'], margins=True)
```

**Result**: Count table showing distribution
```
                Admin  IT/IS  Production  Sales  Software Engineering  All
Exceeds           0      6          27      2                    2   37
Fully Meets       9     42         159     24                    8  243
Needs Improvement 0      1          15      1                    1   18
PIP               0      1           8      4                    0   13
All               9     50         209     31                   11  311
```

**Note**: For "Which department has highest average performance?", you'd need to:
1. Convert PerformanceScore to numeric (Exceeds=4, Fully Meets=3, etc.)
2. Calculate mean per department
3. Find max

#### Case 3: Numeric × Numeric
**Method**: Correlation coefficient

```python
correlation = df['Var1'].corr(df['Var2'])
```

### For 3-Variable Scenarios (k≥3)

**Method**: Multi-dimensional `groupby()`

```python
grouped = df.groupby([group1, group2])[metric].agg(['mean', 'count'])
```

**Example**: `Department` × `EmploymentStatus` × `Salary`
- Groups by Department AND EmploymentStatus
- Calculates mean salary for each combination

## Example: "Which department has highest average performance?"

### Ground Truth Computation

Since `PerformanceScore` is categorical (Exceeds, Fully Meets, etc.), the ground truth uses crosstab. To answer "highest average", you need to:

1. **Convert to numeric**:
   - Exceeds = 4
   - Fully Meets = 3
   - Needs Improvement = 2
   - PIP = 1

2. **Calculate mean per department**:
   ```python
   score_map = {"Exceeds": 4, "Fully Meets": 3, "Needs Improvement": 2, "PIP": 1}
   df['PerfScoreNumeric'] = df['PerformanceScore'].map(score_map)
   dept_means = df.groupby('Department')['PerfScoreNumeric'].mean()
   ```

3. **Find maximum**:
   ```python
   highest_dept = dept_means.idxmax()
   highest_value = dept_means.max()
   ```

### From Your Ground Truth Data

Looking at the crosstab in `test_scenarios.json` for O1:

- **IT/IS**: 6 Exceeds + 42 Fully Meets + 1 Needs Improvement + 1 PIP = (6×4 + 42×3 + 1×2 + 1×1) / 50 = **2.94**
- **Production**: 27 Exceeds + 159 Fully Meets + 15 Needs Improvement + 8 PIP = (27×4 + 159×3 + 15×2 + 8×1) / 209 = **2.89**
- **Sales**: 2 Exceeds + 24 Fully Meets + 1 Needs Improvement + 4 PIP = (2×4 + 24×3 + 1×2 + 4×1) / 31 = **2.61**
- **Software Engineering**: 2 Exceeds + 8 Fully Meets + 1 Needs Improvement = (2×4 + 8×3 + 1×2) / 11 = **2.91**
- **Admin Offices**: 9 Fully Meets = (9×3) / 9 = **3.00**
- **Executive Office**: 1 Fully Meets = **3.00**

**Answer**: Admin Offices and Executive Office tie at 3.00 (highest)

## Testing Your System

You can test queries in the terminal:

```bash
# Answer a query
python answer_query_terminal.py "Which department has the highest average performance score?"

# See details
python answer_query_terminal.py "Which department has the highest average performance score?" --verbose
```

The tool searches the knowledge graph for relevant facts and extracts the answer.

## Comparing with Ground Truth

To compare system answers with ground truth:

1. **Extract answer from system** (using `answer_query_terminal.py`)
2. **Compute expected answer from ground truth** (using the crosstab data)
3. **Compare values** (with tolerance for floating point differences)
4. **Calculate accuracy**

For categorical metrics like PerformanceScore, you may need to:
- Convert categorical values to numeric for comparison
- Handle ties appropriately
- Consider distribution, not just mean

