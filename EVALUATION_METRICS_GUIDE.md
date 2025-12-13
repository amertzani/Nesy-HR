# Evaluation Metrics Extraction Guide

This guide explains how to extract real metrics from your HR dataset and system to replace the dummy values in your LaTeX paper.

## Overview

Your paper uses dummy values like:
- `R = 311` (number of employee records)
- `N = 36` (number of variables)
- `C(36, 2) = 630` (number of 2-variable combinations)

This guide shows you how to extract the **actual values** from your dataset and system.

## Step 1: Locate Your CSV Dataset

Your system has processed a file called `HRDataset_v14.csv`. To extract metrics, you need access to this file.

### Option A: If the CSV file is still available

1. Find the CSV file path:
   ```bash
   # The file might be in:
   # - /tmp/HRDataset_v14.csv
   # - Current directory
   # - An uploads folder
   
   # Search for it:
   find /tmp -name "*HRDataset*.csv" 2>/dev/null
   find . -name "*HRDataset*.csv" 2>/dev/null
   ```

2. Run the evaluation script:
   ```bash
   python evaluation_metrics.py /path/to/HRDataset_v14.csv
   ```

### Option B: If you need to re-upload the CSV

1. Upload the CSV file through your system's web interface
2. Wait for processing to complete
3. Run the evaluation script (it will auto-detect the file):
   ```bash
   python evaluation_metrics.py
   ```

### Option C: Use a sample dataset

If you want to test with a different dataset, provide the path:
```bash
python evaluation_metrics.py /path/to/your/dataset.csv
```

## Step 2: Extract Metrics

The `evaluation_metrics.py` script will:

1. **Load the dataset** and extract:
   - `R`: Number of employee records (rows)
   - `N`: Number of variables (columns)
   - Column names and types

2. **Calculate combinatorial metrics**:
   - `C(N, 2)`: Number of 2-variable combinations (operational queries)
   - `C(N, 3)`: Number of 3-variable combinations (strategic queries)
   - `C(N, k)` for k = 2 to 5
   - Total combinations up to K_max
   - Pairwise interactions E(k) for each k

3. **Calculate cutoff orders**:
   - For different values of E_max, find k* (the cutoff between operational and strategic)

## Step 3: Output Files

The script generates three files:

### 1. `evaluation_metrics_report.txt`
Human-readable report with all metrics.

### 2. `latex_metrics_snippets.tex`
LaTeX code snippets ready to paste into your paper. Example:
```latex
% Replace: 'R = 311' with:
R = 311

% Replace: 'N = 36' with:
N = 36

% Replace in equation (eq:operational-combinations):
\left|\mathcal{C}_2\right| = \binom{36}{2} = \frac{36 \times 35}{2} = 630,
```

### 3. `evaluation_metrics.json`
Machine-readable JSON with all metrics for programmatic use.

## Step 4: Update Your LaTeX Paper

### Replace Dataset Dimensions

In your paper text, replace:
```latex
% OLD (dummy):
We start from an HR analytics dataset containing $R$ employee records and $N$ variables. 
For experimentation purposes, we rely on a public HR dataset from Kaggle which in our case 
comprises $R = 311$ employees and $N = 36$ variables...

% NEW (from script output):
We start from an HR analytics dataset containing $R$ employee records and $N$ variables. 
For experimentation purposes, we rely on a public HR dataset from Kaggle which in our case 
comprises $R = [ACTUAL_VALUE] employees and $N = [ACTUAL_VALUE] variables...
```

### Replace Combinatorial Equations

Replace the values in equations:

```latex
% OLD:
\left|\mathcal{C}_2\right| = \binom{36}{2} = \frac{36 \times 35}{2} = 630,

% NEW (from script):
\left|\mathcal{C}_2\right| = \binom{[N]}{2} = \frac{[N] \times [N-1]}{2} = [C_2],
```

### Update Variable Examples

The script will also list actual column names from your dataset. Use these in your examples:
```latex
% Example from script output:
% \texttt{Salary}, \texttt{Department}, \texttt{EmploymentStatus}, etc.
```

## Step 5: System Performance Metrics (Future Enhancement)

To extract **system performance metrics** (accuracy, response times, etc.), you would need to:

1. **Create query templates** for operational (k=2) and strategic (k≥3) scenarios
2. **Run queries** through your system
3. **Compare with LLM baselines** (GPT-4, Claude, Gemini, Grok)
4. **Calculate accuracy metrics**:
   - Exact match for categorical outputs
   - Tolerance-based correctness for numerical quantities
   - Response time measurements

This would require additional evaluation infrastructure. The current script focuses on **dataset and combinatorial metrics** only.

## Example Workflow

```bash
# 1. Make sure your CSV is accessible
# (Either already uploaded, or provide path)

# 2. Run the evaluation script
python evaluation_metrics.py

# 3. Check the output files
cat evaluation_metrics_report.txt
cat latex_metrics_snippets.tex

# 4. Copy values from latex_metrics_snippets.tex into your paper
```

## Troubleshooting

### "CSV file not found"

**Solution**: Provide the path directly:
```bash
python evaluation_metrics.py /full/path/to/your/file.csv
```

### "Failed to load dataset"

**Solution**: Check that:
- The file is a valid CSV
- The file is readable
- You have pandas installed: `pip install pandas`

### Script runs but values seem wrong

**Solution**: 
- Verify the CSV file is the correct one
- Check that all columns are being read correctly
- Review the `evaluation_metrics_report.txt` to see what was detected

## Next Steps

Once you have the real metrics:

1. ✅ Replace dummy values in your LaTeX paper
2. ✅ Verify all equations use correct values
3. ✅ Update variable name examples with actual column names
4. 🔄 (Future) Add system performance evaluation
5. 🔄 (Future) Compare with LLM baselines

## Questions?

If you encounter issues:
1. Check that your CSV file is accessible
2. Verify pandas is installed: `pip install pandas scipy`
3. Review the error messages for specific guidance

