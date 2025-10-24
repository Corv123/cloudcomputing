# Quick Start Guide

## Get Real Experiment Data in 3 Steps

### Step 1: Navigate to the experiment folder
```bash
cd fakenews/analytics&experiment
```

### Step 2: Run all experiments
```bash
python run_all_experiments.py
```

This will:
- Train the full model with detailed metrics
- Compare 5 baseline models
- Run ablation studies on feature groups
- Save all results to `./results/` directory

**Expected runtime:** 15-45 minutes (depending on dataset size)

### Step 3: View results
```bash
python visualize_results.py
```

This will display:
- Training metrics (accuracy, F1, ROC-AUC)
- Confusion matrix
- Cross-validation scores
- Baseline model comparison table
- Ablation study results with delta F1 scores

## Output Files

After running experiments, you'll have:

```
results/
├── training_results_20251015_174500.json       # Full training metrics
├── baseline_comparison_20251015_175000.json    # Model comparisons
└── ablation_study_20251015_180000.json         # Feature ablation results
```

## Use Results in Your Report

All metrics are saved in JSON format. Load them in Python:

```python
import json

# Load training results
with open('results/training_results_YYYYMMDD_HHMMSS.json', 'r') as f:
    data = json.load(f)

# Get metrics
accuracy = data['test_performance']['accuracy']
f1_score = data['test_performance']['f1_score']
confusion_matrix = data['test_performance']['confusion_matrix']
cv_mean = data['cross_validation']['cv_mean']
```

## Troubleshooting

**No datasets found:**
- Ensure datasets are in `../datasets/` directory
- Required files: BuzzFeed_*.csv, PolitiFact_*.csv, True.csv, Fake.csv

**Import error:**
- Run from `analytics&experiment` directory
- Ensure `features_enhanced.py` exists in `../src/`

**Out of memory:**
- Reduce TF-IDF `max_features` in scripts
- Close other applications

## Next Steps

1. Run experiments: `python run_all_experiments.py`
2. View results: `python visualize_results.py`
3. Update `analytics_experiments_report.txt` with real metrics
4. Use JSON files for tables, charts, and analysis in your report

