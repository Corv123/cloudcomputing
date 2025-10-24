# Analytics & Experiment Scripts

This folder contains scripts to conduct comprehensive experiments on the Fake News Detector ML model.

## 📁 Contents

### Experiment Scripts

1. **`run_full_training.py`**
   - Trains the primary LinearSVC model with complete feature set
   - Generates comprehensive performance metrics
   - Performs 5-fold cross-validation
   - Saves model artifacts and detailed results
   - **Output**: Training metrics, confusion matrix, cross-validation scores

2. **`run_baseline_comparisons.py`**
   - Compares multiple baseline models
   - Models tested:
     - Naive Bayes
     - Logistic Regression
     - Linear SVM (our primary model)
     - Random Forest (100 trees)
     - Random Forest (300 trees)
   - **Output**: Comparative performance table with accuracy, precision, recall, F1, ROC-AUC

3. **`run_ablation_study.py`**
   - Systematically removes feature groups to measure contribution
   - Experiments:
     - Full Model (baseline)
     - TF-IDF only
     - Enhanced features only
     - Remove stylistic features
     - Remove emotional/clickbait features
     - Remove credibility markers
     - Remove structural features
   - **Output**: Ablation results table with delta F1 scores

4. **`run_all_experiments.py`**
   - Master script that runs all experiments in sequence
   - Provides comprehensive summary
   - **Usage**: Run this to execute all experiments at once

## 🚀 Usage

### Prerequisites

Ensure you have the datasets in the `../datasets/` directory:
- `BuzzFeed_fake_news_content.csv`
- `BuzzFeed_real_news_content.csv`
- `PolitiFact_fake_news_content.csv`
- `PolitiFact_real_news_content.csv`
- `True.csv`
- `Fake.csv`

### Running Experiments

**Option 1: Run all experiments at once**
```bash
cd fakenews/analytics&experiment
python run_all_experiments.py
```

**Option 2: Run individual experiments**
```bash
# Full training
python run_full_training.py

# Baseline comparisons
python run_baseline_comparisons.py

# Ablation study
python run_ablation_study.py
```

## 📊 Output

All results are saved in the `./results/` directory with timestamps:

- `training_results_YYYYMMDD_HHMMSS.json` - Full training metrics
- `baseline_comparison_YYYYMMDD_HHMMSS.json` - Baseline model comparison
- `ablation_study_YYYYMMDD_HHMMSS.json` - Ablation study results

Model artifacts are saved in `../models/`:
- `sensationalism_model_YYYYMMDD_HHMMSS.joblib`
- `tfidf_vectorizer_YYYYMMDD_HHMMSS.joblib`
- `scaler_YYYYMMDD_HHMMSS.joblib`
- `*_latest.joblib` - Latest versions for easy reference

## 📈 Expected Results

### Full Training
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrix**: TP, TN, FP, FN
- **Cross-Validation**: Mean F1 with standard deviation

### Baseline Comparisons
- Performance comparison across 5 different models
- Identifies best performing model
- Helps justify choice of Linear SVM

### Ablation Study
- Quantifies contribution of each feature group
- Shows delta F1 when features are removed
- Validates that all features contribute meaningfully

## 🔧 Configuration

Key parameters can be modified in each script:
- `DATASETS_DIR`: Path to datasets (default: `../datasets`)
- `RESULTS_DIR`: Output directory (default: `./results`)
- `SEED`: Random seed for reproducibility (default: 42)
- TF-IDF parameters: `max_features=5000`, `ngram_range=(1,2)`, etc.

## 📝 Notes

- **Runtime**: Full experiment suite takes 15-45 minutes depending on dataset size
- **Memory**: Requires ~4-8GB RAM for large datasets
- **Dependencies**: scikit-learn, pandas, numpy, scipy, joblib
- **Reproducibility**: All experiments use `SEED=42` for consistent results

## 🐛 Troubleshooting

**Dataset not found error:**
- Ensure datasets are in `../datasets/` directory
- Check file names match exactly

**Memory error:**
- Reduce `max_features` in TF-IDF vectorizer
- Use smaller Random Forest (reduce `n_estimators`)

**ImportError:**
- Ensure `features_enhanced.py` is in `../src/` directory
- Check all dependencies are installed: `pip install -r ../requirements.txt`

## 📊 Using Results in Reports

The JSON output files contain all metrics needed for academic reports:

```python
import json

# Load results
with open('results/training_results_YYYYMMDD_HHMMSS.json', 'r') as f:
    results = json.load(f)

# Access metrics
print(f"Accuracy: {results['test_performance']['accuracy']}")
print(f"F1-Score: {results['test_performance']['f1_score']}")
print(f"Confusion Matrix: {results['test_performance']['confusion_matrix']}")
```

## 🎯 Next Steps

After running experiments:
1. Review JSON results files
2. Update `analytics_experiments_report.txt` with actual metrics
3. Use confusion matrices for error analysis
4. Compare ablation results to validate feature engineering
5. Include baseline comparison table in final report

