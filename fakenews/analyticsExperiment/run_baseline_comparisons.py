"""
Baseline Model Comparison Script
Trains multiple baseline models and compares their performance
"""
import os
import sys
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features_enhanced import build_linguistic_feature_matrix

warnings.filterwarnings("ignore")

# Configuration
DATASETS_DIR = "../datasets"
RESULTS_DIR = "./results"
SEED = 42

def ensure_dirs():
    """Create necessary directories"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def load_datasets() -> Tuple[List[str], np.ndarray]:
    """Load and combine all available datasets"""
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    frames = []
    
    # BuzzFeed and PolitiFact datasets
    dataset_files = [
        ("BuzzFeed_fake_news_content.csv", 1, "buzzfeed_fake"),
        ("BuzzFeed_real_news_content.csv", 0, "buzzfeed_real"),
        ("PolitiFact_fake_news_content.csv", 1, "politifact_fake"),
        ("PolitiFact_real_news_content.csv", 0, "politifact_real"),
    ]
    
    for filename, label, source in dataset_files:
        filepath = os.path.join(DATASETS_DIR, filename)
        if os.path.exists(filepath):
            try:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='latin-1')
                
                df = df[["title", "text"]].copy()
                df["text_full"] = (df["title"].fillna("") + "\n\n" + df["text"].fillna(""))
                df["label"] = label
                df["source"] = source
                frames.append(df)
                print(f"✓ Loaded {source}: {len(df)} samples")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
    
    # True.csv and Fake.csv
    true_path = os.path.join(DATASETS_DIR, "True.csv")
    fake_path = os.path.join(DATASETS_DIR, "Fake.csv")
    
    if os.path.exists(true_path):
        try:
            df_true = pd.read_csv(true_path, encoding='latin-1')
            df_true_processed = df_true[["text"]].copy()
            df_true_processed["title"] = ""
            df_true_processed["text_full"] = df_true_processed["text"]
            df_true_processed["label"] = 0
            df_true_processed["source"] = "true_csv"
            frames.append(df_true_processed)
            print(f"✓ Loaded True.csv: {len(df_true_processed)} samples")
        except Exception as e:
            print(f"✗ Error loading True.csv: {e}")
    
    if os.path.exists(fake_path):
        try:
            df_fake = pd.read_csv(fake_path, encoding='latin-1')
            df_fake_processed = df_fake[["text"]].copy()
            df_fake_processed["title"] = ""
            df_fake_processed["text_full"] = df_fake_processed["text"]
            df_fake_processed["label"] = 1
            df_fake_processed["source"] = "fake_csv"
            frames.append(df_fake_processed)
            print(f"✓ Loaded Fake.csv: {len(df_fake_processed)} samples")
        except Exception as e:
            print(f"✗ Error loading Fake.csv: {e}")
    
    if not frames:
        raise FileNotFoundError("No dataset files found in " + DATASETS_DIR)
    
    combined_df = pd.concat(frames, ignore_index=True)
    
    print(f"\n{'=' * 80}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(combined_df)}")
    print(f"Label distribution:")
    print(f"  Real (0): {(combined_df['label'] == 0).sum()}")
    print(f"  Fake (1): {(combined_df['label'] == 1).sum()}")
    print(f"Source distribution:")
    for source, count in combined_df['source'].value_counts().items():
        print(f"  {source}: {count}")
    
    X_texts = combined_df["text_full"].astype(str).tolist()
    y = combined_df["label"].astype(int).values
    
    return X_texts, y

def extract_tfidf_features(X_texts: List[str], max_features: int = 5000) -> Tuple:
    """Extract TF-IDF features"""
    print(f"\nExtracting TF-IDF features (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(X_texts)
    print(f"✓ TF-IDF shape: {X_tfidf.shape}")
    return X_tfidf, vectorizer

def extract_enhanced_features(X_texts: List[str]) -> Tuple:
    """Extract enhanced linguistic features"""
    print("\nExtracting enhanced linguistic features...")
    X_ling = build_linguistic_feature_matrix(X_texts)
    
    scaler = StandardScaler()
    X_ling_scaled = scaler.fit_transform(X_ling)
    
    print(f"✓ Enhanced features shape: {X_ling_scaled.shape}")
    return X_ling_scaled, scaler

def combine_features(X_tfidf, X_ling_scaled):
    """Combine TF-IDF and enhanced features"""
    print("\nCombining features...")
    X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_ling_scaled)], format="csr")
    print(f"✓ Combined features shape: {X_combined.shape}")
    return X_combined

def evaluate_model(model, X_test, y_test, model_name: str) -> Dict:
    """Evaluate a trained model and return metrics"""
    print(f"\nEvaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probability scores for ROC-AUC
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except:
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {tn}, FP: {fp}")
    print(f"    FN: {fn}, TP: {tp}")
    
    return results

def train_and_evaluate_baselines(X_train, X_test, y_train, y_test) -> List[Dict]:
    """Train and evaluate all baseline models"""
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISONS")
    print("=" * 80)
    
    results = []
    
    # Baseline 1: Gaussian Naive Bayes (works with continuous/scaled features)
    print("\n[1/5] Training Gaussian Naive Bayes...")
    # Convert sparse matrix to dense for GaussianNB
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    nb_model = GaussianNB()
    nb_model.fit(X_train_dense, y_train)
    # Evaluate with dense matrix
    y_pred_nb = nb_model.predict(X_test_dense)
    y_scores_nb = nb_model.predict_proba(X_test_dense)[:, 1]
    
    nb_results = {
        "model_name": "Gaussian Naive Bayes",
        "accuracy": float(accuracy_score(y_test, y_pred_nb)),
        "precision": float(precision_score(y_test, y_pred_nb, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_nb, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_nb, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores_nb)),
        "confusion_matrix": {}
    }
    cm = confusion_matrix(y_test, y_pred_nb)
    tn, fp, fn, tp = cm.ravel()
    nb_results["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }
    print(f"  Accuracy:  {nb_results['accuracy']:.4f}")
    print(f"  F1-Score:  {nb_results['f1_score']:.4f}")
    results.append(nb_results)
    
    # Baseline 2: Logistic Regression
    print("\n[2/5] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    lr_model.fit(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results.append(lr_results)
    
    # Baseline 3: Linear SVM (Our primary model)
    print("\n[3/5] Training Linear SVM...")
    svm_model = LinearSVC(max_iter=1000, class_weight='balanced', random_state=SEED)
    svm_model.fit(X_train, y_train)
    svm_results = evaluate_model(svm_model, X_test, y_test, "Linear SVM (Primary)")
    results.append(svm_results)
    
    # Baseline 4: Random Forest (small)
    print("\n[4/5] Training Random Forest (100 trees)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', 
                                      random_state=SEED, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest (100)")
    results.append(rf_results)
    
    # Baseline 5: Random Forest (large)
    print("\n[5/5] Training Random Forest (300 trees)...")
    rf_large_model = RandomForestClassifier(n_estimators=300, max_depth=30, class_weight='balanced',
                                           random_state=SEED, n_jobs=-1)
    rf_large_model.fit(X_train, y_train)
    rf_large_results = evaluate_model(rf_large_model, X_test, y_test, "Random Forest (300)")
    results.append(rf_large_results)
    
    return results

def save_results(results: List[Dict], filename: str):
    """Save results to JSON file"""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filepath}")

def print_comparison_table(results: List[Dict]):
    """Print a comparison table of all models"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    
    # Header
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    
    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    for r in sorted_results:
        roc_auc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "N/A"
        print(f"{r['model_name']:<25} {r['accuracy']:<10.4f} {r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} {r['f1_score']:<10.4f} {roc_auc_str:<10}")
    
    print("=" * 80)

def main():
    """Main execution function"""
    ensure_dirs()
    
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON EXPERIMENT")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    # Load data
    X_texts, y = load_datasets()
    
    # Extract features
    X_tfidf, vectorizer = extract_tfidf_features(X_texts)
    X_ling, scaler = extract_enhanced_features(X_texts)
    X_combined = combine_features(X_tfidf, X_ling)
    
    # Train/test split
    print(f"\nSplitting data (80/20 train/test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"✓ Train size: {X_train.shape[0]}")
    print(f"✓ Test size:  {X_test.shape[0]}")
    
    # Train and evaluate baselines
    results = train_and_evaluate_baselines(X_train, X_test, y_train, y_test)
    
    # Print comparison
    print_comparison_table(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"baseline_comparison_{timestamp}.json")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("Finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

if __name__ == "__main__":
    main()

