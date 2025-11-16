"""
Full Training Script with Detailed Metrics
Trains the primary model and generates comprehensive performance metrics
UPDATED: Ensures NumPy 1.24.3 compatibility for Lambda deployment
"""
import os
import sys
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import joblib
import time

# Verify NumPy version compatibility (must be < 2.0 for Lambda compatibility)
def verify_numpy_version():
    """Verify NumPy version is compatible with Lambda (1.24.3 recommended)"""
    np_version = np.__version__
    print(f"NumPy version: {np_version}")
    
    # Parse version
    major, minor, patch = map(int, np_version.split('.')[:3])
    
    if major >= 2:
        print(f"❌ ERROR: NumPy {np_version} is not compatible with Lambda!")
        print("Models trained with NumPy 2.0+ cannot be loaded in Lambda.")
        print("Please install NumPy 1.24.3: pip install numpy==1.24.3")
        raise RuntimeError(f"NumPy version {np_version} is incompatible. Use NumPy 1.24.3")
    elif major == 1 and minor < 24:
        print(f"⚠️  WARNING: NumPy {np_version} may have compatibility issues.")
        print("Recommended: NumPy 1.24.3")
    else:
        print(f"✓ NumPy {np_version} is compatible with Lambda")
    
    # Check if show_config exists (removed in NumPy 2.0+)
    if not hasattr(np, 'show_config'):
        print("⚠️  WARNING: NumPy show_config not available - this may cause issues with scikit-learn")
    else:
        print("✓ NumPy show_config available (required for scikit-learn)")
    
    return np_version

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features_enhanced import build_linguistic_feature_matrix

warnings.filterwarnings("ignore")

# Configuration
DATASETS_DIR = "../datasets"
RESULTS_DIR = "./results"
MODELS_DIR = "../models"
SEED = 42

def ensure_dirs():
    """Create necessary directories"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def load_datasets() -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """Load and combine all available datasets"""
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    frames = []
    
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
    for filename, label, source in [("True.csv", 0, "true_csv"), ("Fake.csv", 1, "fake_csv")]:
        filepath = os.path.join(DATASETS_DIR, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
                df_processed = df[["text"]].copy()
                df_processed["title"] = ""
                df_processed["text_full"] = df_processed["text"]
                df_processed["label"] = label
                df_processed["source"] = source
                frames.append(df_processed)
                print(f"✓ Loaded {source}: {len(df_processed)} samples")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
    
    if not frames:
        raise FileNotFoundError("No dataset files found")
    
    combined_df = pd.concat(frames, ignore_index=True)
    
    print(f"\n{'=' * 80}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(combined_df)}")
    print(f"Label distribution:")
    label_counts = combined_df['label'].value_counts()
    print(f"  Real (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(combined_df)*100:.1f}%)")
    print(f"  Fake (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(combined_df)*100:.1f}%)")
    print(f"Source distribution:")
    for source, count in combined_df['source'].value_counts().items():
        print(f"  {source}: {count}")
    
    X_texts = combined_df["text_full"].astype(str).tolist()
    y = combined_df["label"].astype(int).values
    
    return X_texts, y, combined_df

def extract_features(X_texts: List[str]) -> Tuple:
    """Extract all features (TF-IDF + Enhanced)"""
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION")
    print("=" * 80)
    
    # TF-IDF
    print("\n[1/3] Extracting TF-IDF features...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(X_texts)
    tfidf_time = time.time() - start_time
    print(f"✓ TF-IDF shape: {X_tfidf.shape}")
    print(f"✓ Time: {tfidf_time:.2f}s")
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Enhanced linguistic features
    print("\n[2/3] Extracting enhanced linguistic features...")
    start_time = time.time()
    X_ling = build_linguistic_feature_matrix(X_texts)
    ling_time = time.time() - start_time
    print(f"✓ Enhanced features shape: {X_ling.shape}")
    print(f"✓ Time: {ling_time:.2f}s")
    print(f"✓ Features: caps_density, exclamation_density, question_density, emotional_words,")
    print(f"           clickbait_patterns, urgency_words, extreme_adjectives, quotes,")
    print(f"           entities, avg_word_length, avg_sentence_length, pronoun_density")
    
    # Scale enhanced features
    print("\n[3/3] Scaling enhanced features...")
    scaler = StandardScaler()
    X_ling_scaled = scaler.fit_transform(X_ling)
    
    # Combine
    X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_ling_scaled)], format="csr")
    print(f"✓ Combined features shape: {X_combined.shape}")
    
    return X_combined, vectorizer, scaler, X_tfidf, X_ling_scaled

def train_model(X_train, y_train) -> Tuple[LinearSVC, float]:
    """Train the primary LinearSVC model"""
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)
    
    print("\nTraining Linear SVM with combined features...")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Model: LinearSVC(max_iter=1000, class_weight='balanced', random_state={SEED})")
    
    start_time = time.time()
    model = LinearSVC(max_iter=1000, class_weight='balanced', random_state=SEED)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} minutes)")
    
    return model, training_time

def evaluate_model(model, X_test, y_test, X_train, y_train) -> Dict:
    """Comprehensive model evaluation"""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Test set predictions
    print("\n[1/3] Test Set Evaluation...")
    y_pred_test = model.predict(X_test)
    y_scores_test = model.decision_function(X_test)
    
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores_test))
    }
    
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    
    test_metrics["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }
    
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Negative:  {tn:5d} (Credible correctly identified)")
    print(f"    False Positive: {fp:5d} (Credible misclassified as Sensational)")
    print(f"    False Negative: {fn:5d} (Sensational misclassified as Credible)")
    print(f"    True Positive:  {tp:5d} (Sensational correctly identified)")
    
    # Training set evaluation (sanity check)
    print("\n[2/3] Training Set Evaluation (Sanity Check)...")
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    
    # Classification report
    print("\n[3/3] Detailed Classification Report:")
    print("\n" + classification_report(y_test, y_pred_test, 
                                      target_names=['Credible (0)', 'Sensational (1)']))
    
    return {
        "test_metrics": test_metrics,
        "train_accuracy": float(train_accuracy)
    }

def cross_validate_model(X_combined, y) -> Dict:
    """Perform cross-validation"""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)
    
    print("\nRunning 5-fold cross-validation...")
    model = LinearSVC(max_iter=1000, class_weight='balanced', random_state=SEED)
    
    cv_scores = cross_val_score(model, X_combined, y, cv=5, scoring='f1')
    
    cv_results = {
        "cv_scores": [float(s) for s in cv_scores],
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_min": float(cv_scores.min()),
        "cv_max": float(cv_scores.max())
    }
    
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean F1:     {cv_results['cv_mean']:.4f}")
    print(f"  Std Dev:     {cv_results['cv_std']:.4f}")
    print(f"  Range:       [{cv_results['cv_min']:.4f}, {cv_results['cv_max']:.4f}]")
    
    return cv_results

def save_model_artifacts(model, vectorizer, scaler):
    """Save trained model and artifacts"""
    print("\n" + "=" * 80)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = os.path.join(MODELS_DIR, f"sensationalism_model_{timestamp}.joblib")
    vec_path = os.path.join(MODELS_DIR, f"tfidf_vectorizer_{timestamp}.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{timestamp}.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Model saved to:      {model_path}")
    print(f"✓ Vectorizer saved to: {vec_path}")
    print(f"✓ Scaler saved to:     {scaler_path}")
    
    # Also save as "latest" for easy reference
    joblib.dump(model, os.path.join(MODELS_DIR, "sensationalism_model_latest.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer_latest.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_latest.joblib"))
    
    print(f"✓ Also saved as 'latest' versions")

def save_results(results: Dict, timestamp: str):
    """Save comprehensive results to JSON"""
    filepath = os.path.join(RESULTS_DIR, f"training_results_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filepath}")

def main():
    """Main execution function"""
    ensure_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL TRAINING")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    # Verify NumPy version before training
    print("\n" + "=" * 80)
    print("VERIFYING ENVIRONMENT COMPATIBILITY")
    print("=" * 80)
    try:
        np_version = verify_numpy_version()
        print(f"✓ Environment verified - NumPy {np_version}")
    except RuntimeError as e:
        print(f"❌ {e}")
        print("\nPlease fix the NumPy version and try again:")
        print("  pip install numpy==1.24.3 scipy==1.10.1 scikit-learn==1.3.0")
        return 1
    print()
    
    # Load data
    X_texts, y, df = load_datasets()
    
    # Extract features
    X_combined, vectorizer, scaler, X_tfidf, X_ling = extract_features(X_texts)
    
    # Train/test split
    print("\n" + "=" * 80)
    print("DATA SPLIT")
    print("=" * 80)
    print(f"\nSplitting data (80% train, 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set:     {X_test.shape[0]} samples")
    print(f"✓ Train label distribution: Real={sum(y_train==0)}, Fake={sum(y_train==1)}")
    print(f"✓ Test label distribution:  Real={sum(y_test==0)}, Fake={sum(y_test==1)}")
    
    # Train model
    model, training_time = train_model(X_train, y_train)
    
    # Evaluate model
    eval_results = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Cross-validation
    cv_results = cross_validate_model(X_combined, y)
    
    # Compile comprehensive results
    comprehensive_results = {
        "experiment_info": {
            "timestamp": timestamp,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "random_seed": SEED
        },
        "dataset_info": {
            "total_samples": len(X_texts),
            "real_samples": int(sum(y == 0)),
            "fake_samples": int(sum(y == 1)),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0])
        },
        "feature_info": {
            "tfidf_features": int(X_tfidf.shape[1]),
            "enhanced_features": int(X_ling.shape[1]),
            "total_features": int(X_combined.shape[1])
        },
        "training_info": {
            "model_type": "LinearSVC",
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60
        },
        "test_performance": eval_results["test_metrics"],
        "train_performance": {
            "accuracy": eval_results["train_accuracy"]
        },
        "cross_validation": cv_results
    }
    
    # Save model artifacts (with comprehensive naming for Lambda)
    save_model_artifacts(model, vectorizer, scaler)
    
    # Also save as "comprehensive" versions for Lambda
    print("\nSaving 'comprehensive' versions for Lambda deployment...")
    joblib.dump(model, os.path.join(MODELS_DIR, "sensationalism_model_comprehensive.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer_comprehensive.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_comprehensive.joblib"))
    print("✓ Saved comprehensive versions for Lambda")
    
    # Save results
    save_results(comprehensive_results, timestamp)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Dataset Size:    {len(X_texts)} samples")
    print(f"Features:        {X_combined.shape[1]} (TF-IDF: {X_tfidf.shape[1]}, Enhanced: {X_ling.shape[1]})")
    print(f"Training Time:   {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"\nTest Performance:")
    print(f"  Accuracy:  {eval_results['test_metrics']['accuracy']:.4f}")
    print(f"  Precision: {eval_results['test_metrics']['precision']:.4f}")
    print(f"  Recall:    {eval_results['test_metrics']['recall']:.4f}")
    print(f"  F1-Score:  {eval_results['test_metrics']['f1_score']:.4f}")
    print(f"  ROC-AUC:   {eval_results['test_metrics']['roc_auc']:.4f}")
    print(f"\nCross-Validation:")
    print(f"  Mean F1:   {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("Finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

if __name__ == "__main__":
    main()

