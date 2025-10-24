"""
Ablation Study Script
Systematically removes feature groups to measure their contribution
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
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features_enhanced import EnhancedLinguisticFeatures, extract_enhanced_linguistic_features

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
    print("Loading datasets...")
    
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
                print(f"  ✓ {source}: {len(df)} samples")
            except Exception as e:
                print(f"  ✗ {filename}: {e}")
    
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
                print(f"  ✓ {source}: {len(df_processed)} samples")
            except Exception as e:
                print(f"  ✗ {filename}: {e}")
    
    if not frames:
        raise FileNotFoundError("No dataset files found")
    
    combined_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(combined_df)} samples")
    
    X_texts = combined_df["text_full"].astype(str).tolist()
    y = combined_df["label"].astype(int).values
    
    return X_texts, y

def extract_tfidf_features(X_texts: List[str]) -> Tuple:
    """Extract TF-IDF features"""
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
    return X_tfidf, vectorizer

def extract_enhanced_features_custom(X_texts: List[str], feature_mask: Dict[str, bool]) -> np.ndarray:
    """
    Extract enhanced features with selective feature groups
    
    feature_mask: dictionary specifying which feature groups to include
    """
    print(f"\nExtracting features with mask: {feature_mask}")
    
    all_features = []
    
    for text in X_texts:
        features_obj = extract_enhanced_linguistic_features(text)
        feature_vector = []
        
        # Stylistic features (caps, punctuation)
        if feature_mask.get('stylistic', True):
            feature_vector.extend([
                features_obj.caps_density,
                features_obj.exclamation_density,
                features_obj.question_density
            ])
        
        # Emotional/Clickbait features
        if feature_mask.get('emotional', True):
            feature_vector.extend([
                features_obj.emotional_word_count,
                features_obj.clickbait_matches,
                features_obj.emotional_intensity,
                features_obj.clickbait_score
            ])
        
        # Credibility markers
        if feature_mask.get('credibility', True):
            feature_vector.extend([
                features_obj.professional_word_count,
                features_obj.balanced_word_count,
                features_obj.evidence_ratio
            ])
        
        # Structural features
        if feature_mask.get('structural', True):
            feature_vector.extend([
                features_obj.avg_word_length,
                features_obj.avg_sentence_length
            ])
        
        all_features.append(feature_vector)
    
    return np.array(all_features)

def run_ablation_experiment(
    X_tfidf, 
    X_texts: List[str],
    y: np.ndarray,
    experiment_name: str,
    use_tfidf: bool = True,
    feature_mask: Dict[str, bool] = None
) -> Dict:
    """
    Run a single ablation experiment
    
    Args:
        X_tfidf: Pre-computed TF-IDF features
        X_texts: Original text data
        y: Labels
        experiment_name: Name of this experiment
        use_tfidf: Whether to include TF-IDF features
        feature_mask: Which enhanced feature groups to include
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    
    # Default: use all feature groups
    if feature_mask is None:
        feature_mask = {
            'stylistic': True,
            'emotional': True,
            'credibility': True,
            'structural': True
        }
    
    # Extract enhanced features
    X_enhanced = extract_enhanced_features_custom(X_texts, feature_mask)
    
    # Check if we have any enhanced features to scale
    if X_enhanced.shape[1] > 0:
        # Scale enhanced features
        scaler = StandardScaler()
        X_enhanced_scaled = scaler.fit_transform(X_enhanced)
    else:
        # No enhanced features (TF-IDF only experiment)
        X_enhanced_scaled = X_enhanced  # Empty array
    
    # Combine features based on configuration
    if use_tfidf and X_enhanced_scaled.shape[1] > 0:
        X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_enhanced_scaled)], format="csr")
        print(f"Features: TF-IDF ({X_tfidf.shape[1]}) + Enhanced ({X_enhanced_scaled.shape[1]}) = {X_combined.shape[1]}")
    elif use_tfidf:
        X_combined = X_tfidf
        print(f"Features: TF-IDF only ({X_tfidf.shape[1]})")
    elif X_enhanced_scaled.shape[1] > 0:
        X_combined = sparse.csr_matrix(X_enhanced_scaled)
        print(f"Features: Enhanced only ({X_enhanced_scaled.shape[1]})")
    else:
        raise ValueError("No features selected!")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Train model
    print(f"\nTraining Linear SVM...")
    model = LinearSVC(max_iter=1000, class_weight='balanced', random_state=SEED)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_scores)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return {
        "experiment_name": experiment_name,
        "use_tfidf": use_tfidf,
        "feature_mask": feature_mask,
        "num_features": X_combined.shape[1],
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }

def calculate_deltas(results: List[Dict], baseline_name: str = "Full Model") -> List[Dict]:
    """Calculate performance deltas relative to baseline"""
    baseline = next((r for r in results if r['experiment_name'] == baseline_name), None)
    
    if not baseline:
        print(f"Warning: Baseline '{baseline_name}' not found")
        return results
    
    baseline_f1 = baseline['f1_score']
    
    for result in results:
        result['delta_f1'] = result['f1_score'] - baseline_f1
        result['delta_f1_pct'] = (result['delta_f1'] / baseline_f1) * 100
    
    return results

def print_ablation_table(results: List[Dict]):
    """Print ablation study results table"""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    
    print(f"{'Configuration':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
          f"{'F1':<10} {'Δ F1':<10} {'Δ F1 %':<10}")
    print("-" * 100)
    
    # Sort by F1 score descending
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    for r in sorted_results:
        delta_str = f"{r.get('delta_f1', 0):.4f}" if 'delta_f1' in r else "baseline"
        delta_pct_str = f"{r.get('delta_f1_pct', 0):+.2f}%" if 'delta_f1_pct' in r else "--"
        
        print(f"{r['experiment_name']:<35} {r['accuracy']:.4f}     {r['precision']:.4f}     "
              f"{r['recall']:.4f}     {r['f1_score']:.4f}     {delta_str:<10} {delta_pct_str:<10}")
    
    print("=" * 100)

def main():
    """Main execution function"""
    ensure_dirs()
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY EXPERIMENT")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    # Load data
    X_texts, y = load_datasets()
    
    # Extract TF-IDF once (reused across experiments)
    print("\nExtracting TF-IDF features...")
    X_tfidf, _ = extract_tfidf_features(X_texts)
    print(f"✓ TF-IDF shape: {X_tfidf.shape}")
    
    # Run ablation experiments
    results = []
    
    # Experiment 1: Full Model (Baseline)
    results.append(run_ablation_experiment(
        X_tfidf, X_texts, y,
        experiment_name="Full Model",
        use_tfidf=True,
        feature_mask={'stylistic': True, 'emotional': True, 'credibility': True, 'structural': True}
    ))
    
    # Experiment 2: TF-IDF Only (no enhanced features)
    results.append(run_ablation_experiment(
        X_tfidf, X_texts, y,
        experiment_name="TF-IDF Only",
        use_tfidf=True,
        feature_mask={'stylistic': False, 'emotional': False, 'credibility': False, 'structural': False}
    ))
    
    # Experiment 3: Enhanced Features Only (no TF-IDF)
    results.append(run_ablation_experiment(
        X_tfidf, X_texts, y,
        experiment_name="Enhanced Features Only",
        use_tfidf=False,
        feature_mask={'stylistic': True, 'emotional': True, 'credibility': True, 'structural': True}
    ))
    
    # Experiment 4: Remove Stylistic Features
    results.append(run_ablation_experiment(
        X_tfidf, X_texts, y,
        experiment_name="- Stylistic (caps, punctuation)",
        use_tfidf=True,
        feature_mask={'stylistic': False, 'emotional': True, 'credibility': True, 'structural': True}
    ))
    
    # Experiment 5: Remove Emotional/Clickbait Features
    results.append(run_ablation_experiment(
        X_tfidf, X_texts, y,
        experiment_name="- Emotional/Clickbait",
        use_tfidf=True,
        feature_mask={'stylistic': True, 'emotional': False, 'credibility': True, 'structural': True}
    ))
    
    # Experiment 6: Remove Credibility Markers
    results.append(run_ablation_experiment(
        X_tfidf, X_texts, y,
        experiment_name="- Credibility (quotes, entities)",
        use_tfidf=True,
        feature_mask={'stylistic': True, 'emotional': True, 'credibility': False, 'structural': True}
    ))
    
    # Experiment 7: Remove Structural Features
    results.append(run_ablation_experiment(
        X_tfidf, X_texts, y,
        experiment_name="- Structural (word/sent length)",
        use_tfidf=True,
        feature_mask={'stylistic': True, 'emotional': True, 'credibility': True, 'structural': False}
    ))
    
    # Calculate deltas
    results = calculate_deltas(results, baseline_name="Full Model")
    
    # Print table
    print_ablation_table(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"ablation_study_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filepath}")
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED")
    print("Finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

if __name__ == "__main__":
    main()

