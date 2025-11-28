import argparse
import os
import warnings
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from collections import Counter
import re

# Verify NumPy version compatibility (must be < 2.0 for Lambda compatibility)
def verify_numpy_version():
    """Verify NumPy version is compatible with Lambda (1.24.3 recommended)"""
    np_version = np.__version__
    print(f"NumPy version: {np_version}")
    
    # Parse version
    major, minor, patch = map(int, np_version.split('.')[:3])
    
    if major >= 2:
        print(f"WARNING: NumPy {np_version} is not compatible with Lambda!")
        print("Models trained with NumPy 2.0+ cannot be loaded in Lambda.")
        print("Please install NumPy 1.24.3: pip install numpy==1.24.3")
        raise RuntimeError(f"NumPy version {np_version} is incompatible. Use NumPy 1.24.3")
    elif major == 1 and minor < 24:
        print(f"WARNING: NumPy {np_version} may have compatibility issues.")
        print("Recommended: NumPy 1.24.3")
    else:
        print(f"✓ NumPy {np_version} is compatible with Lambda")
    
    # Check if show_config exists (removed in NumPy 2.0+)
    if not hasattr(np, 'show_config'):
        print("WARNING: NumPy show_config not available - this may cause issues with scikit-learn")
    
    return np_version

import os as _os, sys as _sys
_SYS_SRC = _os.path.abspath(_os.path.join(_os.path.dirname(__file__)))
if _SYS_SRC not in _sys.path:
	_sys.path.insert(0, _SYS_SRC)
from features_enhanced import build_linguistic_feature_matrix

warnings.filterwarnings("ignore")

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def analyze_dataset_structure(datasets_dir: str) -> Dict:
    """Analyze structure of all available datasets"""
    analysis = {}
    
    # Analyze TrueFalse.csv
    tf_path = os.path.join(datasets_dir, "TrueFalse.csv")
    if os.path.exists(tf_path):
        try:
            df_tf = pd.read_csv(tf_path)
            analysis["TrueFalse"] = {
                "shape": df_tf.shape,
                "columns": list(df_tf.columns),
                "sample_data": {col: str(df_tf[col].iloc[0])[:100] for col in df_tf.columns[:5]},
                "dtypes": df_tf.dtypes.to_dict(),
                "missing": df_tf.isnull().sum().to_dict()
            }
        except Exception as e:
            analysis["TrueFalse"] = {"error": str(e)}
    
    # Analyze WELFake_Dataset.csv
    wel_path = os.path.join(datasets_dir, "WELFake_Dataset.csv")
    if os.path.exists(wel_path):
        try:
            df_wel = pd.read_csv(wel_path, encoding='latin-1', nrows=100)  # Sample for analysis
            analysis["WELFake"] = {
                "shape": df_wel.shape,
                "columns": list(df_wel.columns),
                "sample_data": {col: str(df_wel[col].iloc[0])[:80] if len(df_wel) > 0 else "" for col in df_wel.columns[:3]},
                "dtypes": df_wel.dtypes.to_dict(),
                "label_values": df_wel["label"].unique().tolist() if "label" in df_wel.columns else []
            }
        except Exception as e:
            analysis["WELFake"] = {"error": str(e)}
    
    return analysis

def read_all_datasets(datasets_dir: str) -> pd.DataFrame:
    """Read all available datasets and combine them"""
    frames: List[pd.DataFrame] = []
    
    print("Loading all available datasets...")
    
    # Existing BuzzFeed and PolitiFact CSV datasets
    existing_paths = [
        (os.path.join(datasets_dir, "BuzzFeed_fake_news_content.csv"), 1, "buzzfeed_fake"),
        (os.path.join(datasets_dir, "BuzzFeed_real_news_content.csv"), 0, "buzzfeed_real"),
        (os.path.join(datasets_dir, "PolitiFact_fake_news_content.csv"), 1, "politifact_fake"),
        (os.path.join(datasets_dir, "PolitiFact_real_news_content.csv"), 0, "politifact_real"),
    ]
    
    for p, label, source in existing_paths:
        if os.path.exists(p):
            try:
                # Try UTF-8 first, then fallback to latin-1
                try:
                    df = pd.read_csv(p, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(p, encoding='latin-1')
                
                df = df[["title", "text"]].copy()
                df["text_full"] = (df["title"].fillna("") + "\n\n" + df["text"].fillna(""))
                df["label"] = label
                df["source"] = source
                frames.append(df)
                print(f"  Loaded {source}: {len(df)} samples")
            except Exception as e:
                print(f"  Error loading {p}: {e}")
    
    # Add corrected True.csv and Fake.csv files (load in chunks to avoid memory errors)
    true_path = os.path.join(datasets_dir, "True.csv")
    fake_path = os.path.join(datasets_dir, "Fake.csv")
    
    # Load True.csv (Real content - label as 0) in chunks
    if os.path.exists(true_path):
        try:
            print(f"  Loading True.csv in chunks...")
            chunk_list = []
            chunk_size = 2000  # Smaller chunks for memory efficiency
            
            # Read CSV in chunks with error handling
            try:
                for chunk in pd.read_csv(true_path, encoding='latin-1', chunksize=chunk_size, 
                                        on_bad_lines='skip', engine='python', low_memory=False):
                    # Process chunk
                    if "text" in chunk.columns:
                        chunk_processed = chunk[["text"]].copy()
                        chunk_processed["title"] = ""
                        chunk_processed["text_full"] = chunk_processed["text"]
                        chunk_processed["label"] = 0  # Real content
                        chunk_processed["source"] = "true_corrected"
                        chunk_list.append(chunk_processed)
                    else:
                        print(f"    Warning: Chunk missing 'text' column, columns: {list(chunk.columns)}")
            except Exception as chunk_error:
                print(f"    Warning: Error reading chunks, trying alternative method: {chunk_error}")
                # Fallback: try reading with different parameters
                try:
                    df_true = pd.read_csv(true_path, encoding='latin-1', nrows=1000)  # Test with first 1000 rows
                    print(f"    Test read successful, columns: {list(df_true.columns)}")
                    # If test works, try full read with different engine
                    df_true = pd.read_csv(true_path, encoding='latin-1', engine='python', on_bad_lines='skip')
                    if "text" in df_true.columns:
            df_true_processed = df_true[["text"]].copy()
            df_true_processed["title"] = ""
            df_true_processed["text_full"] = df_true_processed["text"]
                        df_true_processed["label"] = 0
            df_true_processed["source"] = "true_corrected"
                        chunk_list = [df_true_processed]
                except Exception as fallback_error:
                    print(f"    Fallback also failed: {fallback_error}")
                    raise
            
            # Combine all chunks
            if chunk_list:
                df_true_processed = pd.concat(chunk_list, ignore_index=True)
            frames.append(df_true_processed)
            print(f"  Loaded True.csv (Real content): {len(df_true_processed)} samples")
            else:
                print(f"  Warning: True.csv had no data")
        except Exception as e:
            print(f"  Error loading True.csv: {e}")
            import traceback
            traceback.print_exc()
    
    # Load Fake.csv (Fake content - label as 1) in chunks
    if os.path.exists(fake_path):
        try:
            print(f"  Loading Fake.csv in chunks...")
            chunk_list = []
            chunk_size = 2000  # Smaller chunks for memory efficiency
            
            # Read CSV in chunks with error handling
            try:
                for chunk in pd.read_csv(fake_path, encoding='latin-1', chunksize=chunk_size,
                                        on_bad_lines='skip', engine='python', low_memory=False):
                    # Process chunk
                    if "text" in chunk.columns:
                        chunk_processed = chunk[["text"]].copy()
                        chunk_processed["title"] = ""
                        chunk_processed["text_full"] = chunk_processed["text"]
                        chunk_processed["label"] = 1  # Fake content
                        chunk_processed["source"] = "fake_corrected"
                        chunk_list.append(chunk_processed)
                    else:
                        print(f"    Warning: Chunk missing 'text' column, columns: {list(chunk.columns)}")
            except Exception as chunk_error:
                print(f"    Warning: Error reading chunks, trying alternative method: {chunk_error}")
                # Fallback: try reading with different parameters
                try:
                    df_fake = pd.read_csv(fake_path, encoding='latin-1', nrows=1000)  # Test with first 1000 rows
                    print(f"    Test read successful, columns: {list(df_fake.columns)}")
                    # If test works, try full read with different engine
                    df_fake = pd.read_csv(fake_path, encoding='latin-1', engine='python', on_bad_lines='skip')
                    if "text" in df_fake.columns:
            df_fake_processed = df_fake[["text"]].copy()
            df_fake_processed["title"] = ""
            df_fake_processed["text_full"] = df_fake_processed["text"]
                        df_fake_processed["label"] = 1
            df_fake_processed["source"] = "fake_corrected"
                        chunk_list = [df_fake_processed]
                except Exception as fallback_error:
                    print(f"    Fallback also failed: {fallback_error}")
                    raise
            
            # Combine all chunks
            if chunk_list:
                df_fake_processed = pd.concat(chunk_list, ignore_index=True)
            frames.append(df_fake_processed)
            print(f"  Loaded Fake.csv (Fake content): {len(df_fake_processed)} samples")
            else:
                print(f"  Warning: Fake.csv had no data")
        except Exception as e:
            print(f"  Error loading Fake.csv: {e}")
            import traceback
            traceback.print_exc()
    
    # Add WELFake_Dataset.csv
    wel_path = os.path.join(datasets_dir, "WELFake_Dataset.csv")
    if os.path.exists(wel_path):
        try:
            print(f"  Loading WELFake_Dataset.csv...")
            # Try UTF-8 first, then fallback to latin-1
            try:
                df_wel = pd.read_csv(wel_path, encoding='utf-8')
            except UnicodeDecodeError:
                df_wel = pd.read_csv(wel_path, encoding='latin-1')
            
            # Check for required columns
            if "title" not in df_wel.columns or "text" not in df_wel.columns or "label" not in df_wel.columns:
                print(f"    Warning: WELFake_Dataset.csv missing required columns. Found: {list(df_wel.columns)}")
            else:
                # Extract title and text columns
                df_wel_processed = df_wel[["title", "text", "label"]].copy()
                
                # Create text_full from title + text
                df_wel_processed["text_full"] = (
                    df_wel_processed["title"].fillna("") + "\n\n" + 
                    df_wel_processed["text"].fillna("")
                )
                
                # Map label column: "true"/"TRUE" → 0 (real), "false"/"FALSE" → 1 (fake)
                # Handle both string and boolean types, case-insensitive
                def map_label(value):
                    if pd.isna(value):
                        return None
                    # Convert to string and lowercase for comparison
                    str_value = str(value).strip().lower()
                    if str_value in ['true', '1', '1.0', 1]:
                        return 0  # Real content
                    elif str_value in ['false', '0', '0.0', 0]:
                        return 1  # Fake content
                    else:
                        return None  # Unknown value
                
                df_wel_processed["label"] = df_wel_processed["label"].apply(map_label)
                
                # Remove rows with unmapped labels
                df_wel_processed = df_wel_processed.dropna(subset=["label"])
                
                # Convert label to int
                df_wel_processed["label"] = df_wel_processed["label"].astype(int)
                df_wel_processed["source"] = "welfake"
                
                frames.append(df_wel_processed)
                print(f"  Loaded WELFake_Dataset.csv: {len(df_wel_processed)} samples")
                print(f"    Label distribution: {df_wel_processed['label'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  Error loading WELFake_Dataset.csv: {e}")
            import traceback
            traceback.print_exc()
    
    if not frames:
        raise FileNotFoundError("No dataset files found")
    
    combined_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combined dataset: {len(combined_df)} samples")
    print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
    print(f"Source distribution: {combined_df['source'].value_counts().to_dict()}")
    
    return combined_df

def extract_additional_patterns(df: pd.DataFrame) -> Dict:
    """Extract additional patterns from the combined dataset (memory-efficient)"""
    print("\nExtracting additional patterns...")
    
    fake_texts = df[df['label'] == 1]['text_full'].astype(str).tolist()
    real_texts = df[df['label'] == 0]['text_full'].astype(str).tolist()
    
    patterns = {}
    
    # Common sensational phrases in fake news
    sensational_phrases = [
        'breaking', 'exclusive', 'shocking', 'unbelievable', 'you won\'t believe',
        'must see', 'viral', 'outrageous', 'scandal', 'exposed', 'leaked',
        'insider', 'sources say', 'according to sources', 'this changes everything',
        'game changer', 'mind blown', 'jaw dropping', 'stunning', 'incredible',
        'watch video', 'featured image', 'share this', 'go viral'
    ]
    
    # Memory-efficient counting: count phrases in each text individually
    print("  Counting phrases in fake texts...")
    fake_phrase_counts = {phrase: 0 for phrase in sensational_phrases}
    for text in fake_texts:
        text_lower = text.lower()
        for phrase in sensational_phrases:
            fake_phrase_counts[phrase] += text_lower.count(phrase)
    
    print("  Counting phrases in real texts...")
    real_phrase_counts = {phrase: 0 for phrase in sensational_phrases}
    for text in real_texts:
        text_lower = text.lower()
        for phrase in sensational_phrases:
            real_phrase_counts[phrase] += text_lower.count(phrase)
    
    # Find phrases that are significantly more common in fake news
    phrase_ratios = {}
    total_fake = len(fake_texts)
    total_real = len(real_texts)
    
    for phrase in sensational_phrases:
        fake_count = fake_phrase_counts[phrase]
        real_count = real_phrase_counts[phrase]
        
        if total_fake > 0 and total_real > 0:
            fake_ratio = fake_count / total_fake
            real_ratio = real_count / total_real
            if real_ratio > 0:
                phrase_ratios[phrase] = fake_ratio / real_ratio
            elif fake_ratio > 0:
                phrase_ratios[phrase] = float('inf')
    
    # Sort by ratio (phrases more common in fake news)
    top_fake_phrases = sorted(phrase_ratios.items(), key=lambda x: x[1], reverse=True)[:20]
    
    patterns['top_fake_phrases'] = top_fake_phrases
    patterns['fake_phrase_counts'] = fake_phrase_counts
    patterns['real_phrase_counts'] = real_phrase_counts
    
    print(f"Top phrases more common in fake news:")
    for phrase, ratio in top_fake_phrases[:10]:
        print(f"  {phrase}: {ratio:.2f}x more common in fake news")
    
    return patterns

def build_vectorizer(max_features: int, ngrams: Tuple[int, int], min_df: int, max_df: float) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngrams,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,  # Use sublinear TF scaling (1 + log(tf)) for better quality
        norm='l2',          # L2 normalization for better feature scaling
        use_idf=True,       # Use IDF weighting (default, but explicit for clarity)
        smooth_idf=True,    # Smooth IDF weights to avoid division by zero
    )

def train_model(model_type: str):
    if model_type == "logistic_regression":
        # Increased max_iter for better convergence, added dual=False for better performance with many features
        return LogisticRegression(max_iter=5000, class_weight="balanced", dual=False, solver='lbfgs', n_jobs=-1)
    if model_type == "linear_svm":
        # Increased max_iter and tolerance for better convergence
        return LinearSVC(class_weight="balanced", max_iter=10000, tol=1e-5, dual=False)
    if model_type == "random_forest":
        # More trees and deeper trees for better quality
        return RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, 
                                     min_samples_leaf=1, class_weight="balanced", n_jobs=-1, random_state=42)
    raise ValueError(f"Unknown model_type: {model_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "config.yaml"))
    parser.add_argument("--datasets_dir", default="datasets")
    parser.add_argument("--out_model", default=os.path.join("models", "sensationalism_model_comprehensive.joblib"))
    parser.add_argument("--out_vectorizer", default=os.path.join("models", "tfidf_vectorizer_comprehensive.joblib"))
    parser.add_argument("--out_scaler", default=os.path.join("models", "scaler_comprehensive.joblib"))
    args = parser.parse_args()
    
    # Verify NumPy version before training
    print("=" * 80)
    print("VERIFYING ENVIRONMENT COMPATIBILITY")
    print("=" * 80)
    try:
        np_version = verify_numpy_version()
        print(f"✓ Environment verified - NumPy {np_version}")
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1
    print()
    
    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    
    # Analyze dataset structures
    print("=== DATASET STRUCTURE ANALYSIS ===")
    analysis = analyze_dataset_structure(args.datasets_dir)
    for dataset, info in analysis.items():
        print(f"\n{dataset}: {info}")
    
    # Load all datasets
    df = read_all_datasets(args.datasets_dir)
    
    # Extract additional patterns
    patterns = extract_additional_patterns(df)
    
    # Prepare features
    X_texts = df["text_full"].astype(str).tolist()
    y = df["label"].astype(int).values
    
    print("\nBuilding TF-IDF vectorizer...")
    vec = build_vectorizer(
        max_features=int(train_cfg.get("max_features", 50000)),
        ngrams=tuple(train_cfg.get("ngrams", [1, 2])),
        min_df=int(train_cfg.get("min_df", 2)),
        max_df=float(train_cfg.get("max_df", 0.95)),
    )
    X_tfidf = vec.fit_transform(X_texts)
    
    print("Computing linguistic features...")
    X_ling = build_linguistic_feature_matrix(X_texts)
    
    # Scale linguistic features to match inference pipeline
    print("Fitting scaler for linguistic features...")
    scaler = StandardScaler()
    X_ling_scaled = scaler.fit_transform(X_ling)
    
    # Concatenate features
    from scipy import sparse
    X_all = sparse.hstack([X_tfidf, sparse.csr_matrix(X_ling_scaled)], format="csr")
    
    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X_all, y, test_size=float(train_cfg.get("test_size", 0.2)), 
        random_state=int(cfg.get("seed", 42)), stratify=y
    )
    
    print("\nTraining model...")
    model = train_model(train_cfg.get("model_type", "linear_svm"))
    model.fit(x_train, y_train)
    
    print("\nEvaluating on test set...")
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(x_test)
    elif hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(x_test)[:, 1]
    else:
        y_scores = model.predict(x_test)
    
    y_pred = model.predict(x_test)
    
    # Comprehensive evaluation metrics
    try:
        auc = roc_auc_score(y_test, y_scores)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        print("ROC-AUC unavailable for this model.")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake'], digits=4))
    
    # Confusion matrix for better understanding
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real    {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       Fake    {cm[1][0]:5d} {cm[1][1]:5d}")
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Cross-validation with more detailed output
    print(f"\nRunning {train_cfg.get('cv_folds', 10)}-fold cross-validation (this may take a while for best quality)...")
    cv_scores = cross_val_score(model, X_all, y, cv=int(train_cfg.get("cv_folds", 10)), 
                                scoring='roc_auc', n_jobs=-1)
    print(f"Cross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Best fold: {cv_scores.max():.4f}")
    print(f"Worst fold: {cv_scores.min():.4f}")
    
    # Save artifacts
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(model, args.out_model)
    joblib.dump(vec, args.out_vectorizer)
    joblib.dump(scaler, args.out_scaler)
    joblib.dump(patterns, os.path.join("models", "extracted_patterns.joblib"))
    
    print(f"\nSaved model to {args.out_model}")
    print(f"Saved vectorizer to {args.out_vectorizer}")
    print(f"Saved patterns to models/extracted_patterns.joblib")
    print(f"Saved scaler to {args.out_scaler}")

if __name__ == "__main__":
    main()