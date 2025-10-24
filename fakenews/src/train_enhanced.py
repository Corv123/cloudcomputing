import argparse
import os
import warnings
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from collections import Counter
import re

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
    
    # Analyze Politifact.xlsx
    pf_path = os.path.join(datasets_dir, "Politifact.xlsx")
    if os.path.exists(pf_path):
        try:
            excel_file = pd.ExcelFile(pf_path)
            analysis["Politifact"] = {"sheets": excel_file.sheet_names}
            
            for sheet_name in excel_file.sheet_names:
                df_pf = pd.read_excel(pf_path, sheet_name=sheet_name)
                analysis["Politifact"][sheet_name] = {
                    "shape": df_pf.shape,
                    "columns": list(df_pf.columns),
                    "sample_data": {col: str(df_pf[col].iloc[0])[:80] for col in df_pf.columns[:3]},
                    "dtypes": df_pf.dtypes.to_dict()
                }
        except Exception as e:
            analysis["Politifact"] = {"error": str(e)}
    
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
    
    # Add corrected True.csv and Fake.csv files
    true_path = os.path.join(datasets_dir, "True.csv")
    fake_path = os.path.join(datasets_dir, "Fake.csv")
    
    # Load True.csv (Real content - label as 0)
    if os.path.exists(true_path):
        try:
            df_true = pd.read_csv(true_path, encoding='latin-1')
            print(f"  True.csv structure: {list(df_true.columns)}")
            
            # Process True.csv - these are REAL content, label as 0
            df_true_processed = df_true[["text"]].copy()
            df_true_processed["title"] = ""
            df_true_processed["text_full"] = df_true_processed["text"]
            df_true_processed["label"] = 0  # Real content
            df_true_processed["source"] = "true_corrected"
            frames.append(df_true_processed)
            print(f"  Loaded True.csv (Real content): {len(df_true_processed)} samples")
        except Exception as e:
            print(f"  Error loading True.csv: {e}")
    
    # Load Fake.csv (Fake content - label as 1)
    if os.path.exists(fake_path):
        try:
            df_fake = pd.read_csv(fake_path, encoding='latin-1')
            print(f"  Fake.csv structure: {list(df_fake.columns)}")
            
            # Process Fake.csv - these are FAKE content, label as 1
            df_fake_processed = df_fake[["text"]].copy()
            df_fake_processed["title"] = ""
            df_fake_processed["text_full"] = df_fake_processed["text"]
            df_fake_processed["label"] = 1  # Fake content
            df_fake_processed["source"] = "fake_corrected"
            frames.append(df_fake_processed)
            print(f"  Loaded Fake.csv (Fake content): {len(df_fake_processed)} samples")
        except Exception as e:
            print(f"  Error loading Fake.csv: {e}")
    
    # Add Politifact.xlsx
    pf_path = os.path.join(datasets_dir, "Politifact.xlsx")
    if os.path.exists(pf_path):
        try:
            excel_file = pd.ExcelFile(pf_path)
            print(f"  Politifact sheets: {excel_file.sheet_names}")
            
            for sheet_name in excel_file.sheet_names:
                df_pf = pd.read_excel(pf_path, sheet_name=sheet_name)
                print(f"  Sheet {sheet_name} columns: {list(df_pf.columns)}")
                
                # Try to find text and rating columns
                text_col = None
                rating_col = None
                
                for col in ["text", "content", "statement", "claim", "article"]:
                    if col in df_pf.columns:
                        text_col = col
                        break
                
                for col in ["rating", "verdict", "truth", "label", "class"]:
                    if col in df_pf.columns:
                        rating_col = col
                        break
                
                if text_col and rating_col:
                    df_pf_processed = df_pf[[text_col, rating_col]].copy()
                    df_pf_processed["title"] = ""
                    df_pf_processed["text"] = df_pf_processed[text_col]
                    df_pf_processed["text_full"] = df_pf_processed[text_col]
                    
                    # Map PolitiFact ratings to binary (0=truthful, 1=false)
                    rating_map = {
                        "True": 0, "Mostly True": 0, "Half True": 0.5,
                        "Mostly False": 1, "False": 1, "Pants on Fire": 1
                    }
                    df_pf_processed["label"] = df_pf_processed[rating_col].map(rating_map)
                    df_pf_processed["source"] = f"politifact_{sheet_name}"
                    
                    # Remove rows with unmapped ratings
                    df_pf_processed = df_pf_processed.dropna(subset=["label"])
                    frames.append(df_pf_processed)
                    print(f"  Loaded {sheet_name}: {len(df_pf_processed)} samples")
                else:
                    print(f"  Sheet {sheet_name} columns not recognized")
        except Exception as e:
            print(f"  Error loading Politifact.xlsx: {e}")
    
    if not frames:
        raise FileNotFoundError("No dataset files found")
    
    combined_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combined dataset: {len(combined_df)} samples")
    print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
    print(f"Source distribution: {combined_df['source'].value_counts().to_dict()}")
    
    return combined_df

def extract_additional_patterns(df: pd.DataFrame) -> Dict:
    """Extract additional patterns from the combined dataset"""
    print("\nExtracting additional patterns...")
    
    fake_texts = df[df['label'] == 1]['text_full'].astype(str).tolist()
    real_texts = df[df['label'] == 0]['text_full'].astype(str).tolist()
    
    patterns = {}
    
    # Analyze fake news patterns
    fake_text = ' '.join(fake_texts).lower()
    real_text = ' '.join(real_texts).lower()
    
    # Common sensational phrases in fake news
    sensational_phrases = [
        'breaking', 'exclusive', 'shocking', 'unbelievable', 'you won\'t believe',
        'must see', 'viral', 'outrageous', 'scandal', 'exposed', 'leaked',
        'insider', 'sources say', 'according to sources', 'this changes everything',
        'game changer', 'mind blown', 'jaw dropping', 'stunning', 'incredible',
        'watch video', 'featured image', 'share this', 'go viral'
    ]
    
    fake_phrase_counts = {phrase: fake_text.count(phrase) for phrase in sensational_phrases}
    real_phrase_counts = {phrase: real_text.count(phrase) for phrase in sensational_phrases}
    
    # Find phrases that are significantly more common in fake news
    phrase_ratios = {}
    for phrase in sensational_phrases:
        fake_count = fake_phrase_counts[phrase]
        real_count = real_phrase_counts[phrase]
        total_fake = len(fake_texts)
        total_real = len(real_texts)
        
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
    )

def train_model(model_type: str):
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if model_type == "linear_svm":
        return LinearSVC(class_weight="balanced")
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced")
    raise ValueError(f"Unknown model_type: {model_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "config.yaml"))
    parser.add_argument("--datasets_dir", default="datasets")
    parser.add_argument("--out_model", default=os.path.join("models", "sensationalism_model.joblib"))
    parser.add_argument("--out_vectorizer", default=os.path.join("models", "tfidf_vectorizer.joblib"))
    parser.add_argument("--out_scaler", default=os.path.join("models", "scaler_comprehensive.joblib"))
    args = parser.parse_args()
    
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
    
    print("\nEvaluating...")
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(x_test)
    elif hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(x_test)[:, 1]
    else:
        y_scores = model.predict(x_test)
    
    try:
        auc = roc_auc_score(y_test, y_scores)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        print("ROC-AUC unavailable for this model.")
    
    print("\nClassification Report:")
    print(classification_report(y_test, (y_scores > 0 if y_scores.ndim == 1 else model.predict(x_test))))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_all, y, cv=int(train_cfg.get("cv_folds", 5)))
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
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