from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import os
from typing import List, Optional, Dict
from scipy import sparse
import numpy as np
import re
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize

import os as _os, sys as _sys
_SYS_SRC = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "src"))
_SYS_SRC = _os.path.normpath(_SYS_SRC)
if _SYS_SRC not in _sys.path:
    _sys.path.insert(0, _SYS_SRC)
from features_enhanced import build_linguistic_feature_matrix, extract_enhanced_linguistic_features, get_feature_names

# Load the comprehensive model (trained on 44,033 samples!)
MODEL_PATH = os.getenv("MODEL_PATH", "models/sensationalism_model_comprehensive.joblib")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer_comprehensive.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler_comprehensive.joblib")

app = FastAPI(title="Comprehensive Linguistic Cues Detector", version="4.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictRequest(BaseModel):
    text: str

class PredictBatchRequest(BaseModel):
    texts: List[str]

def _load_artifacts():
    """Load the comprehensive model artifacts"""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Loaded comprehensive model from {MODEL_PATH}")
        print(f"✓ Model trained on 44,033 samples with ROC-AUC: 0.9353")
        return model, vectorizer, scaler
    except Exception as e:
        print(f"✗ Error loading model artifacts: {e}")
        raise

model, vectorizer, scaler = _load_artifacts()

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Define word categories for frequency analysis
CREDIBLE_WORDS = [
    "research", "study", "evidence", "data", "analysis", "report", "findings", 
    "conclusion", "results", "statistics", "survey", "experiment", "investigation",
    "expert", "scientist", "researcher", "professor", "doctor", "authority",
    "official", "government", "institution", "university", "journal", "peer-reviewed"
]

SUSPICIOUS_WORDS = [
    "shocking", "unbelievable", "secret", "exposed", "revealed", "breaking",
    "urgent", "alert", "warning", "dangerous", "scandal", "conspiracy",
    "cover-up", "hidden", "forbidden", "insider", "leaked", "exclusive",
    "you won't believe", "doctors hate", "one weird trick", "miracle",
    "instant", "guaranteed", "proven", "amazing", "incredible", "stunning"
]

def analyze_word_frequency(text: str) -> Dict:
    """Analyze word frequency in the text and categorize as credible/suspicious"""
    # Clean and tokenize text
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    
    # Count credible and suspicious words
    credible_found = {}
    suspicious_found = {}
    
    for word, count in word_counts.items():
        if word in CREDIBLE_WORDS:
            credible_found[word] = count
        elif word in SUSPICIOUS_WORDS:
            suspicious_found[word] = count
    
    return {
        "credible_words": credible_found,
        "suspicious_words": suspicious_found,
        "total_words": len(words),
        "unique_words": len(word_counts)
    }

def analyze_sentiment_flow(text: str) -> Dict:
    """Analyze sentiment flow sentence by sentence"""
    try:
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Split into sentences
        sentences = sent_tokenize(text)
        sentiment_scores = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Get VADER sentiment score
                sentiment = sentiment_analyzer.polarity_scores(sentence)
                compound_score = sentiment['compound']
                sentiment_scores.append({
                    "sentence_number": i + 1,
                    "sentence": sentence.strip(),
                    "sentiment_score": compound_score,
                    "positive": sentiment['pos'],
                    "negative": sentiment['neg'],
                    "neutral": sentiment['neu']
                })
        
        return {
            "sentiment_flow": sentiment_scores,
            "total_sentences": len(sentiment_scores),
            "avg_sentiment": np.mean([s["sentiment_score"] for s in sentiment_scores]) if sentiment_scores else 0
        }
    except Exception as e:
        return {
            "sentiment_flow": [],
            "total_sentences": 0,
            "avg_sentiment": 0,
            "error": str(e)
        }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index_comprehensive.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(req: PredictRequest):
    """Predict sensationalism score with detailed feature breakdown"""
    try:
        # Process text
        texts = [req.text or ""]
        X_tfidf = vectorizer.transform(texts)
        X_ling = build_linguistic_feature_matrix(texts)
        X_ling_scaled = scaler.transform(X_ling)
        X = sparse.hstack([X_tfidf, sparse.csr_matrix(X_ling_scaled)], format="csr")
        
        # Get prediction score
        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(X)[:, 1][0])
        elif hasattr(model, "decision_function"):
            val = float(model.decision_function(X)[0])
            score = float(1 / (1 + np.exp(-val)))
        else:
            pred = int(model.predict(X)[0])
            score = float(pred)
        
        # Get detailed linguistic features
        detailed_features = extract_enhanced_linguistic_features(req.text)
        
        # Perform word frequency analysis
        word_frequency_analysis = analyze_word_frequency(req.text)
        
        # Perform sentiment flow analysis
        sentiment_flow_analysis = analyze_sentiment_flow(req.text)
        
        # Create comprehensive feature breakdown
        feature_breakdown = {
            "sensationalism_indicators": {
                "clickbait_score": detailed_features.clickbait_score,
                "emotional_intensity": detailed_features.emotional_intensity,
                "exclamation_density": detailed_features.exclamation_density,
                "caps_density": detailed_features.caps_density,
                "question_density": detailed_features.question_density,
                "clickbait_matches": detailed_features.clickbait_matches,
                "emotional_word_count": detailed_features.emotional_word_count,
                "all_caps_words": detailed_features.all_caps_words,
                "exclamation_count": detailed_features.exclamation_count,
                "question_count": detailed_features.question_count,
            },
            "sentiment_analysis": {
                "sentiment_polarity": detailed_features.sentiment_polarity,
                "sentiment_subjectivity": detailed_features.sentiment_subjectivity,
                "sentiment_intensity": detailed_features.sentiment_intensity,
                "sentiment_balance": detailed_features.sentiment_balance,
            },
            "language_patterns": {
                "intensifier_ratio": detailed_features.intensifier_ratio,
                "tentative_ratio": detailed_features.tentative_ratio,
                "evidence_ratio": detailed_features.evidence_ratio,
                "professional_ratio": detailed_features.professional_ratio,
                "balanced_ratio": detailed_features.balanced_ratio,
            },
            "text_structure": {
                "avg_sentence_length": detailed_features.avg_sentence_length,
                "avg_word_length": detailed_features.avg_word_length,
                "text_length": detailed_features.text_length,
                "word_count": detailed_features.word_count,
                "repetition_ratio": detailed_features.repetition_ratio,
                "unique_word_ratio": detailed_features.unique_word_ratio,
            },
            "content_quality": {
                "professional_word_count": detailed_features.professional_word_count,
                "balanced_word_count": detailed_features.balanced_word_count,
                "data_references": detailed_features.data_references,
            }
        }
        
        return {
            "sensationalism_bias_likelihood": score,
            "feature_breakdown": feature_breakdown,
            "word_frequency_analysis": word_frequency_analysis,
            "sentiment_flow_analysis": sentiment_flow_analysis,
            "interpretation": get_score_interpretation(score),
            "model_version": "Comprehensive v4.0.0",
            "model_stats": {
                "training_samples": 44033,
                "roc_auc": 0.9353,
                "accuracy": 0.83,
                "datasets_used": ["TrueFalse.csv", "Politifact.xlsx", "BuzzFeed", "PolitiFact"]
            }
        }
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "sensationalism_bias_likelihood": 0.5,
            "interpretation": {"level": "Error", "description": "Analysis failed", "color": "gray"}
        }

def get_score_interpretation(score: float) -> Dict[str, str]:
    """Get interpretation of the sensationalism score"""
    if score < 0.3:
        return {
            "level": "Low Sensationalism",
            "description": "The text appears to be neutral and factual with minimal sensationalism or bias. This is typical of well-written news articles and professional content.",
            "color": "green",
            "recommendation": "This content appears trustworthy and well-balanced.",
            "confidence": "High"
        }
    elif score < 0.7:
        return {
            "level": "Moderate Sensationalism", 
            "description": "The text shows some signs of sensationalism, emotional language, or potential bias. This could be legitimate news with some editorial flair, or content that needs closer scrutiny.",
            "color": "orange",
            "recommendation": "Review this content carefully and consider multiple sources.",
            "confidence": "Medium"
        }
    else:
        return {
            "level": "High Sensationalism",
            "description": "The text contains significant sensationalism, clickbait elements, or biased language. This content should be approached with caution and verified through reliable sources.",
            "color": "red",
            "recommendation": "Be skeptical of this content and seek verification from trusted sources.",
            "confidence": "High"
        }

@app.post("/predict_batch")
async def predict_batch(req: PredictBatchRequest):
    """Predict sensationalism scores for multiple texts"""
    try:
        texts = [t or "" for t in req.texts]
        X_tfidf = vectorizer.transform(texts)
        X_ling = build_linguistic_feature_matrix(texts)
        X_ling_scaled = scaler.transform(X_ling)
        X = sparse.hstack([X_tfidf, sparse.csr_matrix(X_ling_scaled)], format="csr")
        
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1].tolist()
        elif hasattr(model, "decision_function"):
            vals = model.decision_function(X)
            scores = (1 / (1 + np.exp(-vals))).tolist()
        else:
            preds = model.predict(X).tolist()
            scores = [float(p) for p in preds]
        
        return {
            "scores": [float(s) for s in scores],
            "interpretations": [get_score_interpretation(s) for s in scores],
            "model_version": "Comprehensive v4.0.0"
        }
        
    except Exception as e:
        return {
            "error": f"Batch prediction failed: {str(e)}",
            "scores": [0.5] * len(req.texts)
        }

@app.get("/features")
async def get_feature_info():
    """Get information about the linguistic features"""
    return {
        "feature_names": get_feature_names(),
        "total_features": len(get_feature_names()),
        "feature_categories": {
            "sensationalism_indicators": 10,
            "sentiment_analysis": 4,
            "language_patterns": 5,
            "text_structure": 6,
            "content_quality": 3
        },
        "model_version": "Comprehensive v4.0.0",
        "model_performance": {
            "roc_auc": 0.9353,
            "accuracy": 0.83,
            "training_samples": 44033,
            "cross_validation": 0.9364
        },
        "datasets_used": [
            "TrueFalse.csv (30,775 samples)",
            "Politifact.xlsx (12,836 samples)", 
            "BuzzFeed datasets (182 samples)",
            "PolitiFact datasets (240 samples)"
        ],
        "improvements": [
            "Trained on 44,033 samples (100x more data)",
            "ROC-AUC improved from 0.52 to 0.9353",
            "Enhanced feature engineering with 28 linguistic features",
            "Professional language detection",
            "Balanced reporting indicators",
            "Data reference counting",
            "Improved sentiment balance analysis"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": "Comprehensive v4.0.0",
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
        "scaler_path": SCALER_PATH,
        "performance": {
            "roc_auc": 0.9353,
            "accuracy": 0.83,
            "training_samples": 44033
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
