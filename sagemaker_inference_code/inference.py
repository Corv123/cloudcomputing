"""
SageMaker inference script for scikit-learn sensationalism model
This script is used by SageMaker to serve predictions
"""

import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def model_fn(model_dir):
    """
    Load the model artifacts from model_dir
    This function is called by SageMaker when the endpoint starts
    """
    print(f"Loading models from {model_dir}")
    
    # Load model, vectorizer, and scaler
    model = joblib.load(os.path.join(model_dir, 'sensationalism_model_comprehensive.joblib'))
    vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer_comprehensive.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler_comprehensive.joblib'))
    
    print("Models loaded successfully")
    return {
        'model': model,
        'vectorizer': vectorizer,
        'scaler': scaler
    }

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    Expected input: {"text": "article text here", "title": "optional title"}
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """
    Perform prediction on the deserialized input
    """
    model = model_dict['model']
    vectorizer = model_dict['vectorizer']
    scaler = model_dict['scaler']
    
    # Extract text and title
    text = input_data.get('text', '')
    title = input_data.get('title', '')
    
    # Combine title and text for analysis
    combined_text = f"{title} {text}".strip() if title else text
    
    # Transform text with TF-IDF
    texts = [combined_text]
    X_tfidf = vectorizer.transform(texts).toarray()
    
    # Extract enhanced features
    word_count = len(text.split())
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_count = sum(1 for c in text if c.isupper())
    
    # Create feature vector
    enhanced_features = np.array([[
        word_count / 1000.0,  # Normalized word count
        exclamation_count / max(len(text), 1),
        question_count / max(len(text), 1),
        caps_count / max(len(text), 1),
        0.5  # Placeholder for other features
    ]])
    
    # Combine features
    combined_features = np.concatenate([X_tfidf, enhanced_features], axis=1)
    scaled_features = scaler.transform(combined_features)
    
    # Predict
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(scaled_features)[0][1])
    else:
        score = float(model.predict(scaled_features)[0])
    
    return {
        'sensationalism_score': score,
        'text_length': word_count
    }

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")

