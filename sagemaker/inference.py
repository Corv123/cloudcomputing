#!/usr/bin/env python3
"""
SageMaker inference script for Fake News Detector
This script is deployed to SageMaker and handles model loading and predictions
Includes Flask server for /ping and /invocations endpoints
"""

import json
import os
import sys
import joblib
import numpy as np
import re
from scipy.sparse import csr_matrix
from flask import Flask, request, Response

# Add code directory to path
sys.path.insert(0, '/opt/ml/code')

# Import feature extraction
try:
    from features_enhanced import build_linguistic_feature_matrix
    FEATURES_AVAILABLE = True
except ImportError as e:
    # Fallback if features_enhanced not available
    FEATURES_AVAILABLE = False
    print(f"[WARN] features_enhanced not available: {e}, using simplified features")

# Initialize Flask app
app = Flask(__name__)

# Model files are loaded from /opt/ml/model/ in SageMaker
MODEL_PATH = '/opt/ml/model/sensationalism_model_comprehensive.joblib'
VECTORIZER_PATH = '/opt/ml/model/tfidf_vectorizer_comprehensive.joblib'
SCALER_PATH = '/opt/ml/model/scaler_comprehensive.joblib'

# Global variables for model components
model = None
vectorizer = None
scaler = None


def model_fn(model_dir):
    """
    Load the model components from disk.
    This function is called once when the SageMaker endpoint starts.
    
    Args:
        model_dir: Path to the directory containing model files (usually /opt/ml/model)
        
    Returns:
        Tuple of (model, vectorizer, scaler)
    """
    global model, vectorizer, scaler
    
    print(f"[INFO] Loading models from: {model_dir}")
    print(f"[INFO] Contents of model_dir: {os.listdir(model_dir) if os.path.exists(model_dir) else 'Directory not found'}")
    
    # Load model components
    model_path = os.path.join(model_dir, 'sensationalism_model_comprehensive.joblib')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer_comprehensive.joblib')
    scaler_path = os.path.join(model_dir, 'scaler_comprehensive.joblib')
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        print(f"[INFO] Loading model from {model_path}...")
        model = joblib.load(model_path)
        print("[INFO] Model loaded successfully")
        
        print(f"[INFO] Loading vectorizer from {vectorizer_path}...")
        vectorizer = joblib.load(vectorizer_path)
        print("[INFO] Vectorizer loaded successfully")
        
        print(f"[INFO] Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        print("[INFO] Scaler loaded successfully")
        
        return (model, vectorizer, scaler)
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body: The body of the request
        request_content_type: The content type of the request
        
    Returns:
        Dictionary with 'text' and optional 'title' keys
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_tuple):
    """
    Perform prediction on the deserialized input.
    
    Args:
        input_data: Dictionary with 'text' and optional 'title'
        model_tuple: Tuple of (model, vectorizer, scaler)
        
    Returns:
        Dictionary with prediction results
    """
    model, vectorizer, scaler = model_tuple
    
    # Extract text
    text = input_data.get('text', '')
    title = input_data.get('title', '')
    
    if not text:
        return {
            'error': 'No text provided',
            'sensationalism_score': 0.0
        }
    
    try:
        # Transform text with TF-IDF (same as training)
        texts = [text]
        X_tfidf = vectorizer.transform(texts)
        
        # Extract linguistic features (same as training - 28 features)
        if FEATURES_AVAILABLE:
            X_ling = build_linguistic_feature_matrix(texts)
        else:
            # Fallback simplified features (should match training if possible)
            print("[WARN] Using simplified features - may not match training exactly")
            word_count = len(text.split())
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_count = sum(1 for c in text if c.isupper())
            X_ling = np.array([[
                word_count / 1000.0,
                exclamation_count / max(len(text), 1),
                question_count / max(len(text), 1),
                caps_count / max(len(text), 1),
                0.5  # Placeholder
            ]])
            # Pad to 28 features (will cause issues if model expects 28)
            if X_ling.shape[1] < 28:
                padding = np.zeros((1, 28 - X_ling.shape[1]))
                X_ling = np.concatenate([X_ling, padding], axis=1)
        
        # Scale linguistic features (same as training)
        X_ling_scaled = scaler.transform(X_ling)
        
        # Combine features (same as training: TF-IDF + scaled linguistic features)
        from scipy import sparse
        combined_features = sparse.hstack([X_tfidf, sparse.csr_matrix(X_ling_scaled)], format="csr")
        
        # Predict (model expects sparse matrix, not scaled_features)
        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(combined_features)[0][1])
        else:
            score = float(model.predict(combined_features)[0])
        
        return {
            'sensationalism_score': round(score, 3),
            'success': True
        }
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return {
            'error': str(e),
            'sensationalism_score': 0.0,
            'success': False
        }


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    
    Args:
        prediction: Dictionary with prediction results
        response_content_type: The content type of the response
        
    Returns:
        JSON string
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")


# Flask routes for SageMaker
@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return Response(response='\n', status=200, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """Prediction endpoint"""
    try:
        # Get content type
        content_type = request.content_type or 'application/json'
        
        # Get request body - SageMaker sends it as bytes
        # Use get_data() which handles both bytes and text
        request_body = request.get_data(as_text=True)
        if not request_body:
            # Fallback: try to decode as UTF-8
            try:
                request_body = request.get_data().decode('utf-8')
            except:
                request_body = request.get_data().decode('utf-8', errors='ignore')
        
        # Parse input
        input_data = input_fn(request_body, content_type)
        
        # Get model tuple
        model_tuple = (model, vectorizer, scaler)
        
        # Make prediction
        prediction = predict_fn(input_data, model_tuple)
        
        # Format output
        output = output_fn(prediction, 'application/json')
        
        return Response(response=output, status=200, mimetype='application/json')
    except Exception as e:
        print(f"[ERROR] Invocation error: {e}")
        import traceback
        traceback.print_exc()
        error_response = {
            'error': str(e),
            'success': False
        }
        return Response(
            response=json.dumps(error_response),
            status=500,
            mimetype='application/json'
        )


if __name__ == '__main__':
    # Load model when container starts
    print("[INFO] Starting SageMaker inference server...")
    print("[INFO] Loading models...")
    model_dir = '/opt/ml/model'
    model_fn(model_dir)
    print("[INFO] Models loaded successfully")
    print("[INFO] Starting Flask server on port 8080...")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8080, threaded=True)

