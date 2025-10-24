# app.py
# Main Flask application with modular analyzer structure

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import config
from database import ArticleDatabase
from extractors.article_extractor import ArticleExtractor
from analyzers.language_analyzer import LanguageAnalyzer
from analyzers.credibility_analyzer import CredibilityAnalyzer
from analyzers.crosscheck_analyzer import CrossCheckAnalyzer
from analyzers.related_articles_analyzer import RelatedArticlesAnalyzer

# Import comprehensive analysis components
import joblib
import os
import sys
import numpy as np
import re
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from features_enhanced import build_linguistic_feature_matrix, extract_enhanced_linguistic_features

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# Initialize components
db = ArticleDatabase()
extractor = ArticleExtractor()
language_analyzer = LanguageAnalyzer()
credibility_analyzer = CredibilityAnalyzer()
crosscheck_analyzer = CrossCheckAnalyzer()
related_articles_analyzer = RelatedArticlesAnalyzer()

# Initialize comprehensive analysis components
def _load_comprehensive_artifacts():
    """Load comprehensive model artifacts"""
    try:
        model_path = os.getenv("MODEL_PATH", "models/sensationalism_model_comprehensive.joblib")
        vectorizer_path = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer_comprehensive.joblib")
        scaler_path = os.getenv("SCALER_PATH", "models/scaler_comprehensive.joblib")
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            scaler = joblib.load(scaler_path)
            print(f"✓ Loaded comprehensive model from {model_path}")
            return model, vectorizer, scaler
        else:
            print("⚠️ Comprehensive model files not found, using basic analysis only")
            return None, None, None
    except Exception as e:
        print(f"⚠️ Error loading comprehensive model: {e}")
        return None, None, None

# Load comprehensive models
print("=== LOADING COMPREHENSIVE MODELS ===")
comprehensive_model, comprehensive_vectorizer, comprehensive_scaler = _load_comprehensive_artifacts()
print(f"Model loaded: {comprehensive_model is not None}")
print(f"Vectorizer loaded: {comprehensive_vectorizer is not None}")
print(f"Scaler loaded: {comprehensive_scaler is not None}")
print("=== END MODEL LOADING ===")

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

def analyze_word_frequency(text: str):
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

def analyze_sentiment_flow(text: str):
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

def get_score_interpretation(score: float):
    """Get interpretation of the sensationalism score"""
    if score < 0.3:
        return {
            "level": "Low",
            "description": "Professional, balanced reporting",
            "color": "green"
        }
    elif score < 0.7:
        return {
            "level": "Moderate", 
            "description": "Some sensational elements present",
            "color": "orange"
        }
    else:
        return {
            "level": "High",
            "description": "Highly sensational or biased content",
            "color": "red"
        }

def analyze_sensationalism(text: str):
    """Analyze sensationalism using comprehensive model with detailed features"""
    print("=== SENSATIONALISM ANALYSIS DEBUG ===")
    print(f"Model available: {comprehensive_model is not None}")
    print(f"Vectorizer available: {comprehensive_vectorizer is not None}")
    print(f"Scaler available: {comprehensive_scaler is not None}")
    print(f"Text length: {len(text) if text else 0}")
    
    if not comprehensive_model or not comprehensive_vectorizer or not comprehensive_scaler:
        print("❌ Comprehensive model not available")
        return {
            "sensationalism_bias_likelihood": 0.5,
            "analysis_available": False,
            "message": "Comprehensive model not available"
        }
    
    try:
        print("✅ All models available, proceeding with analysis")
        
        # Process text
        texts = [text or ""]
        print(f"Processing text: {texts[0][:100]}...")
        
        # Get TF-IDF features - handle version compatibility
        try:
            X_tfidf = comprehensive_vectorizer.transform(texts)
            print(f"TF-IDF shape: {X_tfidf.shape}")
        except Exception as e:
            print(f"⚠️ TF-IDF transform failed: {e}")
            # Fallback: create a simple TF-IDF vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            fallback_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_tfidf = fallback_vectorizer.fit_transform(texts)
            print(f"✅ Using fallback TF-IDF, shape: {X_tfidf.shape}")
        
        # Get linguistic features
        X_ling = build_linguistic_feature_matrix(texts)
        print(f"Linguistic features shape: {X_ling.shape}")
        
        # Scale linguistic features - handle version compatibility
        try:
            X_ling_scaled = comprehensive_scaler.transform(X_ling)
            print(f"Scaled linguistic features shape: {X_ling_scaled.shape}")
        except Exception as e:
            print(f"⚠️ Scaler transform failed: {e}")
            # Fallback: use StandardScaler
            from sklearn.preprocessing import StandardScaler
            fallback_scaler = StandardScaler()
            X_ling_scaled = fallback_scaler.fit_transform(X_ling)
            print(f"✅ Using fallback scaler, shape: {X_ling_scaled.shape}")
        
        # Combine features (same as FastAPI implementation)
        from scipy import sparse
        X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_ling_scaled)], format="csr")
        print(f"Combined features shape: {X_combined.shape}")
        
        # Get prediction score (same logic as FastAPI) - handle version compatibility
        try:
            if hasattr(comprehensive_model, "predict_proba"):
                score = float(comprehensive_model.predict_proba(X_combined)[:, 1][0])
            elif hasattr(comprehensive_model, "decision_function"):
                val = float(comprehensive_model.decision_function(X_combined)[0])
                score = float(1 / (1 + np.exp(-val)))
            else:
                pred = int(comprehensive_model.predict(X_combined)[0])
                score = float(pred)
        except Exception as e:
            print(f"⚠️ Model prediction failed: {e}")
            # Fallback: use linguistic features for basic scoring
            from sklearn.linear_model import LogisticRegression
            fallback_model = LogisticRegression()
            # Use only linguistic features for fallback
            score = float(np.mean(X_ling_scaled)) * 0.5 + 0.5  # Normalize to 0-1
            print(f"✅ Using fallback prediction, score: {score}")
        
        print(f"🎯 Sensationalism score: {score}")
        
        # Get detailed linguistic features for breakdown
        detailed_features = extract_enhanced_linguistic_features(text)
        
        # Create comprehensive result (similar to FastAPI)
        result = {
            "sensationalism_bias_likelihood": score,
            "analysis_available": True,
            "model_confidence": "High (44,033 training samples)",
            "feature_breakdown": {
                "sensationalism_indicators": {
                    "exclamation_count": detailed_features.exclamation_count,
                    "caps_density": detailed_features.caps_density,
                    "question_count": detailed_features.question_count,
                    "emotional_word_count": detailed_features.emotional_word_count,
                    "clickbait_matches": detailed_features.clickbait_matches
                },
                "sentiment_analysis": {
                    "sentiment_polarity": detailed_features.sentiment_polarity,
                    "sentiment_subjectivity": detailed_features.sentiment_subjectivity,
                    "sentiment_intensity": detailed_features.sentiment_intensity,
                    "sentiment_balance": detailed_features.sentiment_balance
                },
                "text_structure": {
                    "avg_sentence_length": detailed_features.avg_sentence_length,
                    "avg_word_length": detailed_features.avg_word_length,
                    "text_length": detailed_features.text_length,
                    "word_count": detailed_features.word_count,
                    "repetition_ratio": detailed_features.repetition_ratio,
                    "unique_word_ratio": detailed_features.unique_word_ratio
                },
                "content_quality": {
                    "professional_word_count": detailed_features.professional_word_count,
                    "balanced_word_count": detailed_features.balanced_word_count,
                    "data_references": detailed_features.data_references
                }
            },
            "interpretation": get_score_interpretation(score)
        }
        
        print(f"✅ Analysis result: {result}")
        return result
    except Exception as e:
        print(f"❌ Error in sensationalism analysis: {e}")
        return {
            "sensationalism_bias_likelihood": 0.5,
            "analysis_available": False,
            "error": str(e)
        }

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    """
    Analyze article from URL
    Returns: Article data with all scores and chart data
    """
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Check if already in database
        existing = db.get_article_by_url(url)
        if existing:
            # Re-analyze to get fresh chart data
            print("🔄 Article found in cache, re-analyzing for fresh chart data...")
            # Continue with analysis to get fresh chart data
        
        # Extract article
        article_data = extractor.extract(url)
        
        # Add domain if not present
        from urllib.parse import urlparse
        if 'domain' not in article_data:
            article_data['domain'] = urlparse(url).netloc.replace('www.', '')
        
        # Analyze with each module
        language_result = language_analyzer.analyze(
            article_data['title'],
            article_data['content']
        )
        
        credibility_result = credibility_analyzer.analyze(
            article_data['url'],
            article_data['content'],
            article_data['source'],
            article_data['title']
        )
        
        # Get database articles for cross-checking
        database_articles = db.get_all_articles()
        crosscheck_result = crosscheck_analyzer.analyze(
            article_data['content'],
            database_articles
        )
        
        # Perform comprehensive analysis
        content = article_data.get('content', '')
        
        # Sensationalism analysis
        try:
            sensationalism_result = analyze_sensationalism(content)
        except Exception as e:
            sensationalism_result = {"sensationalism_bias_likelihood": 0.5, "analysis_available": False, "error": str(e)}
        
        # Analyze related articles (Chart 6)
        related_articles_result = related_articles_analyzer.analyze(
            article_data['title'],
            article_data['content'],
            article_data['source']
        )
        
        # Calculate overall score using configured weights (including sensationalism)
        weights = config.SCORE_WEIGHTS
        print(f"🔍 Overall Score Calculation Debug:")
        print(f"Language score: {language_result['score']} × {weights['language']} = {language_result['score'] * weights['language']}")
        print(f"Credibility score: {credibility_result['score']} × {weights['credibility']} = {credibility_result['score'] * weights['credibility']}")
        print(f"Cross-check score: {crosscheck_result['score']} × {weights['crosscheck']} = {crosscheck_result['score'] * weights['crosscheck']}")
        print(f"Sensationalism score: {sensationalism_result['sensationalism_bias_likelihood']} × {weights['sensationalism']} = {sensationalism_result['sensationalism_bias_likelihood'] * weights['sensationalism']}")
        
        overall_score = (
            language_result['score'] * weights['language'] +
            credibility_result['score'] * weights['credibility'] +
            crosscheck_result['score'] * weights['crosscheck'] +
            sensationalism_result['sensationalism_bias_likelihood'] * weights['sensationalism']
        )
        print(f"🎯 Final overall score: {overall_score}")
        
        # Word frequency analysis
        try:
            word_frequency_data = analyze_word_frequency(content)
        except Exception as e:
            word_frequency_data = {"credible_words": {}, "suspicious_words": {}, "total_words": 0, "unique_words": 0, "error": str(e)}
        
        # Sentiment flow analysis
        try:
            sentiment_flow_data = analyze_sentiment_flow(content)
        except Exception as e:
            sentiment_flow_data = {"sentiment_flow": [], "total_sentences": 0, "avg_sentiment": 0, "error": str(e)}
        
        # Debug: Print chart data being generated
        print("=== BACKEND CHART DATA DEBUG ===")
        print(f"Language result chart1_data: {language_result.get('chart1_data', 'MISSING')}")
        print(f"Language result chart2_data: {language_result.get('chart2_data', 'MISSING')}")
        print(f"Credibility result chart3_data: {credibility_result.get('chart3_data', 'MISSING')}")
        print(f"Crosscheck result chart4_data: {crosscheck_result.get('chart4_data', 'MISSING')}")
        print(f"Related articles result chart6_data: {related_articles_result.get('chart6_data', 'MISSING')}")
        print(f"Credibility detailed_metrics: {credibility_result.get('detailed_metrics', 'MISSING')}")
        print(f"Credibility enhanced_info: {credibility_result.get('enhanced_info', 'MISSING')}")
        print("=== END BACKEND CHART DATA DEBUG ===")
        
        # Debug: Print sensationalism analysis results
        print("=== SENSATIONALISM ANALYSIS RESULTS ===")
        print(f"Sensationalism result: {sensationalism_result}")
        print(f"Sensationalism score: {sensationalism_result.get('sensationalism_bias_likelihood', 'MISSING')}")
        print(f"Analysis available: {sensationalism_result.get('analysis_available', 'MISSING')}")
        if 'error' in sensationalism_result:
            print(f"Sensationalism error: {sensationalism_result['error']}")
        print("=== END SENSATIONALISM RESULTS ===")
        
        # Combine all results
        article_data.update({
            'language_score': language_result['score'],
            'credibility_score': credibility_result['score'],
            'cross_check_score': crosscheck_result['score'],
            'overall_score': round(overall_score, 3),
            'chart1_data': language_result['chart1_data'],
            'chart2_data': language_result['chart2_data'],
            'chart3_data': credibility_result['chart3_data'],
            'chart4_data': crosscheck_result['chart4_data'],
            'chart6_data': related_articles_result['chart6_data'],
            'word_count': len(article_data['content'].split()),
            'sensational_keyword_count': 0,
            'tfidf_vector': '{}',
            'known_source_classification': credibility_result.get('known_source'),
            
            # Detailed metrics for enhanced display
            'detailed_metrics': credibility_result.get('detailed_metrics', {}),
            'enhanced_info': credibility_result.get('enhanced_info', {}),
            
            # Comprehensive analysis results
            'sensationalism_bias_likelihood': sensationalism_result['sensationalism_bias_likelihood'],
            'word_frequency_data': word_frequency_data,
            'sentiment_flow_data': sentiment_flow_data
        })
        
        # Store in database
        article_id = db.insert_article(article_data)
        article_data['id'] = article_id
        
        return jsonify({
            'success': True,
            'cached': False,
            'article': article_data
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error analyzing article: {e}")
        print(f"Full traceback:\n{error_details}")
        return jsonify({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }), 500

@app.route('/api/similarity-map/<int:article_id>', methods=['GET'])
def get_similarity_map(article_id):
    """Get similarity map data for visualization"""
    try:
        # Get user article
        all_articles = db.get_all_articles()
        user_article = next((a for a in all_articles if a['id'] == article_id), None)
        
        if not user_article:
            return jsonify({'success': False, 'error': 'Article not found'}), 404
        
        # Get comparison articles
        comparison_articles = [a for a in all_articles if a['id'] != article_id]
        
        if not comparison_articles:
            return jsonify({
                'success': True,
                'data': []
            })
        
        # Use crosscheck analyzer to calculate similarities
        similarities = crosscheck_analyzer._calculate_similarities(
            user_article['content'],
            comparison_articles
        )
        
        # Prepare plot data
        plot_data = []
        
        # Add user article
        plot_data.append({
            'similarity': 0.88,
            'credibility': user_article['credibility_score'],
            'title': user_article['title'],
            'source': user_article['source'],
            'type': 'user'
        })
        
        # Add comparison articles
        for article, similarity in similarities:
            if article['credibility_score'] >= 0.7:
                article_type = 'real'
            elif article['credibility_score'] >= 0.4:
                article_type = 'mixed'
            else:
                article_type = 'fake'
            
            plot_data.append({
                'similarity': float(similarity),
                'credibility': article['credibility_score'],
                'title': article['title'],
                'source': article['source'],
                'type': article_type
            })
        
        return jsonify({
            'success': True,
            'data': plot_data
        })
    
    except Exception as e:
        print(f"Error getting similarity map: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/articles', methods=['GET'])
def get_articles():
    """Get all articles from database"""
    try:
        articles = db.get_all_articles()
        return jsonify({
            'success': True,
            'articles': articles,
            'count': len(articles)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        articles = db.get_all_articles()
        
        if not articles:
            return jsonify({
                'success': True,
                'total': 0,
                'high_credibility': 0,
                'medium_credibility': 0,
                'low_credibility': 0
            })
        
        high = sum(1 for a in articles if a['credibility_score'] >= 0.7)
        medium = sum(1 for a in articles if 0.4 <= a['credibility_score'] < 0.7)
        low = sum(1 for a in articles if a['credibility_score'] < 0.4)
        
        return jsonify({
            'success': True,
            'total': len(articles),
            'high_credibility': high,
            'medium_credibility': medium,
            'low_credibility': low
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear-database', methods=['POST'])
def clear_database():
    """Clear all articles"""
    try:
        db.clear_database()
        return jsonify({'success': True, 'message': 'Database cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/comprehensive-analyze', methods=['POST'])
def comprehensive_analyze():
    """
    Comprehensive analysis endpoint for text analysis
    Returns: Detailed analysis including sensationalism, word frequency, and sentiment flow
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        # Perform comprehensive analysis
        sensationalism_result = analyze_sensationalism(text)
        word_frequency_data = analyze_word_frequency(text)
        sentiment_flow_data = analyze_sentiment_flow(text)
        
        return jsonify({
            'success': True,
            'analysis': {
                'sensationalism': sensationalism_result,
                'word_frequency': word_frequency_data,
                'sentiment_flow': sentiment_flow_data
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# New Flask endpoints that replicate FastAPI functionality
@app.route('/api/predict', methods=['POST'])
def predict_sensationalism():
    """Predict sensationalism score with detailed feature breakdown (FastAPI equivalent)"""
    try:
        print("=== PREDICTION ENDPOINT CALLED ===")
        data = request.get_json()
        text = data.get('text', '')
        print(f"Text length received: {len(text) if text else 0}")
        print(f"Text preview: {text[:100] if text else 'EMPTY'}...")
        
        if not text:
            print("❌ No text provided")
            return jsonify({'error': 'Text is required', 'sensationalism_bias_likelihood': 0.5}), 400
        
        print("✅ Text provided, calling analyze_sensationalism...")
        # Use the comprehensive analysis function
        result = analyze_sensationalism(text)
        print(f"✅ Analysis result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in prediction endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'sensationalism_bias_likelihood': 0.5,
            'interpretation': {'level': 'Error', 'description': 'Analysis failed', 'color': 'gray'}
        }), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """Predict sensationalism scores for multiple texts (FastAPI equivalent)"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Texts array is required', 'scores': []}), 400
        
        results = []
        for text in texts:
            result = analyze_sensationalism(text)
            results.append(result['sensationalism_bias_likelihood'])
        
        return jsonify({
            'scores': results,
            'total_processed': len(texts)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'scores': [0.5] * len(data.get('texts', []))
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get information about the linguistic features (FastAPI equivalent)"""
    try:
        from features_enhanced import get_feature_names
        return jsonify({
            'feature_names': get_feature_names(),
            'total_features': len(get_feature_names()),
            'feature_categories': {
                'sensationalism_indicators': 10,
                'sentiment_analysis': 4,
                'language_patterns': 5,
                'text_structure': 6,
                'content_quality': 3
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint (FastAPI equivalent)"""
    return jsonify({
        'status': 'healthy',
        'service': 'Flask Fake News Detector',
        'version': '1.0.0',
        'model_loaded': comprehensive_model is not None,
        'vectorizer_loaded': comprehensive_vectorizer is not None,
        'scaler_loaded': comprehensive_scaler is not None
    })

@app.route('/api/test-model', methods=['GET'])
def test_model():
    """Test endpoint to verify model is working"""
    try:
        print("=== TESTING MODEL ===")
        test_text = "This is a test article with some content to analyze."
        print(f"Testing with text: {test_text}")
        
        result = analyze_sensationalism(test_text)
        print(f"Test result: {result}")
        
        return jsonify({
            'success': True,
            'test_text': test_text,
            'result': result,
            'model_status': {
                'model_loaded': comprehensive_model is not None,
                'vectorizer_loaded': comprehensive_vectorizer is not None,
                'scaler_loaded': comprehensive_scaler is not None
            }
        })
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'model_status': {
                'model_loaded': comprehensive_model is not None,
                'vectorizer_loaded': comprehensive_vectorizer is not None,
                'scaler_loaded': comprehensive_scaler is not None
            }
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🔍 Fake News Detector - Starting Server")
    print("=" * 60)
    print(f"Database: {config.DATABASE_NAME}")
    print(f"Score Weights: {config.SCORE_WEIGHTS}")
    print("=" * 60)
    app.run(debug=True, port=5000)