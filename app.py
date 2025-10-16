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

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# Initialize components
db = ArticleDatabase()
extractor = ArticleExtractor()
language_analyzer = LanguageAnalyzer()
credibility_analyzer = CredibilityAnalyzer()
crosscheck_analyzer = CrossCheckAnalyzer()
related_articles_analyzer = RelatedArticlesAnalyzer()

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
            return jsonify({
                'success': True,
                'cached': True,
                'article': existing
            })
        
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

        # DEBUG: Chart 3 Data Flow
        print("\n" + "=" * 60)
        print("🔍 CREDIBILITY ANALYZER OUTPUT:")
        print(f"Score: {credibility_result.get('score')}")
        print(f"Chart3 Data: {credibility_result.get('chart3_data')}")
        print(f"Detailed Metrics: {credibility_result.get('detailed_metrics')}")
        print("=" * 60 + "\n")

        # Get database articles for cross-checking
        database_articles = db.get_all_articles()
        crosscheck_result = crosscheck_analyzer.analyze(
            article_data['content'],
            database_articles
        )

        # Find related articles from reputable sources (Chart 6)
        related_articles_result = related_articles_analyzer.analyze(
            article_data['title'],
            article_data['content'],
            article_data['domain']
        )

        # Calculate overall score using configured weights
        weights = config.SCORE_WEIGHTS
        overall_score = (
            language_result['score'] * weights['language'] +
            credibility_result['score'] * weights['credibility'] +
            crosscheck_result['score'] * weights['crosscheck']
        )
        
        # Combine all results
        article_data.update({
            'language_score': language_result['score'],
            'credibility_score': credibility_result['score'],
            'cross_check_score': crosscheck_result['score'],
            'overall_score': round(overall_score, 3),
            'chart1_data': language_result['chart1_data'],
            'chart2_data': language_result['chart2_data'],
            'chart3_data': credibility_result['chart3_data'],
            'detailed_metrics': credibility_result.get('detailed_metrics', {}),  # NEW: Chart 3 breakdown
            'chart4_data': crosscheck_result['chart4_data'],
            'chart6_data': related_articles_result.get('chart6_data', {'articles': [], 'message': 'No related articles'}),
            'word_count': len(article_data['content'].split()),
            'sensational_keyword_count': 0,
            'tfidf_vector': '{}',
            'known_source_classification': credibility_result.get('known_source')
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

if __name__ == '__main__':
    print("=" * 60)
    print("🔍 Fake News Detector - Starting Server")
    print("=" * 60)
    print(f"Database: {config.DATABASE_NAME}")
    print(f"Score Weights: {config.SCORE_WEIGHTS}")
    print("=" * 60)
    app.run(debug=True, port=5000)