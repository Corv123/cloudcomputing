# analyzers/crosscheck_analyzer.py
# Cross-checks article against database and generates data for Chart 4

from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CrossCheckAnalyzer:
    """
    Cross-checks article against existing articles in database:
    - Similarity to verified articles
    - Uniqueness score
    - Pattern matching
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def analyze(self, content: str, database_articles: List[Dict]) -> Dict[str, Any]:
        """
        Analyze article against database
        
        Args:
            content: Article content to check
            database_articles: List of articles from database
            
        Returns:
            dict with 'score' and 'chart4_data'
        """
        if not database_articles or len(database_articles) < 2:
            # Not enough data for cross-checking
            return {
                "score": 0.5,
                "chart4_data": {
                    "message": "Insufficient data for cross-checking. Add more articles to database."
                },
                "metrics": {
                    "similar_credible": 0,
                    "similar_suspicious": 0,
                    "uniqueness": 1.0
                }
            }
        
        # Calculate similarities
        similarities = self._calculate_similarities(content, database_articles)
        
        # Analyze similarity patterns
        similar_credible = sum(
            1 for article, sim in similarities 
            if sim > 0.3 and article.get('credibility_score', 0) >= 0.7
        )
        
        similar_suspicious = sum(
            1 for article, sim in similarities 
            if sim > 0.3 and article.get('credibility_score', 0) < 0.4
        )
        
        # Calculate uniqueness (inverse of max similarity)
        max_similarity = max([sim for _, sim in similarities], default=0)
        uniqueness = 1 - max_similarity
        
        # Overall cross-check score
        if similar_credible > similar_suspicious:
            crosscheck_score = 0.7 + (similar_credible / len(similarities) * 0.3)
        elif similar_suspicious > similar_credible:
            crosscheck_score = 0.3 - (similar_suspicious / len(similarities) * 0.2)
        else:
            crosscheck_score = 0.5
        
        # Chart 4 Data: Scatter plot of similar articles
        chart4_data = self._prepare_chart_data(similarities)
        
        return {
            "score": round(max(0, min(1, crosscheck_score)), 3),
            "chart4_data": chart4_data,
            "metrics": {
                "similar_credible": similar_credible,
                "similar_suspicious": similar_suspicious,
                "uniqueness": round(uniqueness, 3),
                "total_compared": len(similarities)
            }
        }
    
    def _calculate_similarities(self, content: str, database_articles: List[Dict]) -> List[tuple]:
        """Calculate similarity scores between content and database articles"""
        try:
            # Prepare corpus
            corpus = [content] + [art['content'] for art in database_articles]
            
            # Calculate TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate similarities
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # Pair with articles
            results = [
                (article, float(sim)) 
                for article, sim in zip(database_articles, similarities)
            ]
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
        
        except Exception as e:
            print(f"Error calculating similarities: {e}")
            return []
    
    def _prepare_chart_data(self, similarities: List[tuple]) -> Dict[str, Any]:
        """Prepare data for Chart 4 (scatter plot)"""
        if not similarities:
            return {"message": "No similarity data available"}
        
        # Take top 20 most similar articles
        top_similar = similarities[:20]
        
        data_points = []
        for article, similarity in top_similar:
            data_points.append({
                "x": round(similarity, 3),
                "y": round(article.get('credibility_score', 0.5), 3),
                "title": article.get('title', 'Unknown')[:50],
                "source": article.get('source', 'Unknown')
            })
        
        return {
            "points": data_points,
            "x_label": "Similarity Score",
            "y_label": "Credibility Score"
        }
