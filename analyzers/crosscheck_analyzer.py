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
        Cross-check analysis with INVERTED scoring logic.

        SCORING PHILOSOPHY:
        - Low similarity (unique content) = HIGH score (good for credibility)
        - High similarity (duplicate content) = LOW score (concerning)

        Why? Because:
        - Unique articles show original reporting
        - Very high similarity suggests plagiarism/content farms
        - For credibility assessment, we want UNIQUE content

        Args:
            content: Article content to check
            database_articles: List of articles from database

        Returns:
            dict with 'score' (0-100 scale, inverted from similarity) and 'chart4_data'
        """
        if not database_articles or len(database_articles) == 0:
            # No articles to compare against - return neutral good score
            return {
                "score": 0.80,  # 80/100 - neutral good score on 0-1 scale
                "chart4_data": {
                    "message": "No comparison data available. This is the first article.",
                    "points": []
                },
                "metrics": {
                    "similar_count": 0,
                    "total_compared": 0,
                    "avg_similarity": 0,
                    "max_similarity": 0,
                    "similarity_percentage": 0,
                    "status": "Limited Comparison Data"
                }
            }

        # Calculate similarities with all database articles
        similarities = self._calculate_similarities(content, database_articles)

        if not similarities or len(similarities) == 0:
            # No valid comparisons - return neutral good score
            return {
                "score": 0.75,  # 75/100 on 0-1 scale
                "chart4_data": {
                    "message": "Unable to compare articles.",
                    "points": []
                },
                "metrics": {
                    "similar_count": 0,
                    "total_compared": 0,
                    "avg_similarity": 0,
                    "max_similarity": 0,
                    "similarity_percentage": 0,
                    "status": "Comparison Failed"
                }
            }

        # Calculate statistics
        similarity_values = [sim for _, sim in similarities]
        avg_similarity = float(np.mean(similarity_values))
        max_similarity = float(np.max(similarity_values))

        # Convert to percentages for display
        avg_similarity_pct = avg_similarity * 100
        max_similarity_pct = max_similarity * 100

        # Count how many are "similar" (>60% similarity)
        similar_count = sum(1 for sim in similarity_values if sim > 0.6)

        # === INVERTED SCORING LOGIC ===
        # Lower similarity = higher score (unique content is good!)

        if len(database_articles) < 5:
            # Special case: Very small database
            score = 0.80  # 80/100 - neutral good score for small databases
            status = "Limited Comparison Data"
            description = f"Only {len(database_articles)} article(s) in database for comparison. Cross-check reliability increases with more analyzed articles."

        elif max_similarity_pct > 90:
            # Nearly identical content - very suspicious
            score = 0.20  # 20/100
            status = "Duplicate Content Detected"
            description = f"Extremely high similarity ({max_similarity_pct:.0f}%) with existing articles. Possible duplicate or plagiarized content."

        elif max_similarity_pct > 75:
            # Very similar - concerning
            score = 0.40  # 40/100
            status = "Very Similar Content"
            description = f"Very high similarity ({max_similarity_pct:.0f}%) with {similar_count} database article(s). Content may not be original."

        elif max_similarity_pct > 60:
            # Moderate similarity - expected for related news
            score = 0.70  # 70/100
            status = "Related Coverage"
            description = f"Moderate similarity ({max_similarity_pct:.0f}%) with {similar_count} article(s). Normal overlap for related news stories."

        elif max_similarity_pct > 40:
            # Low similarity - good (unique angle)
            score = 0.85  # 85/100
            status = "Unique Perspective"
            description = f"Low similarity ({max_similarity_pct:.0f}%). Content offers unique perspective or angle on the topic."

        else:
            # Very low similarity - excellent (original reporting)
            score = 0.95  # 95/100
            status = "Original Content"
            description = f"Minimal similarity ({max_similarity_pct:.0f}%). Content appears to be original reporting or unique topic."

        # Chart 4 Data: Scatter plot of similar articles
        chart4_data = self._prepare_chart_data(similarities)
        chart4_data["description"] = description

        return {
            "score": round(score, 3),  # 0-1 scale (will be multiplied by weight)
            "chart4_data": chart4_data,
            "metrics": {
                "similar_count": similar_count,
                "total_compared": len(similarities),
                "avg_similarity": round(avg_similarity, 3),
                "max_similarity": round(max_similarity, 3),
                "avg_similarity_pct": round(avg_similarity_pct, 1),
                "max_similarity_pct": round(max_similarity_pct, 1),
                "similarity_percentage": round(max_similarity_pct, 1),  # For backward compatibility
                "status": status,
                "description": description
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
        """Prepare data for Chart 4 (scatter plot showing similarity analysis)"""
        if not similarities:
            return {
                "message": "No similarity data available",
                "points": []
            }

        # Take top 20 most similar articles
        top_similar = similarities[:20]

        data_points = []
        for article, similarity in top_similar:
            # Convert similarity to percentage for display
            similarity_pct = round(similarity * 100, 1)
            credibility = round(article.get('credibility_score', 0.5) * 100, 1)

            data_points.append({
                "x": round(similarity, 3),  # Keep 0-1 scale for plotting
                "y": round(article.get('credibility_score', 0.5), 3),  # Keep 0-1 scale
                "similarity_pct": similarity_pct,  # For display
                "credibility_pct": credibility,  # For display
                "title": article.get('title', 'Unknown')[:60],
                "source": article.get('source', 'Unknown'),
                "url": article.get('url', '')
            })

        return {
            "points": data_points,
            "x_label": "Content Similarity",
            "y_label": "Article Credibility",
            "total_points": len(data_points)
        }
