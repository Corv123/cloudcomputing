# analyzers/language_analyzer.py
# Analyzes language quality and generates data for Chart 1 & 2

import re
from typing import Dict, Any

# Try to import VADER for proper sentiment analysis
VADER_AVAILABLE = False
_vader_analyzer = None
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_analyzer = SentimentIntensityAnalyzer()
    # Test that it actually works
    test_scores = _vader_analyzer.polarity_scores("This is a test.")
    if test_scores and 'pos' in test_scores:
        VADER_AVAILABLE = True
        print("[OK] VADER sentiment analyzer initialized and tested successfully")
    else:
        print("[WARN] VADER initialized but test failed")
except ImportError as e:
    print(f"[WARN] VADER import failed: {e}")
except Exception as e:
    print(f"[WARN] VADER initialization failed: {e}")
    import traceback
    traceback.print_exc()

class LanguageAnalyzer:
    """
    Analyzes article language quality based on:
    - Grammar and writing style
    - Capitalization patterns
    - Punctuation usage
    - Word complexity
    """
    
    def analyze(self, title: str, content: str) -> Dict[str, Any]:
        """
        Analyze language quality and return score + chart data
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            dict with 'score', 'chart1_data', 'chart2_data'
        """
        text = f"{title} {content}"
        
        # Calculate individual metrics
        capitalization_score = self._check_capitalization(text)
        punctuation_score = self._check_punctuation(text)
        complexity_score = self._check_complexity(text)
        grammar_score = self._check_grammar(text)
        
        # Overall language score (0-1)
        language_score = (
            capitalization_score * 0.25 +
            punctuation_score * 0.25 +
            complexity_score * 0.25 +
            grammar_score * 0.25
        )
        
        # Chart 1 Data: Bar chart of individual metrics
        chart1_data = {
            "labels": ["Capitalization", "Punctuation", "Complexity", "Grammar"],
            "values": [
                round(capitalization_score * 100, 1),
                round(punctuation_score * 100, 1),
                round(complexity_score * 100, 1),
                round(grammar_score * 100, 1)
            ]
        }
        
        # Chart 2 Data: Sentiment/Tone distribution (simplified)
        print(f"[DEBUG] LanguageAnalyzer: Analyzing tone for text length: {len(text)}")
        chart2_data = self._analyze_tone(text)
        print(f"[DEBUG] LanguageAnalyzer: chart2_data result: {chart2_data}")
        
        return {
            "score": round(language_score, 3),
            "chart1_data": chart1_data,
            "chart2_data": chart2_data,
            "metrics": {
                "capitalization": round(capitalization_score, 3),
                "punctuation": round(punctuation_score, 3),
                "complexity": round(complexity_score, 3),
                "grammar": round(grammar_score, 3)
            }
        }
    
    def _check_capitalization(self, text: str) -> float:
        """Check proper capitalization (0-1)"""
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 0.5
        
        properly_capitalized = sum(
            1 for s in sentences 
            if s.strip() and s.strip()[0].isupper()
        )
        
        # Check for excessive ALL CAPS
        words = text.split()
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 2)
        caps_penalty = min(0.3, all_caps / len(words) * 2) if words else 0
        
        base_score = properly_capitalized / len(sentences) if sentences else 0.5
        return max(0, min(1, base_score - caps_penalty))
    
    def _check_punctuation(self, text: str) -> float:
        """Check punctuation usage (0-1)"""
        # Excessive exclamation marks indicate sensationalism
        exclamation_ratio = text.count('!') / max(len(text), 1)
        exclamation_penalty = min(0.3, exclamation_ratio * 100)
        
        # Check for proper sentence ending
        sentences = re.split(r'[.!?]+', text)
        proper_endings = len([s for s in sentences if s.strip()])
        
        score = 0.7 - exclamation_penalty
        return max(0, min(1, score))
    
    def _check_complexity(self, text: str) -> float:
        """Check word complexity (0-1)"""
        words = text.split()
        if not words:
            return 0.5
        
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Optimal range: 4-7 characters
        if 4 <= avg_word_length <= 7:
            return 0.8
        elif avg_word_length < 4:
            return 0.5  # Too simple
        else:
            return 0.6  # Too complex
    
    def _check_grammar(self, text: str) -> float:
        """Basic grammar checking (0-1)"""
        # This is simplified - in production, use a grammar checking library
        # For now, check basic patterns
        
        # Check for double spaces
        double_spaces = text.count('  ')
        
        # Check for spaces before punctuation
        space_before_punct = len(re.findall(r'\s+[.,!?]', text))
        
        errors = double_spaces + space_before_punct
        error_ratio = errors / max(len(text.split()), 1)
        
        return max(0, 1 - error_ratio * 10)
    
    def _analyze_tone(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment distribution for Chart 2 using VADER sentiment analysis.
        Returns format: {labels: ['Positive', 'Negative', 'Neutral'], values: [0.4, 0.3, 0.3]}
        """
        # Check if text is empty or too short
        if not text or len(text.strip()) < 3:
            print(f"[WARN] Text too short for sentiment analysis: length={len(text) if text else 0}")
            return {
                "labels": ["Positive", "Negative", "Neutral"],
                "values": [0.33, 0.33, 0.34]
            }
        
        # Use VADER if available (proper sentiment analysis)
        print(f"[DEBUG] VADER_AVAILABLE: {VADER_AVAILABLE}, _vader_analyzer: {_vader_analyzer is not None}")
        if VADER_AVAILABLE and _vader_analyzer:
            try:
                # Analyze sentiment using VADER
                sentiment_scores = _vader_analyzer.polarity_scores(text)
                
                print(f"[DEBUG] VADER sentiment scores: pos={sentiment_scores['pos']:.4f}, neg={sentiment_scores['neg']:.4f}, neu={sentiment_scores['neu']:.4f}, compound={sentiment_scores.get('compound', 0):.4f}")
                
                # VADER returns proportions that already sum to 1.0
                # Format: {labels: ['Positive', 'Negative', 'Neutral'], values: [pos, neg, neu]}
                pos_val = round(sentiment_scores['pos'], 2)
                neg_val = round(sentiment_scores['neg'], 2)
                neu_val = round(sentiment_scores['neu'], 2)
                
                # Ensure values sum to 1.0 (handle floating point precision)
                total = pos_val + neg_val + neu_val
                if total > 0:
                    # Normalize to ensure exact sum of 1.0
                    pos_val = round(pos_val / total, 2)
                    neg_val = round(neg_val / total, 2)
                    neu_val = round(1.0 - pos_val - neg_val, 2)  # Ensure sum is exactly 1.0
                else:
                    # Fallback if all zeros
                    print("[WARN] All sentiment scores are zero, using fallback")
                    pos_val, neg_val, neu_val = 0.33, 0.33, 0.34
                
                result = {
                    "labels": ["Positive", "Negative", "Neutral"],
                    "values": [pos_val, neg_val, neu_val]
                }
                print(f"[DEBUG] Final sentiment result: {result}")
                return result
            except Exception as e:
                print(f"[WARN] VADER sentiment analysis failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
        
        # Fallback: Simple keyword-based analysis (less accurate)
        text_lower = text.lower()
        words = text.split()
        
        # Count sentiment keywords
        positive_keywords = len(re.findall(
            r'\b(excellent|great|wonderful|success|achievement|good|amazing|love|best|perfect)\b',
            text_lower
        ))
        negative_keywords = len(re.findall(
            r'\b(terrible|awful|disaster|crisis|danger|threat|bad|hate|worst|horrible)\b',
            text_lower
        ))
        
        # Calculate proportions
        total_words = len(words)
        if total_words > 0:
            pos_ratio = min(positive_keywords / total_words * 10, 0.5)  # Cap at 50%
            neg_ratio = min(negative_keywords / total_words * 10, 0.5)  # Cap at 50%
            neu_ratio = max(1.0 - pos_ratio - neg_ratio, 0.0)  # Rest is neutral
            
            # Normalize to sum to 1.0
            total = pos_ratio + neg_ratio + neu_ratio
            if total > 0:
                pos_val = round(pos_ratio / total, 2)
                neg_val = round(neg_ratio / total, 2)
                neu_val = round(1.0 - pos_val - neg_val, 2)
            else:
                pos_val, neg_val, neu_val = 0.33, 0.33, 0.34
        else:
            pos_val, neg_val, neu_val = 0.33, 0.33, 0.34
        
        return {
            "labels": ["Positive", "Negative", "Neutral"],
            "values": [pos_val, neg_val, neu_val]
        }