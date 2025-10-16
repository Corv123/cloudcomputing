# analyzers/language_analyzer.py
# Analyzes language quality and generates data for Chart 1 & 2

import re
from typing import Dict, Any

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
        chart2_data = self._analyze_tone(text)
        
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
        """Analyze tone distribution for Chart 2 (simplified)"""
        text_lower = text.lower()
        
        # Simple keyword-based tone analysis
        neutral_words = len(text.split())
        sensational = len(re.findall(
            r'\b(shocking|unbelievable|amazing|incredible|urgent|breaking)\b',
            text_lower
        ))
        negative = len(re.findall(
            r'\b(terrible|awful|disaster|crisis|danger|threat)\b',
            text_lower
        ))
        positive = len(re.findall(
            r'\b(excellent|great|wonderful|success|achievement)\b',
            text_lower
        ))
        
        total = sensational + negative + positive + max(neutral_words - 100, 0) / 10
        
        return {
            "labels": ["Neutral", "Sensational", "Negative", "Positive"],
            "values": [
                round(max(neutral_words - sensational - negative - positive, 0) / total * 100, 1) if total > 0 else 70,
                round(sensational / total * 100, 1) if total > 0 else 10,
                round(negative / total * 100, 1) if total > 0 else 10,
                round(positive / total * 100, 1) if total > 0 else 10
            ]
        }
