import re
import string
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Ensure required NLTK data is present
def _ensure_nltk():
    import os
    # Use /tmp for Lambda (writable) or ~/nltk_data for local
    if os.path.exists('/tmp'):
        nltk_data_dir = '/tmp/nltk_data'
    else:
        nltk_data_dir = os.path.expanduser('~/nltk_data')
    
    try:
        os.makedirs(nltk_data_dir, exist_ok=True)
    except (OSError, PermissionError):
        # If can't create, try /tmp as fallback
        nltk_data_dir = '/tmp/nltk_data'
        try:
            os.makedirs(nltk_data_dir, exist_ok=True)
        except:
            pass  # Will use fallback methods
    
    # Add to NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk_download(pkg, quiet=True, download_dir=nltk_data_dir)
        except Exception as e:
            # Silently continue if download fails - will use fallback methods
            pass

try:
    _ensure_nltk()
except Exception:
    # If NLTK setup fails, continue with fallback methods
    pass

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english")) if stopwords.words else set()
_VADER = SentimentIntensityAnalyzer()

# Enhanced clickbait patterns - more comprehensive
CLICKBAIT_PATTERNS = [
    r'\b(shocking|unbelievable|incredible|amazing|stunning|devastating)\b',
    r'\b(you won\'t believe|won\'t believe|can\'t believe|can\'t believe)\b',
    r'\b(must see|must read|must watch|breaking|exclusive|urgent)\b',
    r'\b(this will blow your mind|mind blown|jaw dropping|epic)\b',
    r'\b(scandal|outrageous|absurd|ridiculous|ludicrous)\b',
    r'\b(explosive|bombshell|devastating|crushing|destroyed)\b',
    r'\b(secret|revealed|exposed|leaked|insider)\b',
    r'\b(biggest ever|never before|first time|historic|unprecedented)\b',
    r'\b(what happens next|the truth about|what they don\'t want)\b',
    r'\b(this changes everything|game changer|mind blown)\b',
    r'\b(alert|warning|urgent|breaking news|just in)\b',
    r'\b(outrage|fury|rage|anger|furious|enraged)\b'
]

# Emotional intensity words
EMOTIONAL_WORDS = [
    'outrage', 'fury', 'rage', 'anger', 'furious', 'enraged',
    'shock', 'shocked', 'shocking', 'stunned', 'stunning',
    'devastating', 'devastated', 'crushing', 'destroyed',
    'epic', 'legendary', 'historic', 'unprecedented',
    'scandal', 'scandalous', 'controversial', 'outrageous',
    'absurd', 'ridiculous', 'ludicrous', 'insane',
    'incredible', 'amazing', 'stunning', 'breathtaking',
    'terrifying', 'horrifying', 'disgusting', 'appalling'
]

# Intensifiers and hyperbole
INTENSIFIERS = [
    'very', 'extremely', 'highly', 'incredibly', 'really', 'totally',
    'absolutely', 'completely', 'utterly', 'entirely', 'perfectly',
    'massively', 'hugely', 'enormously', 'tremendously', 'vastly',
    'dramatically', 'significantly', 'substantially', 'considerably'
]

# Uncertainty and tentative language
TENTATIVE_WORDS = [
    'maybe', 'perhaps', 'might', 'could', 'possibly', 'probably',
    'likely', 'seems', 'appears', 'suggests', 'indicates',
    'allegedly', 'reportedly', 'supposedly', 'purportedly',
    'rumored', 'claimed', 'asserted', 'maintained'
]

# Evidence-based language
EVIDENCE_WORDS = [
    'according to', 'research shows', 'studies indicate', 'data reveals',
    'evidence suggests', 'findings show', 'analysis reveals',
    'statistics show', 'reports indicate', 'investigation reveals',
    'because', 'therefore', 'thus', 'consequently', 'as a result'
]

# Professional/Institutional language (indicates legitimate news)
PROFESSIONAL_WORDS = [
    'officials', 'authority', 'department', 'agency', 'institute',
    'study', 'research', 'analysis', 'report', 'findings',
    'confirmed', 'announced', 'stated', 'reported', 'according to',
    'budget', 'funding', 'grants', 'partnerships', 'consultation',
    'planning', 'development', 'implementation', 'expansion'
]

# Balanced reporting indicators
BALANCED_WORDS = [
    'however', 'although', 'while', 'despite', 'nevertheless',
    'concerns', 'challenges', 'issues', 'problems', 'criticism',
    'supportive', 'opposed', 'mixed', 'varied', 'different'
]

@dataclass
class EnhancedLinguisticFeatures:
    # Core sensationalism indicators (most important)
    clickbait_score: float
    emotional_intensity: float
    exclamation_density: float
    caps_density: float
    question_density: float
    
    # Sentiment and bias (enhanced)
    sentiment_polarity: float
    sentiment_subjectivity: float
    sentiment_intensity: float
    sentiment_balance: float  # New: how balanced the sentiment is
    
    # Language patterns (enhanced)
    intensifier_ratio: float
    tentative_ratio: float
    evidence_ratio: float
    professional_ratio: float  # New: professional language
    balanced_ratio: float      # New: balanced reporting
    
    # Text structure
    avg_sentence_length: float
    avg_word_length: float
    text_length: int
    word_count: int
    
    # Repetition and redundancy
    repetition_ratio: float
    unique_word_ratio: float
    
    # Specific patterns
    all_caps_words: int
    exclamation_count: int
    question_count: int
    clickbait_matches: int
    emotional_word_count: int
    
    # New: Content quality indicators
    professional_word_count: int
    balanced_word_count: int
    data_references: int  # Numbers, percentages, statistics

def normalize_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)  # Keep only alphanumeric, spaces, and basic punctuation
    
    return text.strip()

def tokenize_text(text: str) -> List[str]:
    """Tokenize and clean text"""
    try:
        tokens = word_tokenize(text.lower())
    except:
        tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords and short words
    tokens = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
    return tokens

def count_pattern_matches(text: str, patterns: List[str]) -> int:
    """Count matches for multiple regex patterns"""
    total_matches = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        total_matches += len(matches)
    return total_matches

def count_word_matches(text: str, word_list: List[str]) -> int:
    """Count exact word matches"""
    text_lower = text.lower()
    count = 0
    for word in word_list:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    return count

def calculate_repetition_ratio(tokens: List[str]) -> float:
    """Calculate ratio of repeated words"""
    if not tokens:
        return 0.0
    
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
    
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    return repeated_words / len(word_counts) if word_counts else 0.0

def count_data_references(text: str) -> int:
    """Count references to data, statistics, numbers"""
    # Look for percentages, numbers, years, statistics
    patterns = [
        r'\d+%',  # Percentages
        r'\d{4}',  # Years
        r'\$\d+',  # Money amounts
        r'\d+\.\d+',  # Decimal numbers
        r'study|research|analysis|report|findings|data|statistics'
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count

def extract_enhanced_linguistic_features(raw_text: str) -> EnhancedLinguisticFeatures:
    """Extract comprehensive linguistic features with enhanced discrimination"""
    text = normalize_text(raw_text)
    tokens = tokenize_text(text)
    words = text.split()
    
    # Core sensationalism indicators
    clickbait_matches = count_pattern_matches(text, CLICKBAIT_PATTERNS)
    clickbait_score = clickbait_matches / max(1, len(words))
    
    emotional_word_count = count_word_matches(text, EMOTIONAL_WORDS)
    emotional_intensity = emotional_word_count / max(1, len(words))
    
    exclamation_count = text.count('!')
    exclamation_density = exclamation_count / max(1, len(words))
    
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_density = len(caps_words) / max(1, len(words))
    
    question_count = text.count('?')
    question_density = question_count / max(1, len(words))
    
    # Enhanced sentiment analysis
    try:
        sentiment = _VADER.polarity_scores(text)
        sentiment_polarity = sentiment['compound']
        sentiment_intensity = abs(sentiment['pos'] - sentiment['neg'])
        
        # Calculate sentiment balance (how evenly distributed positive/negative/neutral are)
        pos = sentiment['pos']
        neg = sentiment['neg']
        neu = sentiment['neu']
        sentiment_balance = 1.0 - abs(pos - neg)  # Higher = more balanced
    except:
        sentiment_polarity = 0.0
        sentiment_intensity = 0.0
        sentiment_balance = 0.0
    
    try:
        sentiment_subjectivity = float(TextBlob(text).sentiment.subjectivity)
    except:
        sentiment_subjectivity = 0.0
    
    # Language patterns
    intensifier_count = count_word_matches(text, INTENSIFIERS)
    intensifier_ratio = intensifier_count / max(1, len(tokens))
    
    tentative_count = count_word_matches(text, TENTATIVE_WORDS)
    tentative_ratio = tentative_count / max(1, len(tokens))
    
    evidence_count = count_word_matches(text, EVIDENCE_WORDS)
    evidence_ratio = evidence_count / max(1, len(tokens))
    
    # New: Professional and balanced language
    professional_count = count_word_matches(text, PROFESSIONAL_WORDS)
    professional_ratio = professional_count / max(1, len(tokens))
    
    balanced_count = count_word_matches(text, BALANCED_WORDS)
    balanced_ratio = balanced_count / max(1, len(tokens))
    
    # Text structure
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = len(words) / max(1, len(sentences))
    avg_word_length = sum(len(w) for w in words) / max(1, len(words))
    
    # Repetition analysis
    repetition_ratio = calculate_repetition_ratio(tokens)
    unique_word_ratio = len(set(tokens)) / max(1, len(tokens))
    
    # Data references
    data_references = count_data_references(text)
    
    return EnhancedLinguisticFeatures(
        # Core sensationalism indicators
        clickbait_score=clickbait_score,
        emotional_intensity=emotional_intensity,
        exclamation_density=exclamation_density,
        caps_density=caps_density,
        question_density=question_density,
        
        # Sentiment and bias (enhanced)
        sentiment_polarity=sentiment_polarity,
        sentiment_subjectivity=sentiment_subjectivity,
        sentiment_intensity=sentiment_intensity,
        sentiment_balance=sentiment_balance,
        
        # Language patterns (enhanced)
        intensifier_ratio=intensifier_ratio,
        tentative_ratio=tentative_ratio,
        evidence_ratio=evidence_ratio,
        professional_ratio=professional_ratio,
        balanced_ratio=balanced_ratio,
        
        # Text structure
        avg_sentence_length=avg_sentence_length,
        avg_word_length=avg_word_length,
        text_length=len(text),
        word_count=len(words),
        
        # Repetition and redundancy
        repetition_ratio=repetition_ratio,
        unique_word_ratio=unique_word_ratio,
        
        # Specific patterns
        all_caps_words=len(caps_words),
        exclamation_count=exclamation_count,
        question_count=question_count,
        clickbait_matches=clickbait_matches,
        emotional_word_count=emotional_word_count,
        
        # New: Content quality indicators
        professional_word_count=professional_count,
        balanced_word_count=balanced_count,
        data_references=data_references,
    )

def features_to_array(feats: EnhancedLinguisticFeatures) -> np.ndarray:
    """Convert features to numpy array"""
    return np.array([
        # Core sensationalism indicators (5)
        feats.clickbait_score,
        feats.emotional_intensity,
        feats.exclamation_density,
        feats.caps_density,
        feats.question_density,
        
        # Sentiment and bias (enhanced) (4)
        feats.sentiment_polarity,
        feats.sentiment_subjectivity,
        feats.sentiment_intensity,
        feats.sentiment_balance,
        
        # Language patterns (enhanced) (5)
        feats.intensifier_ratio,
        feats.tentative_ratio,
        feats.evidence_ratio,
        feats.professional_ratio,
        feats.balanced_ratio,
        
        # Text structure (4)
        feats.avg_sentence_length,
        feats.avg_word_length,
        float(feats.text_length),
        float(feats.word_count),
        
        # Repetition and redundancy (2)
        feats.repetition_ratio,
        feats.unique_word_ratio,
        
        # Specific patterns (5)
        float(feats.all_caps_words),
        float(feats.exclamation_count),
        float(feats.question_count),
        float(feats.clickbait_matches),
        float(feats.emotional_word_count),
        
        # New: Content quality indicators (3)
        float(feats.professional_word_count),
        float(feats.balanced_word_count),
        float(feats.data_references),
    ], dtype=float)

def build_linguistic_feature_matrix(texts: List[str]) -> np.ndarray:
    """Build feature matrix for multiple texts"""
    rows = [features_to_array(extract_enhanced_linguistic_features(t)) for t in texts]
    return np.vstack(rows) if rows else np.zeros((0, 28), dtype=float)

def get_feature_names() -> List[str]:
    """Get feature names for interpretation"""
    return [
        # Core sensationalism indicators
        'clickbait_score', 'emotional_intensity', 'exclamation_density', 
        'caps_density', 'question_density',
        
        # Sentiment and bias (enhanced)
        'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_intensity', 'sentiment_balance',
        
        # Language patterns (enhanced)
        'intensifier_ratio', 'tentative_ratio', 'evidence_ratio', 'professional_ratio', 'balanced_ratio',
        
        # Text structure
        'avg_sentence_length', 'avg_word_length', 'text_length', 'word_count',
        
        # Repetition and redundancy
        'repetition_ratio', 'unique_word_ratio',
        
        # Specific patterns
        'all_caps_words', 'exclamation_count', 'question_count', 
        'clickbait_matches', 'emotional_word_count',
        
        # New: Content quality indicators
        'professional_word_count', 'balanced_word_count', 'data_references',
    ]

        
        # Sentiment and bias (enhanced)
        'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_intensity', 'sentiment_balance',
        
        # Language patterns (enhanced)
        'intensifier_ratio', 'tentative_ratio', 'evidence_ratio', 'professional_ratio', 'balanced_ratio',
        
        # Text structure
        'avg_sentence_length', 'avg_word_length', 'text_length', 'word_count',
        
        # Repetition and redundancy
        'repetition_ratio', 'unique_word_ratio',
        
        # Specific patterns
        'all_caps_words', 'exclamation_count', 'question_count', 
        'clickbait_matches', 'emotional_word_count',
        
        # New: Content quality indicators
        'professional_word_count', 'balanced_word_count', 'data_references',
    ]

        
        # Sentiment and bias (enhanced)
        'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_intensity', 'sentiment_balance',
        
        # Language patterns (enhanced)
        'intensifier_ratio', 'tentative_ratio', 'evidence_ratio', 'professional_ratio', 'balanced_ratio',
        
        # Text structure
        'avg_sentence_length', 'avg_word_length', 'text_length', 'word_count',
        
        # Repetition and redundancy
        'repetition_ratio', 'unique_word_ratio',
        
        # Specific patterns
        'all_caps_words', 'exclamation_count', 'question_count', 
        'clickbait_matches', 'emotional_word_count',
        
        # New: Content quality indicators
        'professional_word_count', 'balanced_word_count', 'data_references',
    ]

        
        # Sentiment and bias (enhanced)
        'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_intensity', 'sentiment_balance',
        
        # Language patterns (enhanced)
        'intensifier_ratio', 'tentative_ratio', 'evidence_ratio', 'professional_ratio', 'balanced_ratio',
        
        # Text structure
        'avg_sentence_length', 'avg_word_length', 'text_length', 'word_count',
        
        # Repetition and redundancy
        'repetition_ratio', 'unique_word_ratio',
        
        # Specific patterns
        'all_caps_words', 'exclamation_count', 'question_count', 
        'clickbait_matches', 'emotional_word_count',
        
        # New: Content quality indicators
        'professional_word_count', 'balanced_word_count', 'data_references',
    ]
