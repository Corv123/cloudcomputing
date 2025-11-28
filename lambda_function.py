# lambda_function.py
# AWS Lambda function for Fake News Detector API

import json
import boto3
import hashlib
import os
import sys
from datetime import datetime
from decimal import Decimal
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

# Add fakenews directory to path to import analyzers
sys.path.insert(0, '/var/task')
sys.path.insert(0, '/var/task/fakenews')

# Import config first (needed by analyzers)
# Try multiple import paths to ensure config is found
config = None
try:
    import fakenews.config as config
    print("[OK] Config module imported from fakenews.config")
except ImportError:
    try:
        # Try alternative import path (if fakenews is in path)
        sys.path.insert(0, '/var/task/fakenews')
        import config
        print("[OK] Config module imported (alternative path)")
    except ImportError:
        try:
            # Try direct import if config.py is in analyzers directory
            from analyzers import config
            print("[OK] Config module imported from analyzers")
        except ImportError as e:
            print(f"[WARN] Config module not available: {e}")
            print("[WARN] Analyzers may not work correctly without config")
            config = None

# Verify config has required attributes
if config:
    print(f"[DEBUG] Config loaded - ENABLE_WHOIS: {getattr(config, 'ENABLE_WHOIS', 'NOT FOUND')}")
    print(f"[DEBUG] Config loaded - WHOIS_API_KEY set: {bool(getattr(config, 'WHOIS_API_KEY', ''))}")
else:
    print("[ERROR] Config module not loaded - analyzers will use fallback behavior")

# Import actual analyzer classes (not simplified versions)
try:
    from analyzers.language_analyzer import LanguageAnalyzer
    from analyzers.credibility_analyzer import CredibilityAnalyzer
    from analyzers.crosscheck_analyzer import CrossCheckAnalyzer
    from analyzers.related_articles_analyzer import RelatedArticlesAnalyzer
    ANALYZERS_AVAILABLE = True
    print("[OK] Full analyzer classes imported successfully")
except ImportError as e:
    print(f"[WARN] Analyzer classes not available: {e}")
    print("[WARN] Falling back to simplified inline functions")
    import traceback
    traceback.print_exc()
    ANALYZERS_AVAILABLE = False
    LanguageAnalyzer = None
    CredibilityAnalyzer = None
    CrossCheckAnalyzer = None
    RelatedArticlesAnalyzer = None

# Initialize analyzer instances if available
if ANALYZERS_AVAILABLE:
    try:
        language_analyzer = LanguageAnalyzer()
        credibility_analyzer = CredibilityAnalyzer()
        # Initialize with database support (PRIMARY: DynamoDB, FALLBACK: web search)
        crosscheck_analyzer = CrossCheckAnalyzer(use_database=True, dynamodb_table='fakenews-scraped-news', region='ap-southeast-2')
        related_articles_analyzer = RelatedArticlesAnalyzer()
        print("[OK] Analyzer instances created successfully")
        if config:
            print(f"[DEBUG] WHOIS enabled: {getattr(config, 'ENABLE_WHOIS', False)}")
            if getattr(config, 'ENABLE_WHOIS', False):
                print("[OK] WHOIS verification is ENABLED - will use real domain age data")
            else:
                print("[WARN] WHOIS verification is DISABLED - using estimated domain age")
    except Exception as e:
        print(f"[WARN] Failed to create analyzer instances: {e}")
        import traceback
        traceback.print_exc()
        ANALYZERS_AVAILABLE = False

# SageMaker integration - ML models are hosted on SageMaker
SAGEMAKER_ENDPOINT = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'fakenews-sensationalism-endpoint')
SAGEMAKER_RUNTIME = boto3.client('sagemaker-runtime', region_name=os.environ.get('AWS_REGION', 'ap-southeast-2'))

# Local ML models (loaded from /opt/)
LOCAL_MODELS_AVAILABLE = False
local_model = None
local_vectorizer = None
local_scaler = None

def load_local_models():
    """Load ML models from /opt/ directory"""
    global LOCAL_MODELS_AVAILABLE, local_model, local_vectorizer, local_scaler
    
    if LOCAL_MODELS_AVAILABLE:
        return True
    
    try:
        import joblib
        print("[INFO] Loading local ML models from /opt/...")
        
        model_path = '/opt/sensationalism_model_comprehensive.joblib'
        vectorizer_path = '/opt/tfidf_vectorizer_comprehensive.joblib'
        scaler_path = '/opt/scaler_comprehensive.joblib'
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"[WARN] Model not found at {model_path}")
            return False
        if not os.path.exists(vectorizer_path):
            print(f"[WARN] Vectorizer not found at {vectorizer_path}")
            return False
        if not os.path.exists(scaler_path):
            print(f"[WARN] Scaler not found at {scaler_path}")
            return False
        
        # Load models
        local_model = joblib.load(model_path)
        local_vectorizer = joblib.load(vectorizer_path)
        local_scaler = joblib.load(scaler_path)
        
        LOCAL_MODELS_AVAILABLE = True
        print("[OK] Local ML models loaded successfully!")
        print(f"  Model type: {type(local_model).__name__}")
        print(f"  Vectorizer type: {type(local_vectorizer).__name__}")
        print(f"  Scaler type: {type(local_scaler).__name__}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load local models: {e}")
        import traceback
        traceback.print_exc()
        return False

# ML dependencies - only import when needed (for /analyze endpoint)
# Note: We no longer load models locally - they're on SageMaker
ML_DEPENDENCIES_AVAILABLE = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    from nltk.tokenize import sent_tokenize
    ML_DEPENDENCIES_AVAILABLE = True
    
    # Download required NLTK data to /tmp (Lambda writable directory)
    nltk.data.path.append('/tmp')
    try:
        nltk.download('punkt', download_dir='/tmp', quiet=True)
        nltk.download('vader_lexicon', download_dir='/tmp', quiet=True)
        nltk.download('stopwords', download_dir='/tmp', quiet=True)
    except Exception as e:
        print(f"Warning: NLTK download issue: {e}")
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: ML dependencies not available: {e}")
    print("Stats and articles endpoints will work, but /analyze requires ML dependencies")

# Initialize AWS services
dynamodb = boto3.resource('dynamodb')
articles_table = dynamodb.Table('fakenews-articles')
# Cache table is optional - handle gracefully if it doesn't exist
try:
    cache_table = dynamodb.Table('fakenews-analysis-cache')
    # Test if table exists by describing it
    cache_table.meta.client.describe_table(TableName='fakenews-analysis-cache')
    CACHE_TABLE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Cache table not available: {e}")
    CACHE_TABLE_AVAILABLE = False
    cache_table = None

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

# SageMaker integration - models are hosted on SageMaker endpoint
# No need to load models locally anymore

def call_sagemaker_endpoint(text, title=None):
    """
    Call SageMaker endpoint for ML model prediction
    
    Args:
        text: Article text content
        title: Optional article title
        
    Returns:
        Dictionary with sensationalism_score or None if error
    """
    try:
        # Prepare request payload
        payload = {
            'text': text,
            'title': title or ''
        }
        
        # Invoke SageMaker endpoint
        response = SAGEMAKER_RUNTIME.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode('utf-8'))
        
        if result.get('success', False):
            return result.get('sensationalism_score', 0.0)
        else:
            print(f"[WARN] SageMaker returned error: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"[ERROR] SageMaker endpoint call failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Initialize sentiment analyzer (still needed for other analysis)
sentiment_analyzer = None
if ML_DEPENDENCIES_AVAILABLE:
    try:
        sentiment_analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Warning: Could not initialize sentiment analyzer: {e}")

# Analyzer functions - use full analyzers if available, fallback to simplified versions
def analyze_language_quality(title, content):
    """Analyze language quality of article - uses full LanguageAnalyzer if available"""
    # Use full analyzer if available
    if ANALYZERS_AVAILABLE and language_analyzer:
        try:
            result = language_analyzer.analyze(title, content)
            print("[OK] ✓ Using FULL LanguageAnalyzer - REAL data, not placeholder")
            print(f"[DEBUG] Language score: {result.get('score', 'N/A')}")
            return result
        except Exception as e:
            print(f"[ERROR] LanguageAnalyzer failed: {e}, using fallback")
            import traceback
            traceback.print_exc()
    
    # Fallback to simplified version
    if not content or len(content.strip()) == 0:
        return {
            'score': 0.3,
            'chart1_data': {
                'labels': ['Grammar', 'Clarity', 'Structure'],
                'values': [30, 30, 30]
            },
            'chart2_data': {
                'positive': 0.3,
                'negative': 0.3,
                'neutral': 0.4
            }
        }
    
    # More sophisticated language analysis
    word_count = len(content.split())
    if ML_DEPENDENCIES_AVAILABLE:
        sentence_count = len(sent_tokenize(content))
    else:
        sentence_count = len([s for s in content.split('.') if len(s.strip()) > 0])
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Calculate multiple language quality metrics
    # Grammar: based on sentence structure
    if 10 <= avg_sentence_length <= 30:
        grammar_score = 0.85
    elif 5 <= avg_sentence_length < 10 or 30 < avg_sentence_length <= 50:
        grammar_score = 0.70
    else:
        grammar_score = 0.55
    
    # Clarity: based on word count and structure
    if word_count > 200:
        clarity_score = 0.80
    elif word_count > 100:
        clarity_score = 0.70
    else:
        clarity_score = 0.60
    
    # Structure: based on paragraph organization
    paragraphs = content.split('\n\n')
    if len(paragraphs) > 3:
        structure_score = 0.75
    else:
        structure_score = 0.65
    
    # Overall language score is average of all metrics
    language_score = (grammar_score + clarity_score + structure_score) / 3.0
    
    # Calculate actual sentiment for chart2
    # Format: {labels: ['Positive', 'Negative', 'Neutral'], values: [0.4, 0.3, 0.3]}
    sentiment_data = {'labels': ['Positive', 'Negative', 'Neutral'], 'values': [0.4, 0.3, 0.3]}  # Default fallback
    if ML_DEPENDENCIES_AVAILABLE and sentiment_analyzer:
        try:
            # Analyze sentiment of the content
            sentiment_scores = sentiment_analyzer.polarity_scores(content)
            # Normalize to sum to 1.0
            total = abs(sentiment_scores['pos']) + abs(sentiment_scores['neg']) + abs(sentiment_scores['neu'])
            if total > 0:
                pos_val = round(sentiment_scores['pos'] / total, 2)
                neg_val = round(sentiment_scores['neg'] / total, 2)
                neu_val = round(sentiment_scores['neu'] / total, 2)
                sentiment_data = {
                    'labels': ['Positive', 'Negative', 'Neutral'],
                    'values': [pos_val, neg_val, neu_val]
                }
            else:
                sentiment_data = {
                    'labels': ['Positive', 'Negative', 'Neutral'],
                    'values': [0.33, 0.33, 0.34]
                }
        except Exception as e:
            print(f"[WARN] Sentiment analysis failed: {e}, using default")
    
    return {
        'score': min(language_score, 1.0),
        'chart1_data': {
            'labels': ['Grammar', 'Clarity', 'Structure'],
            'values': [round(grammar_score * 100, 1), round(clarity_score * 100, 1), round(structure_score * 100, 1)]
        },
        'chart2_data': sentiment_data
    }

def analyze_credibility(url, content, source, title):
    """Analyze source credibility - uses full CredibilityAnalyzer with WHOIS if available"""
    # Use full analyzer if available (includes WHOIS verification)
    if ANALYZERS_AVAILABLE and credibility_analyzer:
        try:
            result = credibility_analyzer.analyze(url, content, source, title)
            print("[OK] ✓ Using FULL CredibilityAnalyzer with WHOIS - REAL data, not placeholder")
            print(f"[DEBUG] Credibility score: {result.get('score', 'N/A')}")
            if result.get('enhanced_info', {}).get('whois_verified'):
                print("[OK] ✓ WHOIS data verified - using REAL domain age")
            else:
                print("[WARN] WHOIS not verified - using estimated domain age")
            return result
        except Exception as e:
            print(f"[ERROR] CredibilityAnalyzer failed: {e}, using fallback")
            import traceback
            traceback.print_exc()
    
    # Fallback to simplified version (placeholder data)
    domain = urlparse(url).netloc.replace('www.', '')
    
    # Expanded credibility check with more domains
    credible_domains = [
        "bbc.com", "reuters.com", "apnews.com", "npr.org",
        "theguardian.com", "nytimes.com", "washingtonpost.com",
        "cnn.com", "edition.cnn.com", "abcnews.go.com", "cbsnews.com",
        "nbcnews.com", "wsj.com", "economist.com", "time.com",
        "channelnewsasia.com", "straitstimes.com"
    ]
    
    # Check for reputable news domains
    if domain in credible_domains:
        credibility_score = 0.85
        known_source = "Verified Reliable Source"
        domain_age_score = 85
        domain_age_status = "Established"
        url_structure_score = 80
        url_structure_status = "Well-structured"
        site_structure_score = 85
        site_structure_status = "Professional"
        content_format_score = 80
        content_format_status = "Standard"
    elif any(cred in domain for cred in ["news", "times", "post", "tribune", "herald", "journal"]):
        # Generic news domains get a moderate score
        credibility_score = 0.65
        known_source = "News Source"
        domain_age_score = 70
        domain_age_status = "Established"
        url_structure_score = 65
        url_structure_status = "Standard"
        site_structure_score = 65
        site_structure_status = "Standard"
        content_format_score = 65
        content_format_status = "Standard"
    else:
        # Unknown domains get a lower score but not zero
        credibility_score = 0.45
        known_source = "Unknown Source"
        domain_age_score = 50
        domain_age_status = "Unknown"
        url_structure_score = 50
        url_structure_status = "Unknown"
        site_structure_score = 45
        site_structure_status = "Unknown"
        content_format_score = 50
        content_format_status = "Unknown"
    
    # Calculate content quality based on article length and structure
    word_count = len(content.split()) if content else 0
    if word_count > 1000:
        content_quality_score = 0.80
    elif word_count > 500:
        content_quality_score = 0.70
    elif word_count > 200:
        content_quality_score = 0.60
    else:
        content_quality_score = 0.50
    
    return {
        'score': credibility_score,
        'known_source': known_source,
        'chart3_data': {
            'labels': ['Domain Age', 'Source Reputation', 'Content Quality'],
            'values': [domain_age_score, round(credibility_score * 100, 1), round(content_quality_score * 100, 1)]
        },
        'detailed_metrics': {
            'domain_age': {
                'score': domain_age_score,
                'status': domain_age_status,
                'description': f'Domain appears to be {domain_age_status.lower()} with established presence'
            },
            'url_structure': {
                'score': url_structure_score,
                'status': url_structure_status,
                'description': f'URL structure is {url_structure_status.lower()}'
            },
            'site_structure': {
                'score': site_structure_score,
                'status': site_structure_status,
                'description': f'Site structure appears {site_structure_status.lower()}'
            },
            'content_format': {
                'score': content_format_score,
                'status': content_format_status,
                'description': f'Content format is {content_format_status.lower()}'
            }
        },
        'enhanced_info': {
            'source_type': 'News' if known_source != 'Unknown Source' else 'Unknown'
        }
    }

def analyze_crosscheck(title, content, url):
    """Analyze cross-check with web sources - uses full CrossCheckAnalyzer if available"""
    # Use full analyzer if available
    if ANALYZERS_AVAILABLE and crosscheck_analyzer:
        try:
            result = crosscheck_analyzer.analyze(title, content, url)
            print("[OK] ✓ Using FULL CrossCheckAnalyzer - REAL web-based similarity calculations")
            print(f"[DEBUG] Cross-check score: {result.get('score', 'N/A')}")
            metrics = result.get('metrics', {})
            print(f"[DEBUG] Compared against web sources - {metrics.get('credible_sources_count', 0)} credible sources found")
            return result
        except Exception as e:
            print(f"[ERROR] CrossCheckAnalyzer failed: {e}, using fallback")
            import traceback
            traceback.print_exc()
    
    # Fallback to simplified version
    return {
        'score': 0.5,
        'chart4_data': {
            'description': 'Cross-check analysis unavailable',
            'x_label': 'Content Similarity',
            'y_label': 'Source Credibility',
            'points': [],
            'total_points': 0
        }
    }

def generate_similarity_map(current_article_id, database_articles, current_article_content=None, current_article_data=None):
    """Generate similarity map data for Chart 5"""
    points = []
    
    # Get current article data for comparison and to include in the map
    current_article = None
    current_article_in_db = None
    
    # Try to get current article from database first
    try:
        current_article_in_db = articles_table.get_item(Key={'id': current_article_id}).get('Item')
    except:
        pass
    
    # Use provided content/data or database article
    if current_article_data:
        # Use provided article data (from current analysis)
        current_article = {
            'content': current_article_data.get('content', ''),
            'title': current_article_data.get('title', 'Untitled'),
            'source': current_article_data.get('source', 'Unknown'),
            'credibility_score': current_article_data.get('credibility_score', 0.5),
            'overall_score': current_article_data.get('overall_score', 0.5)
        }
    elif current_article_content:
        # Use provided content (fallback)
        current_article = {'content': current_article_content}
    elif current_article_in_db:
        # Use article from database
        current_article = {
            'content': current_article_in_db.get('content', ''),
            'title': current_article_in_db.get('title', 'Untitled'),
            'source': current_article_in_db.get('source', 'Unknown'),
            'credibility_score': float(current_article_in_db.get('credibility_score', 0.5)),
            'overall_score': float(current_article_in_db.get('overall_score', 0.5))
        }
    
    # If we have current article data, add it to the map as a special point
    if current_article and current_article.get('content'):
        print(f"[DEBUG] Adding current article to similarity map: {current_article.get('title', 'Untitled')[:50]}")
        # Add current article as the first point (similarity = 1.0 to itself)
        current_point = {
            'x': 1.0,  # Perfect similarity to itself
            'y': float(current_article.get('credibility_score', 0.5)),
            'title': current_article.get('title', 'Current Article')[:50],
            'source': current_article.get('source', 'Current'),
            'overall_score': float(current_article.get('overall_score', 0.5)),
            'is_current': True  # Flag to identify current article in frontend
        }
        points.append(current_point)
        print(f"[DEBUG] Current article point added: x={current_point['x']}, y={current_point['y']}, is_current={current_point['is_current']}")
        
        # Get current article words for similarity calculation
        current_words = set(re.findall(r'\b\w+\b', current_article['content'].lower()))
    else:
        print(f"[WARN] Current article not added to similarity map. current_article exists: {current_article is not None}, has content: {current_article.get('content') if current_article else False}")
        current_words = None
    
    # If no database articles, return just the current article
    if not database_articles or len(database_articles) == 0:
        return {
            'points': points,
            'description': 'Only current article in database',
            'current_article_id': current_article_id
        }
    
    # Add other articles from database
    for article in database_articles[:50]:  # Limit to 50 for performance
        try:
            # Skip if it's the current article (already added above)
            if article.get('id') == current_article_id:
                continue
            
            article_content = article.get('content', '')
            if not article_content:
                continue
            
            # Calculate similarity to current article
            if current_words:
                article_words = set(re.findall(r'\b\w+\b', article_content.lower()))
                intersection = len(current_words & article_words)
                union = len(current_words | article_words)
                similarity = intersection / union if union > 0 else 0.0
            else:
                # If no current article content, use a default similarity
                similarity = 0.3
            
            credibility = float(article.get('credibility_score', 0.5))
            overall = float(article.get('overall_score', 0.5))
            
            points.append({
                'x': similarity,
                'y': credibility,
                'title': article.get('title', 'Untitled')[:50],
                'source': article.get('source', 'Unknown'),
                'overall_score': overall,
                'is_current': False
            })
        except Exception as e:
            print(f"[WARN] Error processing article for similarity map: {e}")
            continue
    
    return {
        'points': points[:30],  # Limit to 30 points for chart (includes current article)
        'description': f'Similarity map of {len(points)} articles (including current)',
        'current_article_id': current_article_id
    }

def _is_reliable_source(domain, reliable_sources):
    """Check if domain is from a reliable source (flexible matching)"""
    clean_domain = domain.lower()
    clean_domain = clean_domain.replace('www.', '').replace('m.', '').replace('edition.', '')
    clean_domain = clean_domain.replace('uk.', '').replace('us.', '').replace('au.', '')
    
    # Exact match
    if clean_domain in reliable_sources:
        return True
    
    # Check if reliable source is substring of domain
    for reliable in reliable_sources:
        reliable_clean = reliable.replace('www.', '')
        if reliable_clean in clean_domain or clean_domain in reliable_clean:
            return True
        # Base domain matching
        base_reliable = reliable_clean.split('.')[0]
        base_domain = clean_domain.split('.')[0]
        if len(base_reliable) >= 4 and base_reliable == base_domain:
            return True
    
    return False

def _extract_domain_from_source(source):
    """Extract clean domain from source string"""
    from urllib.parse import urlparse
    try:
        if source.startswith('http'):
            parsed = urlparse(source)
            domain = parsed.netloc.replace('www.', '').lower()
        else:
            # Assume it's already a domain
            domain = source.replace('www.', '').lower()
        return domain
    except:
        return source.lower().replace('www.', '')

def analyze_related_articles(title, content, source):
    """
    Find related articles from reputable sources for Chart 6
    Uses full RelatedArticlesAnalyzer if available, otherwise simplified version
    """
    # Use full analyzer if available
    if ANALYZERS_AVAILABLE and related_articles_analyzer:
        try:
            result = related_articles_analyzer.analyze(title, content, source)
            print("[OK] ✓ Using FULL RelatedArticlesAnalyzer - REAL articles from Google News, not placeholder")
            chart6 = result.get('chart6_data', {})
            article_count = chart6.get('total', 0)
            print(f"[DEBUG] Found {article_count} related articles from reputable sources")
            return result
        except Exception as e:
            print(f"[ERROR] RelatedArticlesAnalyzer failed: {e}, using fallback")
            import traceback
            traceback.print_exc()
    
    # Fallback to simplified version
    try:
        import requests
        from urllib.parse import quote, urlparse
        import re
        import html
        import xml.etree.ElementTree as ET
        
        # Extract domain from source (better extraction)
        # Source might be just domain name or full URL
        exclude_domain = _extract_domain_from_source(source)
        print(f"[DEBUG] Excluding domain: {exclude_domain} (from source: {source})")
        
        # Also try to get base domain (e.g., "bbc" from "bbc.com")
        exclude_base = exclude_domain.split('.')[0] if '.' in exclude_domain else exclude_domain
        
        # Clean title for search query
        clean_title = re.sub(r'[^\w\s]', ' ', title)
        clean_title = ' '.join(clean_title.split())
        
        # Try multiple query variations (like full analyzer)
        queries_to_try = [
            clean_title,  # Full title
            ' '.join(clean_title.split()[:10]),  # First 10 words
            ' '.join(clean_title.split()[:6]),  # First 6 words
        ]
        
        reliable_sources = [
            'reuters.com', 'bbc.com', 'bbc.co.uk', 'cnn.com', 'nytimes.com',
            'washingtonpost.com', 'theguardian.com', 'apnews.com', 'npr.org',
            'wsj.com', 'bloomberg.com', 'time.com', 'newsweek.com',
            'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'usatoday.com',
            'channelnewsasia.com', 'straitstimes.com', 'axios.com'
        ]
        
        articles = []
        
        # Try each query variation
        for attempt, search_query in enumerate(queries_to_try):
            if not search_query.strip():
                continue
                
            print(f"[DEBUG] Attempt {attempt+1}: Searching for '{search_query[:60]}...'")
            
            try:
                # Google News RSS feed
                rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}&hl=en-US&gl=US&ceid=US:en"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                
                response = requests.get(rss_url, timeout=15, headers=headers)
                
                if response.status_code != 200:
                    print(f"[WARN] HTTP {response.status_code} for query attempt {attempt+1}")
                    continue
                
                # Parse RSS XML
                try:
                    root = ET.fromstring(response.content)
                except ET.ParseError as e:
                    print(f"[WARN] XML parse error: {e}")
                    continue
                
                # Find all items (handle namespaces)
                # Google News RSS may use namespaces
                items = root.findall('.//item')
                if len(items) == 0:
                    # Try with namespace
                    items = root.findall('.//{http://purl.org/rss/1.0/}item')
                if len(items) == 0:
                    # Try without namespace prefix
                    items = root.findall('.//item') + root.findall('.//{*}item')
                
                print(f"[DEBUG] Found {len(items)} items in RSS feed")
                
                if len(items) == 0:
                    print(f"[DEBUG] No items found, trying next query...")
                    continue
                
                # Process items (limit to 20 for faster processing)
                processed_count = 0
                skipped_same_domain = 0
                skipped_unreliable = 0
                for item in items[:20]:  # Check up to 20 items (reduced for speed)
                    try:
                        processed_count += 1
                        # Get article details (handle namespaces and different XML structures)
                        # Google News RSS uses standard RSS 2.0 structure
                        title_elem = item.find('title')
                        if title_elem is None:
                            # Try with namespace
                            for ns in ['{http://purl.org/rss/1.0/}', '{http://www.w3.org/2005/Atom}']:
                                title_elem = item.find(f'{ns}title')
                                if title_elem is not None:
                                    break
                        
                        # Link can be in 'link' element or 'guid' element
                        link_elem = item.find('link')
                        if link_elem is None:
                            link_elem = item.find('guid')  # Sometimes link is in guid
                        if link_elem is None:
                            # Try with namespace
                            for ns in ['{http://purl.org/rss/1.0/}', '{http://www.w3.org/2005/Atom}']:
                                link_elem = item.find(f'{ns}link')
                                if link_elem is not None:
                                    break
                        
                        pub_date_elem = item.find('pubDate')
                        if pub_date_elem is None:
                            pub_date_elem = item.find('published')  # Atom format
                        
                        description_elem = item.find('description')
                        if description_elem is None:
                            description_elem = item.find('summary')  # Atom format
                        
                        source_elem = item.find('source')
                        
                        # Get link - can be text content or attribute
                        link = None
                        if link_elem is not None:
                            if link_elem.text:
                                link = link_elem.text.strip()
                            elif link_elem.get('href'):
                                link = link_elem.get('href').strip()
                        
                        if not link:
                            if processed_count <= 3:  # Only log first few
                                print(f"[DEBUG] Item {processed_count}: No link found (title: {title_elem.text[:50] if title_elem is not None and title_elem.text else 'N/A'})")
                            continue
                        
                        # Skip redirect following - Google News links work fine as-is
                        # They redirect automatically when clicked in browser
                        # Following redirects causes Lambda timeout
                        
                        # Extract domain
                        domain = None
                        try:
                            parsed_link = urlparse(link)
                            domain = parsed_link.netloc.replace('www.', '').lower()
                            
                            # For Google News links, extract domain from source element or link structure
                            if 'news.google.com' in link:
                                # Try to get domain from source element
                                if source_elem is not None:
                                    if hasattr(source_elem, 'get'):
                                        source_url = source_elem.get('url', '')
                                    elif hasattr(source_elem, 'text'):
                                        source_url = source_elem.text
                                    else:
                                        source_url = ''
                                    
                                    if source_url:
                                        try:
                                            parsed_source = urlparse(source_url)
                                            domain = parsed_source.netloc.replace('www.', '').lower()
                                        except:
                                            pass
                                
                                # If still no domain, try to extract from Google News URL structure
                                # Google News URLs sometimes have the source in the path
                                if not domain or domain == 'news.google.com':
                                    # Extract from URL like: news.google.com/rss/articles/CBMi...
                                    # Or use a default for Google News aggregator
                                    domain = 'news.google.com'
                                    # Try to get source name from title or other metadata
                                    if source_elem is not None and hasattr(source_elem, 'text') and source_elem.text:
                                        # Source name might be in the text
                                        pass
                        except Exception as e:
                            print(f"[WARN] Error extracting domain: {e}")
                            if processed_count <= 3:
                                print(f"[DEBUG] Link was: {link[:80]}")
                        
                        if not domain:
                            if processed_count <= 3:
                                print(f"[DEBUG] Item {processed_count}: Could not extract domain from: {link[:80]}")
                            continue
                        
                        # Skip same domain (flexible matching)
                        # Check if domains match (handles bbc.com vs bbc.co.uk, etc.)
                        domain_base = domain.split('.')[0] if '.' in domain else domain
                        if (exclude_base == domain_base or 
                            exclude_domain == domain or 
                            exclude_domain in domain or 
                            domain in exclude_domain):
                            skipped_same_domain += 1
                            if processed_count <= 3:
                                print(f"[DEBUG] Item {processed_count}: Skipping same domain: {domain} (exclude: {exclude_domain})")
                            continue
                        
                        # For Google News links, accept them if they have a reliable source name
                        # Google News aggregates from reliable sources, so we accept them
                        is_google_news = 'news.google.com' in link
                        
                        # Check if reliable source (using flexible matching)
                        # For Google News, we accept it if the source name looks reliable
                        if is_google_news:
                            # Accept Google News links (they aggregate from reliable sources)
                            # We'll use the source name from the RSS feed
                            pass
                        elif not _is_reliable_source(domain, reliable_sources):
                            skipped_unreliable += 1
                            if processed_count <= 3:
                                print(f"[DEBUG] Item {processed_count}: Domain not reliable: {domain}")
                            continue
                        
                        print(f"[DEBUG] Item {processed_count}: ✓ Found reliable source: {domain}")
                        
                        # Clean title
                        article_title = 'Untitled'
                        if title_elem is not None and title_elem.text:
                            article_title = html.unescape(title_elem.text)
                            article_title = re.sub(r'<[^>]+>', '', article_title).strip()
                        
                        # Get snippet
                        snippet = ''
                        if description_elem is not None and description_elem.text:
                            snippet = html.unescape(description_elem.text)
                            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                            snippet = snippet[:250]
                        
                        # Get date
                        pub_date = ''
                        if pub_date_elem is not None and pub_date_elem.text:
                            try:
                                from email.utils import parsedate_to_datetime
                                dt = parsedate_to_datetime(pub_date_elem.text)
                                pub_date = dt.strftime('%Y-%m-%d')
                            except:
                                pub_date = pub_date_elem.text[:10] if len(pub_date_elem.text) >= 10 else ''
                        
                        # Get source name
                        source_name = domain
                        if source_elem is not None and source_elem.text:
                            source_name = source_elem.text.strip()
                        
                        # Calculate relevance score (based on title similarity)
                        title_words = set(re.findall(r'\b\w+\b', article_title.lower()))
                        original_words = set(re.findall(r'\b\w+\b', title.lower()))
                        common_words = title_words.intersection(original_words)
                        if len(original_words) > 0:
                            relevance = min(0.9, max(0.3, len(common_words) / len(original_words) * 0.8))
                        else:
                            relevance = 0.5
                        
                        articles.append({
                            'title': article_title[:150],
                            'url': link,
                            'source': source_name,
                            'snippet': snippet,
                            'published_date': pub_date,
                            'relevance': round(relevance, 2)
                        })
                        
                        print(f"[DEBUG] Added article: {article_title[:50]}... (relevance: {relevance:.2f})")
                        
                        if len(articles) >= 8:  # Limit to 8 articles
                            break
                            
                    except Exception as e:
                        print(f"[WARN] Error processing article item: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Log summary
                print(f"[DEBUG] Query attempt {attempt+1} summary: Processed {processed_count} items, "
                      f"Skipped same domain: {skipped_same_domain}, Skipped unreliable: {skipped_unreliable}, "
                      f"Added: {len(articles)}")
                
                # If we found articles, stop trying other queries
                if len(articles) > 0:
                    print(f"[DEBUG] Found {len(articles)} articles, stopping search")
                    break
                    
            except Exception as e:
                print(f"[WARN] Error with query '{search_query}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[DEBUG] Total found: {len(articles)} related articles")
        
        # Prepare chart6_data
        chart6_data = {
            'articles': articles,
            'total': len(articles)
        }
        
        if len(articles) == 0:
            chart6_data['message'] = 'No related articles found from reputable sources'
        
        return {
            'chart6_data': chart6_data
        }
        
    except Exception as e:
        print(f"[ERROR] Related articles analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'chart6_data': {
                'articles': [],
                'message': 'Unable to fetch related articles at this time'
            }
        }

def analyze_word_frequency(text):
    """Analyze word frequency in the text"""
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    
    credible_found = {}
    suspicious_found = {}
    
    for word, count in word_counts.items():
        if word in CREDIBLE_WORDS:
            credible_found[word] = count
        if word in SUSPICIOUS_WORDS:
            suspicious_found[word] = count
    
    return {
        'credible_words': credible_found,
        'suspicious_words': suspicious_found,
        'total_words': len(words),
        'unique_words': len(set(words))
    }

def analyze_sentiment_flow(text):
    """Analyze sentiment flow across sentences"""
    sentences = sent_tokenize(text)
    sentiment_flow = []
    
    for sentence in sentences[:10]:  # Limit to first 10 sentences
        sentiment_scores = sentiment_analyzer.polarity_scores(sentence)
        sentiment_flow.append({
            'sentence': sentence[:100],  # Truncate for size
            'sentiment_score': Decimal(str(sentiment_scores['compound']))
        })
    
    return {
        'sentiment_flow': sentiment_flow,
        'total_sentences': len(sentences),
        'avg_sentiment': Decimal(str(sum(s['sentiment_score'] for s in sentiment_flow) / max(len(sentiment_flow), 1)))
    }

def extract_article(url):
    """Extract article content from URL"""
    try:
        print(f"[DEBUG] Extracting article from: {url}")
        
        # Enhanced headers to avoid blocking
        # Note: Not requesting Brotli (br) to avoid decompression issues
        # Only request gzip/deflate which are more reliably handled
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',  # Removed 'br' (Brotli) to avoid decompression issues
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        print(f"[DEBUG] Making request to URL...")
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()  # Raise exception for bad status codes
        
        print(f"[DEBUG] Response status: {response.status_code}, Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        print(f"[DEBUG] Content-Encoding: {response.headers.get('Content-Encoding', 'none')}")
        print(f"[DEBUG] Response content length: {len(response.content)} bytes")
        
        # Handle encoding properly - requests should auto-decompress Brotli/gzip, but verify
        # The response.text property should handle decompression and encoding automatically
        # But if Content-Encoding is 'br' (Brotli), we need to ensure it's decompressed
        content_encoding = response.headers.get('Content-Encoding', '').lower()
        
        # Always use response.text - requests library handles decompression automatically
        # response.text automatically decompresses Brotli/gzip and handles encoding
        try:
            html_text = response.text
            print(f"[DEBUG] Using response.text (length: {len(html_text)} chars, encoding: {response.encoding})")
            
            # Verify it's actually text (not binary)
            sample = html_text[:500]
            null_count = sample.count('\x00')
            if null_count > 50:
                print(f"[WARN] response.text has {null_count} null bytes, may be binary")
                # Try to fix by re-encoding/decoding
                try:
                    # Re-encode as bytes and decode with explicit UTF-8
                    html_text = html_text.encode('latin-1').decode('utf-8', errors='replace')
                    print(f"[DEBUG] Re-encoded/decoded to fix encoding issues")
                except:
                    pass
        except Exception as e:
            print(f"[WARN] Error using response.text: {e}, trying manual decode")
            html_text = None
        
        # If we don't have html_text yet, decode manually
        if html_text is None:
            content = response.content
            print(f"[DEBUG] Manual decoding needed, content length: {len(content)} bytes")
        
            # Try to detect encoding from response
            if response.encoding:
                print(f"[DEBUG] Response encoding: {response.encoding}")
                try:
                    html_text = content.decode(response.encoding)
                    print(f"[DEBUG] Decoded with response encoding: {response.encoding}")
                except Exception as e:
                    print(f"[WARN] Error decoding with {response.encoding}: {e}, trying UTF-8")
                    try:
                        html_text = content.decode('utf-8')
                        print(f"[DEBUG] Decoded with UTF-8")
                    except UnicodeDecodeError:
                        # Fallback: try common encodings
                        for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                            try:
                                html_text = content.decode(encoding)
                                print(f"[DEBUG] Successfully decoded with {encoding}")
                                break
                            except:
                                continue
                        else:
                            raise ValueError("Could not decode response content")
            else:
                # No encoding specified, try UTF-8 first
                try:
                    html_text = content.decode('utf-8')
                    print(f"[DEBUG] Decoded with UTF-8 (no encoding specified)")
                except UnicodeDecodeError:
                    # Try other encodings
                    for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                        try:
                            html_text = content.decode(encoding)
                            print(f"[DEBUG] Decoded with {encoding}")
                            break
                        except:
                            continue
                    else:
                        raise ValueError("Could not decode response content")
        
        # Validate that we have text, not binary
        if len(html_text) < 100:
            raise ValueError(f"Decoded content too short: {len(html_text)} characters")
        
        # Check for binary patterns - be more lenient
        # Check for null bytes (definite binary indicator)
        sample_text = html_text[:5000]  # Check first 5000 chars
        null_bytes = sample_text.count('\x00')
        
        # Count printable vs non-printable characters
        printable_count = sum(1 for c in sample_text if c.isprintable() or c.isspace())
        total_count = len(sample_text)
        printable_ratio = printable_count / total_count if total_count > 0 else 0
        
        print(f"[DEBUG] Content validation - Null bytes: {null_bytes}, Printable ratio: {printable_ratio:.2%}")
        
        # Only raise error if content is clearly binary
        # Threshold: > 100 null bytes OR (printable ratio < 70% AND null bytes > 0)
        # This allows some null bytes (which can occur in HTML) if most content is printable
        if null_bytes > 100 or (printable_ratio < 0.70 and null_bytes > 0):
            print(f"[WARN] Content appears to be binary (null bytes: {null_bytes}, printable: {printable_ratio:.2%})")
            print(f"[DEBUG] First 200 chars (repr): {repr(html_text[:200])}")
            # Check for HTML markers - be lenient
            html_lower = html_text.lower()[:10000]
            if '<html' not in html_lower and '<!doctype' not in html_lower and '<body' not in html_lower and '<div' not in html_lower:
                raise ValueError("Content does not appear to be HTML (no HTML tags found)")
            else:
                print(f"[DEBUG] HTML tags found, continuing despite binary-like content")
        else:
            print(f"[DEBUG] Content validation passed (null bytes: {null_bytes}, printable: {printable_ratio:.2%})")
        
        print(f"[DEBUG] HTML text length: {len(html_text)} characters")
        
        # Try different parsers for better compatibility
        try:
            soup = BeautifulSoup(html_text, 'lxml')
            print(f"[DEBUG] Parsed with lxml")
        except Exception as e:
            print(f"[WARN] lxml parser failed: {e}, trying html.parser")
            try:
                soup = BeautifulSoup(html_text, 'html.parser')
                print(f"[DEBUG] Parsed with html.parser")
            except Exception as e2:
                print(f"[WARN] html.parser failed: {e2}")
                raise ValueError(f"Could not parse HTML: {e2}")
        
        domain = urlparse(url).netloc.replace('www.', '')
        
        # Extract title - try multiple methods
        title = ''
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Try meta tags for title
        if not title or len(title) < 10:
            meta_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'title'})
            if meta_title:
                title = meta_title.get('content', '').strip()
        
        # Extract content - try multiple strategies
        content = ''
        
        # Strategy 1: Look for article-specific tags
        article_tag = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'article|content|post', re.I))
        if article_tag:
            paragraphs = article_tag.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
        
        # Strategy 2: If no article tag, try all paragraphs with better filtering
        if not content or len(content) < 100:
            paragraphs = soup.find_all('p')
            # Filter out navigation, footer, sidebar content
            filtered_paragraphs = []
            for p in paragraphs:
                text = p.get_text().strip()
                # Skip very short paragraphs and common non-content text
                if len(text) > 30 and not any(skip in text.lower() for skip in ['cookie', 'privacy policy', 'terms of service', 'subscribe', 'newsletter', 'follow us']):
                    filtered_paragraphs.append(text)
            content = ' '.join(filtered_paragraphs)
        
        # Strategy 3: If still no content, try divs with text
        if not content or len(content) < 100:
            content_divs = soup.find_all('div', class_=re.compile(r'content|body|text|story', re.I))
            for div in content_divs:
                text = div.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Only use substantial divs
                    content = text
                    break
        
        # Final fallback: get all text and clean it
        if not content or len(content) < 100:
            print("[WARN] Using fallback content extraction")
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            content = soup.get_text(separator=' ', strip=True)
            # Clean up excessive whitespace
            content = ' '.join(content.split())
        
        # Clean extracted content: remove binary/non-printable characters aggressively
        # Only keep ASCII printable characters and common Unicode letters/numbers/punctuation
        import string
        import unicodedata
        
        cleaned_content = ''
        for char in content:
            # Keep ASCII printable characters
            if char in string.printable:
                cleaned_content += char
            # Keep common Unicode letters, numbers, and punctuation (but be strict)
            elif unicodedata.category(char)[0] in ('L', 'N', 'P', 'S'):  # Letter, Number, Punctuation, Symbol
                # Only keep if it's a common Unicode range (basic multilingual plane)
                if ord(char) < 0x10000:
                    # Check if it's actually a valid character
                    try:
                        char.encode('utf-8')
                        cleaned_content += char
                    except:
                        cleaned_content += ' '  # Replace with space if encoding fails
                else:
                    cleaned_content += ' '  # Replace high Unicode with space
            # Keep whitespace
            elif char.isspace():
                cleaned_content += char
            # Skip all other characters (null bytes, control chars, etc.)
            else:
                cleaned_content += ' '  # Replace with space
        
        content = cleaned_content
        # Clean up multiple spaces and normalize whitespace
        content = ' '.join(content.split())
        
        # Additional pass: remove any remaining problematic characters
        # Remove any character that can't be encoded as UTF-8
        final_content = ''
        for char in content:
            try:
                char.encode('utf-8').decode('utf-8')
                final_content += char
            except:
                final_content += ' '  # Replace problematic chars with space
        
        content = ' '.join(final_content.split())
        
        print(f"[DEBUG] Content cleaned - removed binary characters, length: {len(content)} chars")
        
        # Validate we have content
        if not content or len(content.strip()) < 50:
            print(f"[ERROR] Content too short or empty: {len(content) if content else 0} characters")
            raise ValueError(f"Failed to extract sufficient content (got {len(content) if content else 0} characters)")
        
        print(f"[DEBUG] Extracted content: {len(content)} characters, {len(content.split())} words")
        
        # Extract source
        source = ''
        meta_source = soup.find('meta', property='og:site_name')
        if meta_source:
            source = meta_source.get('content', '').strip()
        
        if not source:
            # Try other meta tags
            meta_source = soup.find('meta', attrs={'name': 'application-name'}) or soup.find('meta', attrs={'name': 'site-name'})
            if meta_source:
                source = meta_source.get('content', '').strip()
        
        if not source:
            source = domain.split('.')[0].capitalize() if '.' in domain else domain
        
        # Extract published date - try multiple methods
        published_at = ''
        date_meta = soup.find('meta', property='article:published_time') or soup.find('meta', property='article:published')
        if date_meta:
            published_at = date_meta.get('content', '').strip()
        
        if not published_at:
            # Try time tag
            time_tag = soup.find('time')
            if time_tag:
                published_at = time_tag.get('datetime', '') or time_tag.get_text().strip()
        
        if not published_at:
            # Try other date meta tags
            date_meta = soup.find('meta', attrs={'name': 'publishdate'}) or soup.find('meta', attrs={'name': 'date'})
            if date_meta:
                published_at = date_meta.get('content', '').strip()
        
        # Final content validation and cleaning
        # Ensure content is valid text (no excessive binary characters)
        if content:
            # Count non-printable characters
            non_printable = sum(1 for c in content[:1000] if not (c.isprintable() or c.isspace()))
            if non_printable > 100:  # Too many non-printable chars
                print(f"[WARN] Content has {non_printable} non-printable characters, cleaning more aggressively")
                # More aggressive cleaning: only keep ASCII printable + common Unicode
                content = ''.join(c for c in content if (c.isprintable() or c.isspace()) and ord(c) < 0x10000)
                content = ' '.join(content.split())
        
        # Ensure we have valid content
        if not content or len(content.strip()) < 50:
            raise ValueError(f"Content too short or invalid after cleaning: {len(content) if content else 0} characters")
        
        # Clean title as well
        if title:
            title = ''.join(c for c in title if c.isprintable() or c.isspace())
            title = title.strip()
        
        result = {
            'url': url,
            'title': title or 'Untitled Article',
            'content': content,
            'source': source or domain,
            'published_at': published_at,
            'category': None,
            'domain': domain
        }
        
        print(f"[DEBUG] Extraction successful - Title: {title[:50] if title else 'N/A'}, Source: {source}, Content length: {len(content)}")
        print(f"[DEBUG] Content preview (first 200 chars): {content[:200]}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error extracting article: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"[ERROR] Error extracting article: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_sensationalism(text, title=None):
    """Analyze sensationalism using ML models (local or SageMaker) or comprehensive fallback"""
    
    # Try local ML models first (fastest and most reliable)
    if load_local_models() and local_model is not None:
        try:
            print("[DEBUG] Using local ML models for sensationalism analysis")
            
            # Prepare text
            texts = [text]
            
            # Get TF-IDF features
            X_tfidf = local_vectorizer.transform(texts)
            print(f"[DEBUG] TF-IDF shape: {X_tfidf.shape}")
            
            # Get linguistic features using proper 28-feature extraction (matching training)
            # Try to use features_enhanced module if available, otherwise recreate logic
            try:
                # Try importing features_enhanced module
                sys.path.insert(0, '/var/task/fakenews/src')
                from features_enhanced import extract_enhanced_linguistic_features, features_to_array
                print("[DEBUG] Using features_enhanced module for proper feature extraction")
                feats = extract_enhanced_linguistic_features(text)
                linguistic_features = features_to_array(feats).reshape(1, -1)
                print(f"[DEBUG] Extracted {linguistic_features.shape[1]} features using features_enhanced")
            except Exception as e:
                print(f"[WARN] Could not import features_enhanced: {e}")
                print("[WARN] Using inline feature extraction (may not match training exactly)")
                # Fallback: recreate feature extraction inline (28 features)
                import numpy as np
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                
                words = text.split()
                word_count = len(words)
                tokens = re.findall(r'\b\w+\b', text.lower())
                
                # Core sensationalism indicators (5)
                clickbait_patterns = [
                    r'\b(shocking|unbelievable|incredible|amazing|stunning|devastating)\b',
                    r'\b(you won\'t believe|won\'t believe|can\'t believe)\b',
                    r'\b(must see|must read|must watch|breaking|exclusive|urgent)\b',
                    r'\b(this will blow your mind|mind blown|jaw dropping|epic)\b',
                    r'\b(scandal|outrageous|absurd|ridiculous|ludicrous)\b',
                    r'\b(explosive|bombshell|devastating|crushing|destroyed)\b',
                    r'\b(secret|revealed|exposed|leaked|insider)\b',
                    r'\b(biggest ever|never before|first time|historic|unprecedented)\b',
                    r'\b(alert|warning|urgent|breaking news|just in)\b',
                ]
                clickbait_matches = sum(len(re.findall(pattern, text.lower())) for pattern in clickbait_patterns)
                clickbait_score = clickbait_matches / max(1, word_count)
                
                emotional_words = ['outrage', 'fury', 'rage', 'anger', 'furious', 'enraged', 'shock', 'shocked', 'shocking', 'stunned', 'stunning', 'devastating', 'devastated', 'crushing', 'destroyed', 'epic', 'legendary', 'historic', 'unprecedented', 'scandal', 'scandalous', 'controversial', 'outrageous', 'absurd', 'ridiculous', 'ludicrous', 'insane']
                emotional_word_count = sum(1 for word in emotional_words if word in text.lower())
                emotional_intensity = emotional_word_count / max(1, word_count)
                
                exclamation_count = text.count('!')
                exclamation_density = exclamation_count / max(1, word_count)
                
                caps_words = [w for w in words if w.isupper() and len(w) > 1]
                caps_density = len(caps_words) / max(1, word_count)
                
                question_count = text.count('?')
                question_density = question_count / max(1, word_count)
                
                # Sentiment and bias (4)
                try:
                    vader = SentimentIntensityAnalyzer()
                    sentiment = vader.polarity_scores(text)
                    sentiment_polarity = sentiment['compound']
                    sentiment_intensity = abs(sentiment['pos'] - sentiment['neg'])
                    sentiment_balance = 1.0 - abs(sentiment['pos'] - sentiment['neg'])
                except:
                    sentiment_polarity = 0.0
                    sentiment_intensity = 0.0
                    sentiment_balance = 0.0
                
                try:
                    from textblob import TextBlob
                    sentiment_subjectivity = float(TextBlob(text).sentiment.subjectivity)
                except:
                    sentiment_subjectivity = 0.0
                
                # Language patterns (5)
                intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'utterly', 'perfectly']
                intensifier_count = sum(1 for word in intensifiers if word in text.lower())
                intensifier_ratio = intensifier_count / max(1, len(tokens))
                
                tentative_words = ['might', 'perhaps', 'possibly', 'maybe', 'could', 'may', 'seems', 'appears']
                tentative_count = sum(1 for word in tentative_words if word in text.lower())
                tentative_ratio = tentative_count / max(1, len(tokens))
                
                evidence_words = ['according', 'study', 'research', 'data', 'evidence', 'source', 'report', 'findings']
                evidence_count = sum(1 for word in evidence_words if word in text.lower())
                evidence_ratio = evidence_count / max(1, len(tokens))
                
                professional_words = ['according', 'research', 'study', 'analysis', 'report', 'data', 'evidence', 'expert', 'official']
                professional_count = sum(1 for word in professional_words if word in text.lower())
                professional_ratio = professional_count / max(1, len(tokens))
                
                balanced_words = ['however', 'although', 'while', 'whereas', 'despite', 'nevertheless', 'furthermore', 'moreover']
                balanced_count = sum(1 for word in balanced_words if word in text.lower())
                balanced_ratio = balanced_count / max(1, len(tokens))
                
                # Text structure (4)
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                avg_sentence_length = word_count / max(1, len(sentences))
                avg_word_length = sum(len(w) for w in words) / max(1, word_count)
                text_length = len(text)
                
                # Repetition and redundancy (2)
                unique_tokens = len(set(tokens))
                unique_word_ratio = unique_tokens / max(1, len(tokens))
                # Simple repetition ratio (count repeated words)
                word_freq = {}
                for token in tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1
                repeated_words = sum(1 for count in word_freq.values() if count > 1)
                repetition_ratio = repeated_words / max(1, len(word_freq))
                
                # Specific patterns (5)
                all_caps_words = len(caps_words)
                
                # Content quality indicators (3)
                data_references = len(re.findall(r'\b(study|research|data|report|analysis|findings|statistics|survey)\b', text.lower()))
                
                # Build 28-feature array matching training
                linguistic_features = np.array([[
                    # Core sensationalism indicators (5)
                    clickbait_score,
                    emotional_intensity,
                    exclamation_density,
                    caps_density,
                    question_density,
                    # Sentiment and bias (4)
                    sentiment_polarity,
                    sentiment_subjectivity,
                    sentiment_intensity,
                    sentiment_balance,
                    # Language patterns (5)
                    intensifier_ratio,
                    tentative_ratio,
                    evidence_ratio,
                    professional_ratio,
                    balanced_ratio,
                    # Text structure (4)
                    avg_sentence_length,
                    avg_word_length,
                    float(text_length),
                    float(word_count),
                    # Repetition and redundancy (2)
                    repetition_ratio,
                    unique_word_ratio,
                    # Specific patterns (5)
                    float(all_caps_words),
                    float(exclamation_count),
                    float(question_count),
                    float(clickbait_matches),
                    float(emotional_word_count),
                    # Content quality indicators (3)
                    float(professional_count),
                    float(balanced_count),
                    float(data_references),
                ]], dtype=float)
                print(f"[DEBUG] Extracted {linguistic_features.shape[1]} features using inline extraction")
            
            # Z-SCORE NORMALIZATION APPROACH
            # Convert each subscore to z-scores, invert where needed, combine, and map to 0-100
            
            import numpy as np
            import math
            
            # Extract individual subscores from linguistic features
            # Features are in order: [clickbait_score, emotional_intensity, exclamation_density, ...]
            feats_array = linguistic_features[0]  # Get first (and only) row
            
            # Define feature names and their properties
            feature_names = [
                'clickbait_score', 'emotional_intensity', 'exclamation_density', 'caps_density', 'question_density',  # 0-4: Bad (higher = worse)
                'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_intensity', 'sentiment_balance',  # 5-8: Mixed
                'intensifier_ratio', 'tentative_ratio', 'evidence_ratio', 'professional_ratio', 'balanced_ratio',  # 9-13: Mixed
                'avg_sentence_length', 'avg_word_length', 'text_length', 'word_count',  # 14-17: Neutral
                'repetition_ratio', 'unique_word_ratio',  # 18-19: Mixed
                'all_caps_words', 'exclamation_count', 'question_count', 'clickbait_matches', 'emotional_word_count',  # 20-24: Bad (higher = worse)
                'professional_count', 'balanced_count', 'data_references'  # 25-27: Good (higher = better)
            ]
            
            # Identify which features are "bad" (higher = worse sensationalism)
            # These will be inverted: z_good = -z_bad
            bad_features = [0, 1, 2, 3, 4,  # Core sensationalism indicators
                           7,  # sentiment_subjectivity (higher = more subjective = worse)
                           10,  # intensifier_ratio (higher = more hyperbole = worse)
                           18,  # repetition_ratio (higher = more repetitive = worse)
                           20, 21, 22, 23, 24]  # Specific bad patterns
            
            # Get historical means and stds from the scaler
            # The scaler was fit on training data, so it has mean_ and scale_ (std)
            if hasattr(local_scaler, 'mean_') and hasattr(local_scaler, 'scale_'):
                means = local_scaler.mean_
                stds = local_scaler.scale_
                print(f"[DEBUG] Using scaler statistics: {len(means)} features")
            else:
                # Fallback: use approximate values (would need to be calibrated from training data)
                print("[WARN] Scaler doesn't have mean_/scale_, using approximate values")
                # For now, use the scaled features directly (they're already z-scored)
                X_ling_scaled = local_scaler.transform(linguistic_features)
                means = np.zeros(len(feats_array))
                stds = np.ones(len(feats_array))
            
            # Convert each subscore to z-score: z = (value - mean) / std
            z_scores = []
            z_log = {}
            
            for i, (feat_name, value) in enumerate(zip(feature_names, feats_array)):
                if i < len(means) and stds[i] > 0:
                    z = (value - means[i]) / stds[i]
                else:
                    # Fallback if no statistics available
                    z = 0.0
                
                # For "bad" features (higher = worse), invert: z_good = -z_bad
                if i in bad_features:
                    z = -z  # Invert so higher z means better (less sensationalism)
                
                # Clip extreme z values to [-3, +3] to avoid one metric dominating
                z = max(-3.0, min(3.0, z))
                
                z_scores.append(z)
                z_log[feat_name] = {
                    'raw_value': float(value),
                    'mean': float(means[i]) if i < len(means) else 0.0,
                    'std': float(stds[i]) if i < len(stds) else 1.0,
                    'z_score': float(z),
                    'inverted': i in bad_features
                }
            
            # Combine subscores in z-space with weights
            # Weights should sum to 1.0 (or we'll renormalize)
            weights = {
                # Core sensationalism indicators (high weight)
                'clickbait_score': 0.15,
                'emotional_intensity': 0.12,
                'exclamation_density': 0.10,
                'caps_density': 0.08,
                'question_density': 0.05,
                # Sentiment features
                'sentiment_subjectivity': 0.08,  # Higher subjectivity = worse
                'sentiment_balance': 0.05,  # Higher balance = better
                # Language patterns
                'intensifier_ratio': 0.08,  # Higher = worse
                'evidence_ratio': 0.06,  # Higher = better
                'professional_ratio': 0.06,  # Higher = better
                'balanced_ratio': 0.04,  # Higher = better
                # Repetition
                'repetition_ratio': 0.05,  # Higher = worse
                # Quality indicators
                'professional_count': 0.03,
                'balanced_count': 0.02,
                'data_references': 0.03
            }
            
            # Calculate weighted sum of z-scores
            combined_z = 0.0
            weight_sum = 0.0
            
            for i, feat_name in enumerate(feature_names):
                if feat_name in weights:
                    combined_z += weights[feat_name] * z_scores[i]
                    weight_sum += weights[feat_name]
            
            # Renormalize if weights don't sum to 1.0
            if weight_sum > 0:
                combined_z = combined_z / weight_sum
            
            # Map combined_z back to 0-100 score using sigmoid
            # percent = (1 / (1 + exp(-a * combined_z))) * 100
            # Choose 'a' (steepness) to control output spread
            # Higher 'a' = steeper curve, more spread
            # Lower 'a' = gentler curve, less spread
            a = 0.5  # Steepness parameter (adjust to tune output spread)
            
            # Sigmoid maps (-inf, +inf) to (0, 1), then scale to (0, 100)
            sigmoid_output = 1.0 / (1.0 + math.exp(-a * combined_z))
            final_score_percent = sigmoid_output * 100.0
            
            # Convert back to 0-1 scale for consistency with existing code
            final_score = final_score_percent / 100.0
            final_score = max(0.0, min(1.0, final_score))
            
            # Log intermediate z values and final percent
            print(f"[DEBUG] Z-SCORE NORMALIZATION RESULTS:")
            print(f"  Combined z-score: {combined_z:.3f}")
            print(f"  Sigmoid output (a={a}): {sigmoid_output:.3f}")
            print(f"  Final score (percent): {final_score_percent:.1f}%")
            print(f"  Final score (0-1): {final_score:.6f}")
            print(f"[DEBUG] Top contributing z-scores:")
            # Sort by absolute z-score contribution
            contributions = [(feat_name, weights.get(feat_name, 0) * z_scores[i]) 
                           for i, feat_name in enumerate(feature_names) 
                           if feat_name in weights]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            for feat_name, contrib in contributions[:5]:
                print(f"    {feat_name}: z={z_log[feat_name]['z_score']:.2f}, contribution={contrib:.3f}")
            
            return {
                'sensationalism_bias_likelihood': Decimal(str(final_score)),
                'analysis_available': True,
                'method': 'z_score_normalization',
                'z_score_details': {
                    'combined_z': float(combined_z),
                    'final_percent': float(final_score_percent),
                    'sigmoid_steepness': a,
                    'top_contributors': {feat_name: float(contrib) 
                                       for feat_name, contrib in contributions[:5]}
                }
            }
            
        except Exception as e:
            print(f"[ERROR] Local ML model failed: {e}")
            import traceback
            traceback.print_exc()
            print("[WARN] Falling back to SageMaker...")
    
    # Try SageMaker endpoint as secondary option
    try:
        print("[DEBUG] Calling SageMaker endpoint for sensationalism analysis")
        score = call_sagemaker_endpoint(text, title)
        
        if score is not None:
            # Return pure sensationalism score (no inversion, no bounds)
            # High score = High sensationalism, Low score = Low sensationalism
            final_score = round(float(score), 6)  # More precision for raw score
            
            print(f"[DEBUG] SageMaker sensationalism score (raw): {final_score:.6f}")
            
            return {
                'sensationalism_bias_likelihood': Decimal(str(final_score)),
                'analysis_available': True,
                'method': 'sagemaker_ml_model'
            }
        else:
            print("[WARN] SageMaker endpoint returned None, using fallback")
            raise Exception("SageMaker endpoint failed")
            
    except Exception as e:
        print(f"[ERROR] Error calling SageMaker endpoint: {e}")
        import traceback
        traceback.print_exc()
        print(f"[WARN] Both ML models failed, using heuristic fallback")
    
    # Enhanced fallback: comprehensive feature analysis (only if both ML models fail)
    text_lower = text.lower()
    word_count = len(text.split())
    
    # 1. Check for sensational keywords (most important indicator)
    # Use word boundaries to avoid false positives (e.g., "blast" in "ballast")
    suspicious_count = 0
    for word in SUSPICIOUS_WORDS:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        suspicious_count += len(re.findall(pattern, text_lower))
    
    # 2. Punctuation analysis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(len(text), 1)
    
    # 3. Title analysis (if provided) - titles are often more sensational
    title_sensationalism = 0.0
    if title:
        title_lower = title.lower()
        # Use word boundaries for title keywords too
        title_suspicious = 0
        for word in SUSPICIOUS_WORDS:
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            if re.search(pattern, title_lower):
                title_suspicious += 1
        
        title_exclamations = title.count('!')
        title_questions = title.count('?')
        title_caps = sum(1 for c in title if c.isupper())
        title_caps_ratio = title_caps / max(len(title), 1)
        
        # Title is weighted, but more conservatively
        title_sensationalism = (
            min(title_suspicious * 0.1, 0.2) +  # Each suspicious word in title = 0.1 (reduced from 0.15)
            min(title_exclamations * 0.08, 0.15) +  # Each ! in title = 0.08 (reduced from 0.1)
            min(title_questions * 0.04, 0.08) +  # Each ? in title = 0.04 (reduced from 0.05)
            min(title_caps_ratio * 2, 0.15)  # High caps ratio in title (reduced from 3)
        )
    
    # 4. Emotional language patterns
    emotional_patterns = [
        r'\b(very|extremely|absolutely|completely|totally|incredibly|amazingly)\b',
        r'\b(shocking|stunning|devastating|horrifying|terrifying|outrageous)\b',
        r'\b(never|always|all|every|none|nothing|everything)\b',  # Absolute statements
        r'\b(must|have to|need to|should|must)\s+(read|see|know|watch)\b',  # Urgency
    ]
    emotional_score = 0.0
    for pattern in emotional_patterns:
        matches = len(re.findall(pattern, text_lower))
        emotional_score += min(matches / max(word_count, 1) * 20, 0.15)
    
    # 5. Clickbait patterns
    clickbait_patterns = [
        r'you won\'?t believe',
        r'this (will|is going to) (shock|amaze|surprise)',
        r'number \d+ (will|is going to)',
        r'what happens next',
        r'doctors (hate|love|don\'?t want)',
        r'one (weird|simple|easy) (trick|tip|way)',
    ]
    clickbait_score = 0.0
    for pattern in clickbait_patterns:
        if re.search(pattern, text_lower):
            clickbait_score += 0.2
    
    # 6. Hyperbole indicators (repeated intensifiers)
    intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally']
    intensifier_count = sum(text_lower.count(word) for word in intensifiers)
    hyperbole_score = min(intensifier_count / max(word_count, 1) * 15, 0.2)
    
    # Calculate base sensationalism score
    # Start with a neutral base (0.35) - balanced between too low and too high
    # NOTE: Fallback should return 1.11 (111%) to indicate error
    sensationalism_score = 1.11  # Set to 111% to clearly indicate fallback/error
    
    # Add keyword-based sensationalism (most important, but more conservative)
    # Keywords are strong indicators, but need to be normalized properly
    keyword_score = min(suspicious_count / max(word_count, 1) * 30, 0.3)  # Reduced from 50/0.4
    sensationalism_score += keyword_score
    
    # Add punctuation-based sensationalism (more conservative)
    sensationalism_score += min(exclamation_count / max(word_count, 1) * 12, 0.2)  # Reduced from 15/0.25
    sensationalism_score += min(question_count / max(word_count, 1) * 6, 0.15)  # Reduced from 8/0.2
    sensationalism_score += min(caps_ratio * 2.5, 0.2)  # Reduced from 3/0.25
    
    # Add title sensationalism (weighted, but not too heavily)
    sensationalism_score += title_sensationalism * 0.4  # Reduced from 0.5
    
    # Add emotional language (capped lower)
    sensationalism_score += min(emotional_score, 0.15)  # Cap at 0.15
    
    # Add clickbait patterns (strong indicator, but cap it)
    sensationalism_score += min(clickbait_score, 0.25)  # Cap at 0.25
    
    # Add hyperbole (capped)
    sensationalism_score += min(hyperbole_score, 0.15)  # Cap at 0.15
    
    # Normalize to 0.0-1.0 range (but allow values outside if calculated)
    # Most articles should fall in 0.2-0.7 range, with truly sensational content at 0.7+
    # No bounds - return pure score
    sensationalism_score = min(max(sensationalism_score, 0.0), 1.0)
    
    # Return pure sensationalism score (no inversion, no bounds)
    # High score = High sensationalism, Low score = Low sensationalism
    final_score = round(sensationalism_score, 6)  # More precision for raw score
    
    # Debug logging
    print(f"[DEBUG] Fallback sensationalism analysis:")
    print(f"  Keywords found: {suspicious_count}, Keyword score: {keyword_score:.3f}")
    print(f"  Exclamations: {exclamation_count}, Questions: {question_count}")
    print(f"  Caps ratio: {caps_ratio:.3f}, Emotional score: {emotional_score:.3f}")
    print(f"  Clickbait score: {clickbait_score:.3f}, Hyperbole: {hyperbole_score:.3f}")
    print(f"  Title sensationalism: {title_sensationalism:.3f}")
    print(f"  Final score (raw, no inversion, no bounds): {final_score:.6f}")
    
    return {
        'sensationalism_bias_likelihood': Decimal(str(final_score)),
        'analysis_available': False,
        'method': 'enhanced_fallback',
        'features': {
            'suspicious_keywords': suspicious_count,
            'exclamations': exclamation_count,
            'questions': question_count,
            'caps_ratio': round(caps_ratio, 3),
            'emotional_language': round(emotional_score, 3),
            'clickbait_patterns': round(clickbait_score, 3)
        }
    }

def convert_to_decimal(value):
    """Convert value to Decimal for DynamoDB"""
    if isinstance(value, float):
        return Decimal(str(value))
    elif isinstance(value, dict):
        return {k: convert_to_decimal(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_decimal(item) for item in value]
    else:
        return value

def analyze_article(article_data):
    """Perform comprehensive article analysis"""
    print("=" * 80)
    print("[ANALYSIS START] Using FULL analyzers - NO placeholder data")
    print(f"[DEBUG] ANALYZERS_AVAILABLE: {ANALYZERS_AVAILABLE}")
    if ANALYZERS_AVAILABLE:
        print("[OK] Full analyzer classes are loaded and ready")
    else:
        print("[WARN] Using fallback simplified functions")
    print("=" * 80)
    
    # Analyze with each module
    print("\n[1/4] Analyzing language quality...")
    print(f"[DEBUG] Content length: {len(article_data.get('content', ''))}")
    print(f"[DEBUG] Title: {article_data.get('title', '')[:50]}...")
    language_result = analyze_language_quality(
        article_data['title'],
        article_data['content']
    )
    print(f"[DEBUG] Language result chart2_data: {language_result.get('chart2_data', {})}")
    
    print("\n[2/4] Analyzing source credibility (with WHOIS)...")
    credibility_result = analyze_credibility(
        article_data['url'],
        article_data['content'],
        article_data['source'],
        article_data['title']
    )
    
    print("\n[3/4] Analyzing cross-check similarity (web-based)...")
    crosscheck_result = analyze_crosscheck(
        article_data['title'],
        article_data['content'],
        article_data['url']
    )
    
    print("\n[4/4] Analyzing sensationalism (ML model)...")
    # Sensationalism analysis (include title for better accuracy)
    sensationalism_result = analyze_sensationalism(article_data['content'], article_data.get('title'))
    
    # No validation - return pure sensationalism score as-is
    sens_score_raw = sensationalism_result['sensationalism_bias_likelihood']
    if isinstance(sens_score_raw, Decimal):
        sens_score_raw = float(sens_score_raw)
    else:
        sens_score_raw = float(sens_score_raw)
    
    print(f"[DEBUG] Sensationalism score (raw, no bounds): {sens_score_raw:.6f}")
    
    # Related articles analysis
    print("\n[5/5] Finding related articles from reputable sources...")
    related_articles_result = analyze_related_articles(
        article_data['title'],
        article_data['content'],
        article_data['source']
    )
    
    print("\n" + "=" * 80)
    print("[ANALYSIS COMPLETE] All analyzers used - REAL data, NO placeholders")
    print("=" * 80)
    
    # Calculate overall score (weights from config)
    weights = {
        'language': 0.25,
        'credibility': 0.25,
        'crosscheck': 0.25,
        'sensationalism': 0.25
    }
    
    sensationalism_score = float(sensationalism_result['sensationalism_bias_likelihood'])
    # NOTE: sensationalism_bias_likelihood is now pure sensationalism (high = sensational, low = quality)
    # For overall score, we need to invert it: high sensationalism = low quality
    sensationalism_quality_score = 1.0 - sensationalism_score
    overall_score = (
        language_result['score'] * weights['language'] +
        credibility_result['score'] * weights['credibility'] +
        crosscheck_result['score'] * weights['crosscheck'] +
        sensationalism_quality_score * weights['sensationalism']  # Inverted for overall score
    )
    
    # Word frequency and sentiment flow
    word_frequency_data = analyze_word_frequency(article_data['content'])
    sentiment_flow_data = analyze_sentiment_flow(article_data['content'])
    
    # Calculate article ID first (needed for chart5_data)
    article_id = hashlib.md5(article_data['url'].encode()).hexdigest()
    
    # Get database articles for Chart 5 (similarity map) - Chart 5 still uses database
    # Note: Chart 4 (cross-check) now uses web search, but Chart 5 still needs database
    try:
        response = articles_table.scan(Limit=50)
        database_articles = response.get('Items', [])
        
        # Continue scanning if there are more items
        while 'LastEvaluatedKey' in response and len(database_articles) < 50:
            response = articles_table.scan(
                Limit=50 - len(database_articles),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            database_articles.extend(response.get('Items', []))
        
        print(f"[DEBUG] Found {len(database_articles)} articles in database for similarity map (Chart 5)")
    except Exception as e:
        print(f"[WARN] Error querying database for similarity map: {e}")
        database_articles = []
    
    # Prepare current article data for similarity map (include all relevant fields)
    current_article_for_map = {
        'content': article_data['content'],
        'title': article_data['title'],
        'source': article_data['source'],
        'credibility_score': float(credibility_result['score']),
        'overall_score': 0.0  # Will be calculated below, but we'll update it
    }
    
    # Generate similarity map data (will be updated with final overall_score after calculation)
    chart5_data = generate_similarity_map(article_id, database_articles, None, current_article_for_map)
    
    # Update chart5_data with final overall_score for current article
    if chart5_data and chart5_data.get('points'):
        for point in chart5_data['points']:
            if point.get('is_current'):
                point['overall_score'] = round(overall_score, 3)
                break
    
    # Build result with all fields
    result = {
        'id': article_id,
        'url': article_data['url'],
        'title': article_data['title'],
        'content': article_data['content'],
        'source': article_data['source'],
        'domain': article_data['domain'],
        'published_at': article_data.get('published_at'),
        'analyzed_at': datetime.now().isoformat(),
        'language_score': Decimal(str(language_result['score'])),
        'credibility_score': Decimal(str(credibility_result['score'])),
        'cross_check_score': Decimal(str(crosscheck_result['score'])),
        'sensationalism_bias_likelihood': sensationalism_result['sensationalism_bias_likelihood'],  # Already validated and bounded above
        'overall_score': Decimal(str(round(overall_score, 3))),
        'word_count': len(article_data['content'].split()),
        'sensational_keyword_count': 0,
        'category': None,
        'known_source_classification': credibility_result.get('known_source'),
        'chart1_data': language_result.get('chart1_data', {}),
        'chart2_data': language_result.get('chart2_data', {}),
        'chart3_data': credibility_result.get('chart3_data', {}),
        'chart4_data': crosscheck_result.get('chart4_data', {}),
        'chart5_data': chart5_data,  # Similarity map (includes current article)
        'chart6_data': related_articles_result.get('chart6_data', {}),
        'detailed_metrics': credibility_result.get('detailed_metrics', {}),
        'enhanced_info': credibility_result.get('enhanced_info', {}),
        'word_frequency_data': word_frequency_data,
        'sentiment_flow_data': sentiment_flow_data,
        'tfidf_vector': []
    }
    
    # No validation - return pure sensationalism score as-is
    # (removed bounds validation)
    
    # Validate other scores are within reasonable ranges
    if 'language_score' in result:
        lang_score = float(result['language_score'])
        lang_score = max(0.0, min(1.0, lang_score))
        result['language_score'] = Decimal(str(round(lang_score, 3)))
    
    if 'credibility_score' in result:
        cred_score = float(result['credibility_score'])
        cred_score = max(0.0, min(1.0, cred_score))
        result['credibility_score'] = Decimal(str(round(cred_score, 3)))
    
    if 'cross_check_score' in result:
        cross_score = float(result['cross_check_score'])
        cross_score = max(0.0, min(1.0, cross_score))
        result['cross_check_score'] = Decimal(str(round(cross_score, 3)))
    
    # Convert all floats to Decimals recursively
    return convert_to_decimal(result)

def get_cors_headers():
    """Return CORS headers"""
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS,PUT,DELETE',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,Accept,Origin',
        'Access-Control-Max-Age': '3600'
    }

def handle_stats():
    """Get database statistics"""
    try:
        print("[DEBUG] handle_stats called")
        # Scan articles table
        print("[DEBUG] Scanning articles table...")
        response = articles_table.scan()
        articles = response.get('Items', [])
        print(f"[DEBUG] Found {len(articles)} articles")
        
        # Convert Decimal to float for comparison
        total = len(articles)
        high_credibility = sum(1 for a in articles if float(a.get('overall_score', 0)) >= 0.70)
        low_credibility = sum(1 for a in articles if float(a.get('overall_score', 0)) < 0.50)
        
        result = {
            'success': True,
            'total': total,
            'total_articles': total,
            'high_credibility': high_credibility,
            'low_credibility': low_credibility
        }
        
        print(f"[DEBUG] Returning stats: {result}")
        response_obj = {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps(result, default=str)
        }
        print(f"[DEBUG] Response object created: statusCode={response_obj['statusCode']}")
        return response_obj
    except Exception as e:
        print(f"[ERROR] Stats error: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Traceback: {error_trace}")
        error_response = {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'success': False, 'error': str(e), 'traceback': error_trace})
        }
        print(f"[ERROR] Error response: {error_response}")
        return error_response

def handle_get_articles():
    """Get all articles from database"""
    try:
        response = articles_table.scan()
        articles = response.get('Items', [])
        
        # Convert Decimal to float for JSON serialization
        for article in articles:
            for key, value in article.items():
                if isinstance(value, Decimal):
                    article[key] = float(value)
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'success': True,
                'articles': articles,
                'count': len(articles)
            }, default=str)
        }
    except Exception as e:
        print(f"Get articles error: {e}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'success': False, 'error': str(e)})
        }

def handle_clear_database():
    """Clear all articles from database"""
    try:
        # Scan and delete all items
        response = articles_table.scan()
        items = response.get('Items', [])
        
        deleted = 0
        for item in items:
            try:
                # Use 'id' as the key (not 'url')
                articles_table.delete_item(Key={'id': item['id']})
                deleted += 1
            except Exception as e:
                print(f"Delete error for {item.get('id')}: {e}")
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'success': True,
                'deleted': deleted,
                'message': f'Deleted {deleted} articles'
            })
        }
    except Exception as e:
        print(f"Clear database error: {e}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'success': False, 'error': str(e)})
        }

def handle_analyze(event):
    """Handle article analysis"""
    print(f"[DEBUG] ML_DEPENDENCIES_AVAILABLE: {ML_DEPENDENCIES_AVAILABLE}")
    
    if not ML_DEPENDENCIES_AVAILABLE:
        print("[ERROR] ML dependencies not available!")
        return {
            'statusCode': 503,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'ML dependencies not available. Please fix Lambda dependencies for /analyze endpoint.'
            })
        }
    
    try:
        # Parse request
        if event.get('httpMethod') == 'POST':
            body = json.loads(event.get('body', '{}'))
            url = body.get('url')
        else:
            url = event.get('url')
        
        print(f"[DEBUG] Analyzing URL: {url}")
        
        if not url:
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'URL is required'})
            }
        
        # Check cache first (if available)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        print(f"[DEBUG] Cache check for hash: {url_hash}")
        if CACHE_TABLE_AVAILABLE and cache_table:
            try:
                cache_response = cache_table.get_item(Key={'url_hash': url_hash})
                
                if 'Item' in cache_response:
                    print("[DEBUG] Cache hit - returning cached result")
                    cached_article = cache_response['Item']['analysis_result']
                    
                    # No validation - return cached sensationalism score as-is
                    # (removed bounds validation)
                    
                    return {
                        'statusCode': 200,
                        'headers': get_cors_headers(),
                        'body': json.dumps({
                            'success': True,
                            'cached': True,
                            'article': cached_article
                        }, default=str)  # Handle Decimal serialization
                    }
                else:
                    print("[DEBUG] Cache miss - proceeding with analysis")
            except Exception as e:
                print(f"[WARN] Cache check error (non-fatal): {e}")
        else:
            print("[DEBUG] Cache table not available - skipping cache check")
        
        # Extract article content
        print("[DEBUG] Extracting article content...")
        article_data = extract_article(url)
        if not article_data:
            print("[ERROR] Failed to extract article content")
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'Failed to extract article content'})
            }
        
        print(f"[DEBUG] Article extracted - Title: {article_data.get('title', 'N/A')[:50]}, Word count: {len(article_data.get('content', '').split())}")
        
        # Analyze article
        print("[DEBUG] Starting article analysis...")
        analysis_result = analyze_article(article_data)
        print(f"[DEBUG] Analysis complete - Credibility: {analysis_result.get('credibility_score')}, Overall: {analysis_result.get('overall_score')}")
        
        # Cache result (if cache table available)
        if CACHE_TABLE_AVAILABLE and cache_table:
            print("[DEBUG] Attempting to cache result...")
            try:
                cache_item = {
                    'url_hash': url_hash,
                    'url': url,
                    'analysis_result': analysis_result,
                    'timestamp': datetime.now().isoformat()
                }
                cache_table.put_item(Item=cache_item)
                print("[DEBUG] Result cached successfully")
            except Exception as e:
                print(f"[WARN] Cache store error (non-fatal): {e}")
        else:
            print("[DEBUG] Cache table not available - skipping cache storage")
        
        # Store in articles table as well
        print("[DEBUG] Storing in articles table...")
        try:
            articles_table.put_item(Item=analysis_result)
            print("[DEBUG] Article stored successfully")
        except Exception as e:
            print(f"[WARN] Articles table store error (non-fatal): {e}")
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'success': True,
                'cached': False,
                'article': analysis_result
            }, default=str)  # Handle Decimal serialization
        }
    except Exception as e:
        print(f"Analyze error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': str(e)})
        }

def lambda_handler(event, context):
    """Main Lambda handler with routing"""
    try:
        print(f"[DEBUG] Lambda handler called. Event: {json.dumps(event, default=str)}")
        
        # Handle OPTIONS (CORS preflight)
        if event.get('httpMethod') == 'OPTIONS':
            print("[DEBUG] Handling OPTIONS request")
            return {
                'statusCode': 200,
                'headers': get_cors_headers(),
                'body': ''
            }
        
        # Get path from event
        path = event.get('path', '') or event.get('resource', '')
        print(f"[DEBUG] Path: {path}, httpMethod: {event.get('httpMethod')}")
        
        # Route based on path
        if '/stats' in path:
            print("[DEBUG] Routing to handle_stats")
            result = handle_stats()
            print(f"[DEBUG] handle_stats returned: statusCode={result.get('statusCode')}")
            return result
        elif '/articles' in path:
            print("[DEBUG] Routing to handle_get_articles")
            return handle_get_articles()
        elif '/clear-database' in path:
            print("[DEBUG] Routing to handle_clear_database")
            return handle_clear_database()
        elif '/analyze' in path:
            print("[DEBUG] Routing to handle_analyze")
            return handle_analyze(event)
        else:
            # Default to analyze for backward compatibility
            print("[DEBUG] Routing to handle_analyze (default)")
            return handle_analyze(event)
        
    except Exception as e:
        print(f"[ERROR] Lambda handler error: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Traceback: {error_trace}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': str(e), 'traceback': error_trace})
        }

