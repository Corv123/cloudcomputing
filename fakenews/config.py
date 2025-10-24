# config.py
# Configuration for Fake News Detector

# Database
DATABASE_NAME = "articles.db"

# NewsAPI (optional - get free key from https://newsapi.org/)
NEWSAPI_KEY = ""

# Scoring Weights (must sum to 1.0)
# All four components weighted equally for balanced assessment
SCORE_WEIGHTS = {
    "language": 0.25,       # 25% - Language Quality
    "credibility": 0.25,    # 25% - Source Credibility
    "crosscheck": 0.25,     # 25% - Cross-Check Score
    "sensationalism": 0.25  # 25% - Sensationalism Score
}

# Credible Domains
CREDIBLE_DOMAINS = [
    "bbc.com", "reuters.com", "apnews.com", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "wsj.com", "economist.com", "ft.com", "bloomberg.com"
]

# Suspicious Domains
SUSPICIOUS_DOMAINS = [
    "beforeitsnews.com", "naturalnews.com", "infowars.com"
]

# WHOIS Configuration
ENABLE_WHOIS = False  # Set to True if you have WHOIS API key
WHOIS_API_URL = "https://www.whoisxmlapi.com/whoisserver/WhoisService"
WHOIS_API_KEY = ""  # Get free key from https://www.whoisxmlapi.com/
WHOIS_TIMEOUT_SECONDS = 5
WHOIS_CACHE_HOURS = 24

# Trusted Registrars
TRUSTED_REGISTRARS = [
    "GoDaddy.com, LLC", "Namecheap, Inc.", "Google LLC",
    "Amazon Registrar, Inc.", "Network Solutions, LLC"
]

# Enhanced Credibility Sources (for credibility_analyzer.py)
PROBLEMATIC_NEWS_SOURCES = {
    "naturalnews.com": "Health misinformation",
    "infowars.com": "Conspiracy theories", 
    "theonion.com": "Satirical content",
    "babylonbee.com": "Satirical content",
    "activistpost.com": "Conspiracy promotion",
    "bipartisanreport.com": "Misleading headlines",
    "beforeitsnews.com": "User-generated conspiracy content",
    "collective-evolution.com": "Pseudoscience content",
    "davidwolfe.com": "Health misinformation",
    "worldnewsdailyreport.com": "Satirical fake news",
    "nationalreport.net": "Satirical fake news"
}

RELIABLE_NEWS_SOURCES = [
    "reuters.com", "apnews.com", "bbc.com", "npr.org",
    "channelnewsasia.com", "straitstimes.com", "cnn.com",
    "washingtonpost.com", "nytimes.com", "wsj.com",
    "economist.com", "theguardian.com", "usatoday.com",
    "abcnews.go.com", "cbsnews.com", "nbcnews.com",
    "pbs.org", "time.com", "newsweek.com", "bloomberg.com",
    "politico.com", "axios.com", "propublica.org", "ft.com"
]

OLD_ESTABLISHED_DOMAINS = [
    'bbc.com', 'nytimes.com', 'washingtonpost.com', 
    'reuters.com', 'npr.org', 'wsj.com', 'economist.com'
]

MODERATE_AGE_DOMAINS = [
    'cnn.com', 'theguardian.com', 'axios.com', 
    'politico.com', 'bloomberg.com'
]

# Chart Configuration
CHART_SETTINGS = {
    "chart1": {
        "enabled": True,
        "title": "Language Quality Analysis",
        "type": "bar"
    },
    "chart2": {
        "enabled": True,
        "title": "Sentiment Distribution",
        "type": "pie"
    },
    "chart3": {
        "enabled": True,
        "title": "Credibility Radar",
        "type": "radar"
    },
    "chart4": {
        "enabled": True,
        "title": "Cross-Check Analysis",
        "type": "scatter"
    },
    "chart5": {
        "enabled": True,
        "title": "Similarity Map",
        "type": "scatter"
    },
    "chart6": {
        "enabled": True,
        "title": "Related Articles from Reputable Sources",
        "type": "related_articles"
    }
}