# config.py
# Configuration for Fake News Detector

# Database
DATABASE_NAME = "articles.db"

# NewsAPI (optional - get free key from https://newsapi.org/)
NEWSAPI_KEY = ""

# Scoring Weights (must sum to 1.0)
# Chart 3 (Credibility) weighted most heavily based on team member's implementation
SCORE_WEIGHTS = {
    "language": 0.35,      # 25% - Charts 1 & 2
    "credibility": 0.35,   # 50% - Chart 3 (INCREASED - domain-based credibility)
    "crosscheck": 0.30     # 25% - Chart 4
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
        "enabled": False,  # Placeholder
        "title": "Additional Analysis",
        "type": "placeholder"
    }
}