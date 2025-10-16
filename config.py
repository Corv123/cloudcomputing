# config.py
# Configuration for Fake News Detector

# Database configuration
DATABASE_NAME = 'fakenews.db'

# Score weights for overall calculation
SCORE_WEIGHTS = {
    'language': 0.25,
    'credibility': 0.50,
    'crosscheck': 0.25
}

# =============================================================================
# SHARED RELIABLE NEWS SOURCES (Used by credibility_analyzer & related_articles_analyzer)
# =============================================================================

RELIABLE_NEWS_SOURCES = [
    # ===== TIER 1: Wire Services & International News Agencies (Highest credibility) =====
    "reuters.com", "reuters.co.uk",
    "apnews.com", "ap.org",
    "afp.com",  # Agence France-Presse
    "dpa.com",  # Deutsche Presse-Agentur

    # ===== TIER 2: Major International Broadcasters =====
    "bbc.com", "bbc.co.uk",
    "npr.org", "pbs.org",
    "aljazeera.com",
    "dw.com",  # Deutsche Welle
    "france24.com",

    # ===== TIER 3: Major US Newspapers (National) =====
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",  # Wall Street Journal
    "usatoday.com",
    "latimes.com",
    "chicagotribune.com",
    "bostonglobe.com",
    "miamiherald.com",
    "seattletimes.com",
    "denverpost.com",
    "sfchronicle.com",
    "dallasnews.com",

    # ===== TIER 4: Major UK Newspapers =====
    "theguardian.com",
    "telegraph.co.uk",
    "independent.co.uk",
    "thetimes.co.uk",
    "ft.com",  # Financial Times
    "economist.com",
    "spectator.co.uk",
    "dailymail.co.uk",  # Mixed credibility but mainstream

    # ===== TIER 5: Major TV Networks (US) =====
    "cnn.com",
    "cbsnews.com",
    "nbcnews.com",
    "abcnews.go.com",
    "msnbc.com",
    "cnbc.com",
    "foxnews.com",

    # ===== TIER 6: Business & Finance News =====
    "bloomberg.com",
    "forbes.com",
    "fortune.com",
    "businessinsider.com",
    "marketwatch.com",
    "barrons.com",
    "investopedia.com",
    "fool.com",  # The Motley Fool

    # ===== TIER 7: Technology News =====
    "theverge.com",
    "arstechnica.com",
    "wired.com",
    "techcrunch.com",
    "cnet.com",
    "zdnet.com",
    "engadget.com",
    "techmeme.com",
    "geekwire.com",

    # ===== TIER 8: Asia-Pacific News =====
    "channelnewsasia.com",
    "straitstimes.com",
    "todayonline.sg",
    "businesstimes.com.sg",
    "scmp.com",  # South China Morning Post
    "japantimes.co.jp",
    "thestar.com.my",
    "bangkokpost.com",
    "jakartapost.com",
    "philstar.com",
    "smh.com.au",  # Sydney Morning Herald
    "theage.com.au",
    "abc.net.au",  # Australian Broadcasting Corporation
    "nzherald.co.nz",
    "stuff.co.nz",

    # ===== TIER 9: Canadian News =====
    "theglobeandmail.com",
    "cbc.ca",
    "thestar.com",  # Toronto Star
    "nationalpost.com",
    "globalnews.ca",

    # ===== TIER 10: European News =====
    "irishtimes.com",
    "rte.ie",  # Radio Telefís Éireann
    "thelocal.com",  # Multiple European countries
    "euronews.com",
    "politico.eu",

    # ===== TIER 11: Political & Policy News =====
    "politico.com",
    "axios.com",
    "thehill.com",
    "rollcall.com",
    "nationaljournal.com",
    "cookpolitical.com",

    # ===== TIER 12: Investigative Journalism & Nonprofits =====
    "propublica.org",
    "icij.org",  # International Consortium of Investigative Journalists
    "revealnews.org",
    "themarshallproject.org",
    "publicintegrity.org",
    "bellingcat.com",

    # ===== TIER 13: Fact-Checking Organizations =====
    "factcheck.org",
    "politifact.com",
    "snopes.com",
    "fullfact.org",
    "checkyourfact.com",

    # ===== TIER 14: National Magazines (US & International) =====
    "time.com",
    "newsweek.com",
    "theatlantic.com",
    "newyorker.com",
    "vanityfair.com",
    "nationalgeographic.com",
    "smithsonianmag.com",
    "foreignaffairs.com",
    "foreignpolicy.com",

    # ===== TIER 15: Opinion & Analysis (Reputable) =====
    "vox.com",
    "slate.com",
    "salon.com",
    "thedailybeast.com",
    "motherjones.com",
    "newrepublic.com",
    "reason.com",
    "nationalreview.com",

    # ===== TIER 16: Science & Health News =====
    "scientificamerican.com",
    "newscientist.com",
    "nature.com",
    "science.org",
    "sciencedaily.com",
    "livescience.com",
    "healthline.com",
    "webmd.com",
    "mayoclinic.org",

    # ===== TIER 17: Sports News =====
    "espn.com",
    "si.com",  # Sports Illustrated
    "bleacherreport.com",
    "sbnation.com",
    "cbssports.com",

    # ===== TIER 18: Entertainment & Culture =====
    "variety.com",
    "hollywoodreporter.com",
    "deadline.com",
    "rollingstone.com",
    "billboard.com",

    # ===== TIER 19: Local/Regional (Major US Cities) =====
    "nydailynews.com",
    "newsday.com",
    "sfgate.com",
    "chron.com",  # Houston Chronicle
    "ajc.com",  # Atlanta Journal-Constitution
    "startribune.com",  # Minneapolis Star Tribune
    "stltoday.com",  # St. Louis Post-Dispatch

    # ===== TIER 20: Government & Official Sources =====
    # .gov domains handled separately in scoring, but listing for completeness
    "whitehouse.gov",
    "congress.gov",
    "supremecourt.gov",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
]

# Known problematic sources (for credibility analyzer)
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
    "nationalreport.net": "Satirical fake news",
    "yournewswire.com": "Fake news",
    "newslo.com": "Fake news",
    "empirenews.net": "Fake news",
    "huzlers.com": "Fake news",
}

# Domain age tiers (for credibility scoring)
OLD_ESTABLISHED_DOMAINS = [
    'bbc.com', 'nytimes.com', 'washingtonpost.com',
    'reuters.com', 'npr.org', 'wsj.com', 'economist.com',
    'theguardian.com', 'ft.com', 'latimes.com'
]

MODERATE_AGE_DOMAINS = [
    'cnn.com', 'axios.com', 'politico.com', 'bloomberg.com',
    'businessinsider.com', 'theverge.com', 'vox.com'
]

print(f"✅ Loaded {len(RELIABLE_NEWS_SOURCES)} reliable news sources")
print(f"✅ Loaded {len(PROBLEMATIC_NEWS_SOURCES)} problematic sources")

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
        "enabled": True,  # Related Articles from Reputable Sources
        "title": "Related Articles from Reputable Sources",
        "type": "article_cards"
    }
}