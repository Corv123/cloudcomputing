# analyzers/credibility_analyzer.py
# Analyzes source credibility using domain-based scoring (Chart 3 Radar)
# Integrated from team member's credibility scorer

from urllib.parse import urlparse
from typing import Dict, Any
import re

class CredibilityAnalyzer:
    """
    Analyzes source credibility based on domain characteristics.
    Generates radar chart data for Chart 3.
    """
    
    def __init__(self):
        # Known problematic sources (from team member's code)
        self.problematic_sources = {
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
        }
        
        # Known reliable sources (from team member's code + our config)
        self.reliable_sources = [
            "reuters.com", "apnews.com", "bbc.com", "npr.org",
            "channelnewsasia.com", "straitstimes.com", "cnn.com",
            "washingtonpost.com", "nytimes.com", "wsj.com",
            "economist.com", "theguardian.com", "usatoday.com",
            "abcnews.go.com", "cbsnews.com", "nbcnews.com",
            "pbs.org", "time.com", "newsweek.com", "bloomberg.com",
            "politico.com", "axios.com", "propublica.org", "ft.com"
        ]
        
        # Highly credible old domains
        self.old_domains = [
            'bbc.com', 'nytimes.com', 'washingtonpost.com', 
            'reuters.com', 'npr.org', 'wsj.com', 'economist.com'
        ]
        
        # Moderate age domains
        self.moderate_age_domains = [
            'cnn.com', 'theguardian.com', 'axios.com', 
            'politico.com', 'bloomberg.com'
        ]
    
    def analyze(self, url: str, content: str, source: str, title: str) -> Dict[str, Any]:
        """
        Analyze source credibility and return score + radar chart data
        
        Returns:
            dict with 'score' and 'chart3_data' (radar chart for 4 metrics)
        """
        domain = self._extract_domain(url)
        
        # Calculate 4 credibility dimensions (for radar chart)
        domain_age = self._score_domain_age(domain)
        url_structure = self._score_url_structure(domain, url)
        site_structure = self._score_site_structure(domain)
        content_format = self._score_content_format(domain, url, content, title)
        
        # Overall credibility score (weighted average)
        credibility_score = (
            domain_age * 0.25 +
            url_structure * 0.20 +
            site_structure * 0.30 +
            content_format * 0.25
        ) / 100  # Convert to 0-1 scale
        
        # Chart 3 Data: Radar chart
        chart3_data = {
            "labels": [
                "Domain Age",
                "URL Structure",
                "Site Structure", 
                "Content Format"
            ],
            "values": [
                round(domain_age, 1),
                round(url_structure, 1),
                round(site_structure, 1),
                round(content_format, 1)
            ]
        }
        
        return {
            "score": round(credibility_score, 3),
            "chart3_data": chart3_data,
            "metrics": {
                "domain_age": round(domain_age / 100, 3),
                "url_structure": round(url_structure / 100, 3),
                "site_structure": round(site_structure / 100, 3),
                "content_format": round(content_format / 100, 3)
            },
            "known_source": self._get_source_classification(domain)
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        try:
            parsed = urlparse(url if url.startswith('http') else f'https://{url}')
            domain = parsed.netloc.replace('www.', '').lower()
            return domain
        except:
            return url.lower().replace('www.', '')
    
    def _score_domain_age(self, domain: str) -> float:
        """
        Score based on domain age/reputation (0-100)
        From team member's implementation
        """
        if domain in self.old_domains:
            return 90
        if domain in self.moderate_age_domains:
            return 75
        if domain in self.reliable_sources:
            return 70
        if domain in self.problematic_sources:
            return 20
        
        # Default for unknown domains
        return 50
    
    def _score_url_structure(self, domain: str, url: str) -> float:
        """
        Score URL structure quality (0-100)
        From team member's implementation
        """
        score = 50  # Base score
        
        # Check TLD
        if domain.endswith('.edu'):
            score += 30
        elif domain.endswith('.gov'):
            score += 30
        elif domain.endswith('.org'):
            score += 20
        elif domain.endswith('.com'):
            score += 10
        elif re.search(r'\.(tk|ml|ga|cf)$', domain):
            score -= 30  # Suspicious TLDs
        
        # Check for news indicators
        if 'news' in domain or 'report' in domain or 'times' in domain:
            score += 5
        
        # Check for suspicious patterns
        if re.search(r'\d{4,}', domain):  # Random numbers
            score -= 15
        if len(domain.split('.')[0]) > 20:  # Very long domain
            score -= 10
        if domain.count('-') > 3:  # Too many hyphens
            score -= 10
        
        # URL path quality
        try:
            parsed = urlparse(url if url.startswith('http') else f'https://{url}')
            path = parsed.path
            if '/article/' in path or '/news/' in path:
                score += 5
            if path.count('/') > 6:  # Very deep paths
                score -= 5
        except:
            pass
        
        return max(0, min(100, score))
    
    def _score_site_structure(self, domain: str) -> float:
        """
        Score site structure quality (0-100)
        From team member's implementation
        """
        score = 50
        
        if domain in self.reliable_sources:
            score = 85
        elif domain in self.problematic_sources:
            score = 25
        else:
            # Estimate based on domain characteristics
            if any(word in domain for word in ['news', 'times', 'post', 'herald']):
                score += 15
            if len(domain) < 15:  # Short, memorable domains
                score += 10
            if re.search(r'\.(edu|gov|org)$', domain):
                score += 20
        
        return max(0, min(100, score))
    
    def _score_content_format(self, domain: str, url: str, content: str, title: str) -> float:
        """
        Score content format quality (0-100)
        From team member's implementation + content analysis
        """
        score = 50
        
        if domain in self.reliable_sources:
            score = 90
        elif domain in self.problematic_sources:
            if "Satirical" in self.problematic_sources.get(domain, ""):
                score = 60  # Satire is well-formatted but not news
            else:
                score = 30
        else:
            # URL pattern analysis
            if re.search(r'/\d{4}/\d{2}/\d{2}/', url):  # Date in URL
                score += 15
            if '/article' in url or '/story' in url:
                score += 10
            if len(url) > 150:  # Very long URLs
                score -= 10
            if sum(1 for c in url if c.isupper()) > len(url) * 0.3:  # Too many caps
                score -= 15
            
            # Content quality checks
            if content:
                word_count = len(content.split())
                if word_count > 300:
                    score += 10
                elif word_count < 100:
                    score -= 15
                
                # Check for professional formatting indicators
                if '\n\n' in content:  # Paragraphs
                    score += 5
            
            # Title quality
            if title:
                if len(title) > 100:  # Clickbait-length title
                    score -= 10
                if title.count('!') > 2:  # Too many exclamations
                    score -= 10
        
        return max(0, min(100, score))
    
    def _get_source_classification(self, domain: str) -> str:
        """Get classification of known sources"""
        if domain in self.problematic_sources:
            return f"Known Issue: {self.problematic_sources[domain]}"
        elif domain in self.reliable_sources:
            return "Verified Reliable Source"
        else:
            return None