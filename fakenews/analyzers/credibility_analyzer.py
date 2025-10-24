# analyzers/credibility_analyzer.py
# Analyzes source credibility using domain-based scoring (Chart 3 Radar)
# Integrated from team member's credibility scorer

from urllib.parse import urlparse
from typing import Dict, Any
import re
import config  # ADD THIS IMPORT

class CredibilityAnalyzer:
    """
    Analyzes source credibility based on domain characteristics.
    Generates radar chart data for Chart 3.
    """
    
    def __init__(self):
        # Import from shared config
        self.problematic_sources = config.PROBLEMATIC_NEWS_SOURCES
        self.reliable_sources = config.RELIABLE_NEWS_SOURCES
        self.old_domains = config.OLD_ESTABLISHED_DOMAINS
        self.moderate_age_domains = config.MODERATE_AGE_DOMAINS

        # Storage for WHOIS results (used in descriptions)
        self._last_whois_result = {'verified': False}

        print(f"📊 CredibilityAnalyzer: Using {len(self.reliable_sources)} reliable sources from config")
    
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
        
        # Prepare detailed metrics for Chart 3 breakdown display
        detailed_metrics = {
            "domain_age": {
                "score": int(domain_age),
                "status": self._get_domain_age_status(domain_age),
                "description": self._get_domain_age_description(domain, domain_age)
            },
            "url_structure": {
                "score": int(url_structure),
                "status": self._get_url_structure_status(url_structure),
                "description": self._get_url_structure_description(domain, url_structure)
            },
            "site_structure": {
                "score": int(site_structure),
                "status": self._get_site_structure_status(site_structure),
                "description": self._get_site_structure_description(domain, site_structure)
            },
            "content_format": {
                "score": int(content_format),
                "status": self._get_content_format_status(content_format),
                "description": self._get_content_format_description(url, content_format)
            }
        }

        return {
            "score": round(credibility_score, 3),
            "chart3_data": chart3_data,
            "detailed_metrics": detailed_metrics,  # NEW: For detailed breakdown cards
            "metrics": {
                "domain_age": round(domain_age / 100, 3),
                "url_structure": round(url_structure / 100, 3),
                "site_structure": round(site_structure / 100, 3),
                "content_format": round(content_format / 100, 3)
            },
            "known_source": self._get_source_classification(domain),
            "enhanced_info": self.get_enhanced_metrics()  # WHOIS verification data for UI
        }

    def get_enhanced_metrics(self) -> Dict:
        """
        Get metrics with WHOIS verification flags for frontend display.

        Returns enhanced metadata for UI credibility cards:
        - whois_verified: Boolean flag for verification badge
        - domain_created: Registration date for display
        - domain_registrar: Registrar name for additional context
        - verification_source: Label for data source
        """
        whois_result = getattr(self, '_last_whois_result', {'verified': False})

        return {
            'whois_verified': whois_result.get('verified', False),
            'domain_created': whois_result.get('created_date', ''),
            'domain_registrar': whois_result.get('registrar', ''),
            'verification_source': 'WHOIS API' if whois_result.get('verified') else 'Pattern Estimation'
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
        Score domain age with WHOIS verification (0-100)

        Scoring thresholds (based on real domain age data):
        - 20+ years: 95 (highly established organizations like BBC, NYTimes)
        - 10-20 years: 90 (well-established presence)
        - 5-10 years: 85 (established presence)
        - 3-5 years: 75 (moderate history)
        - 2-3 years: 65 (some history)
        - 1-2 years: 55 (new but legitimate)
        - <1 year: 40 (very new, higher risk)

        Fallback logic: If WHOIS unavailable, use pattern-based estimation
        """
        # Initialize with unverified result
        whois_result = {'verified': False, 'age_years': None}

        try:
            from .whois_helper import get_domain_age as whois_get_domain_age

            if config.ENABLE_WHOIS:
                # Attempt WHOIS lookup (with cache)
                whois_result = whois_get_domain_age(domain) or whois_result

                if whois_result.get('verified') and whois_result.get('age_years') is not None:
                    age = whois_result['age_years']

                    # Store for use in descriptions
                    self._last_whois_result = whois_result

                    # Score based on actual WHOIS age
                    if age >= 20:
                        return 95.0  # Very established (20+ years)
                    elif age >= 10:
                        return 90.0  # Well established (10-20 years)
                    elif age >= 5:
                        return 85.0  # Established (5-10 years)
                    elif age >= 3:
                        return 75.0  # Moderate (3-5 years)
                    elif age >= 2:
                        return 65.0  # Some history (2-3 years)
                    elif age >= 1:
                        return 55.0  # New but legitimate (1-2 years)
                    else:
                        return 40.0  # Very new (<1 year)

        except Exception as e:
            print(f"⚠️ WHOIS error for {domain}: {e}")

        # Store that we're using fallback
        self._last_whois_result = {'verified': False}

        # FALLBACK: Use existing pattern-based estimation
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

    # Helper methods for detailed metric descriptions
    def _get_domain_age_status(self, score: float) -> str:
        """Get status label for domain age score"""
        if score >= 85:
            return "Established"
        elif score >= 70:
            return "Mature"
        elif score >= 50:
            return "Moderate"
        else:
            return "New/Unknown"

    def _get_domain_age_description(self, domain: str, score: float) -> str:
        """
        Create detailed domain age description using comprehensive WHOIS data.

        Description includes:
        - Registration date and age
        - Registrar information
        - Ownership transparency
        - Expiry information
        - Security features
        - Hosting infrastructure
        - Warning flags
        """
        # Check if we have WHOIS data from scoring
        whois = getattr(self, '_last_whois_result', {'verified': False})

        if whois.get('verified') and whois.get('age_years') is not None:
            parts = []

            # 1. BASIC AGE INFO
            created = whois.get('created_date', '')
            age = whois.get('age_years', 0)

            if created and age:
                year = created[:4]
                age_int = int(age)

                if age >= 20:
                    parts.append(f"Domain registered in {year} ({age_int}+ years). Highly established online presence")
                elif age >= 10:
                    parts.append(f"Domain registered in {year} ({age_int} years). Well-established presence")
                elif age >= 5:
                    parts.append(f"Domain registered in {year} ({age_int} years). Established presence")
                elif age >= 2:
                    parts.append(f"Domain registered in {year} ({age_int} years). Moderate history")
                else:
                    parts.append(f"Domain registered in {year} ({age_int} year(s)). Relatively new")

            # 2. REGISTRAR INFO
            registrar = whois.get('registrar', '')
            if registrar:
                if whois.get('registrar_trusted'):
                    parts.append(f"Registered with reputable registrar: {registrar}")
                else:
                    parts.append(f"Registered with: {registrar}")

            # 3. TRANSPARENCY
            if whois.get('privacy_protected'):
                parts.append("⚠️ Domain uses privacy protection - owner identity hidden")
            elif whois.get('registrant_org'):
                org = whois.get('registrant_org')
                parts.append(f"Public registration: {org}")

            # 4. EXPIRY INFORMATION (NEW)
            expires_date = whois.get('expires_date')
            days_until_expiry = whois.get('days_until_expiry')

            if days_until_expiry is not None:
                if days_until_expiry < 30:
                    parts.append(f"⚠️ Domain expires in {days_until_expiry} days - highly suspicious for established news org")
                elif days_until_expiry < 90:
                    parts.append(f"⚠️ Domain expires soon ({days_until_expiry} days)")
                elif expires_date:
                    expire_year = expires_date[:4]
                    parts.append(f"Valid until {expire_year}")

            # 5. UPDATE INFORMATION (NEW)
            days_since_update = whois.get('days_since_update')

            if days_since_update is not None:
                if days_since_update < 7:
                    parts.append("⚠️ Domain information updated within last week - potential red flag")
                elif days_since_update < 30:
                    parts.append("Domain information recently updated")

            # 6. HOSTING INFRASTRUCTURE (NEW)
            hosting = whois.get('hosting_provider', '')
            if hosting and hosting != 'Unknown':
                parts.append(f"Hosted on {hosting}")

            # 7. SECURITY FEATURES (NEW)
            security_features = []
            if whois.get('dnssec_enabled'):
                security_features.append('DNSSEC')
            if whois.get('domain_locked'):
                security_features.append('Transfer Lock')

            if security_features:
                parts.append(f"Security features: {', '.join(security_features)} enabled")

            # Join all parts into readable description
            return '. '.join(parts) + '.'

        # FALLBACK: Use existing pattern-based descriptions
        if domain in self.old_domains:
            return f"Long-standing reputable news organization ({domain})"
        elif domain in self.moderate_age_domains:
            return f"Established news outlet with good track record"
        elif domain in self.reliable_sources:
            return f"Recognized news source in the industry"
        elif score >= 50:
            return "Domain shows moderate maturity indicators"
        else:
            return "Limited history or unknown domain reputation"

    def _get_url_structure_status(self, score: float) -> str:
        """Get status label for URL structure score"""
        if score >= 85:
            return "Professional"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Standard"
        else:
            return "Suspicious"

    def _get_url_structure_description(self, domain: str, score: float) -> str:
        """Get description for URL structure"""
        if score >= 85:
            return "Professional domain structure with trusted TLD"
        elif score >= 70:
            return "Well-structured domain name with news indicators"
        elif score >= 50:
            return "Standard domain structure without red flags"
        else:
            return "Domain structure shows suspicious patterns"

    def _get_site_structure_status(self, score: float) -> str:
        """Get status label for site structure score"""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Adequate"
        else:
            return "Poor"

    def _get_site_structure_description(self, domain: str, score: float) -> str:
        """Get description for site structure"""
        if domain in self.reliable_sources:
            return "Established news organization with full transparency"
        elif score >= 70:
            return "Professional news site structure"
        elif score >= 50:
            return "Standard website structure for news content"
        else:
            return "Site structure lacks professional news indicators"

    def _get_content_format_status(self, score: float) -> str:
        """Get status label for content format score"""
        if score >= 85:
            return "Professional"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Standard"
        else:
            return "Low Quality"

    def _get_content_format_description(self, url: str, score: float) -> str:
        """Get description for content format"""
        if score >= 85:
            return "Professional article formatting with proper structure"
        elif score >= 70:
            return "Well-formatted content with news article indicators"
        elif score >= 50:
            return "Standard content formatting"
        else:
            return "Content format shows quality concerns"