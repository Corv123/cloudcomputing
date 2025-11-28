# analyzers/whois_helper.py
# WHOIS integration module for accurate domain age verification
# Provides real domain registration data with SQLite caching to reduce API calls

import sqlite3
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import config

class WhoisHelper:
    """
    WHOIS API integration with intelligent caching.

    Features:
    - Real domain registration date verification via WHOIS XML API
    - 24-hour SQLite cache to minimize API usage (500 free requests/month)
    - Graceful fallback to estimation if API unavailable
    - Comprehensive error handling and logging
    """

    def __init__(self):
        # Use /tmp for Lambda (writable directory) or local path for other environments
        db_name = getattr(config, 'DATABASE_NAME', 'articles.db')
        if os.path.exists('/tmp'):  # Lambda environment
            self.db_name = '/tmp/whois_cache.db'
        else:
            self.db_name = db_name
        self._init_cache_table()

    def get_domain_age(self, domain: str) -> Optional[Dict]:
        """
        Get comprehensive domain WHOIS information with privacy and registrar analysis.

        Process:
        1. Check SQLite cache first (valid if <24 hours old)
        2. If not cached or stale, call WHOIS API
        3. If API fails, return unverified result for fallback handling

        Args:
            domain (str): Clean domain name (e.g., "bbc.com")

        Returns:
            Dict with structure:
            {
                'age_years': 28.5,
                'created_date': '1996-03-10',
                'registrar': 'GoDaddy.com, LLC',
                'privacy_protected': False,  # True if using privacy service
                'registrant_org': 'British Broadcasting Corporation',  # If public
                'verified': True,  # False if using fallback
                'status': 'active',
                'registrar_trusted': True  # True if in TRUSTED_REGISTRARS list
            }

            Returns None if domain invalid, empty dict with verified=False if API fails
        """
        if not domain:
            return None

        # Step 1: Check cache first (24-hour freshness)
        cached = self._get_cached_age(domain)
        if cached:
            # Cache hit - return immediately
            return cached

        # Step 2: Cache miss - fetch from WHOIS API
        if config.ENABLE_WHOIS:
            whois_data = self._fetch_whois_data(domain)
            if whois_data:
                result = self._calculate_age_from_whois(whois_data)
                if result:
                    # Cache successful WHOIS lookup
                    self._cache_age(domain, result)
                    return result

        # Step 3: API disabled or failed - return unverified for fallback
        return {'verified': False, 'age_years': None, 'source': 'estimated'}

    def _fetch_whois_data(self, domain: str) -> Optional[Dict]:
        """
        Fetch WHOIS data from WHOIS XML API.

        API: https://www.whoisxmlapi.com/whoisserver/WhoisService
        Rate Limit: 500 requests/month (free tier)
        Timeout: 5 seconds to prevent hanging

        Args:
            domain (str): Domain to lookup

        Returns:
            Full WHOIS JSON response or None if error
        """
        try:
            response = requests.get(
                config.WHOIS_API_URL,
                params={
                    'apiKey': config.WHOIS_API_KEY,
                    'domainName': domain,
                    'outputFormat': 'JSON'
                },
                timeout=config.WHOIS_TIMEOUT_SECONDS
            )

            if response.status_code == 200:
                data = response.json()
                # Successful WHOIS lookup
                if 'WhoisRecord' in data:
                    return data
                else:
                    print(f"⚠️ WHOIS API returned unexpected format for {domain}")
                    return None
            else:
                print(f"⚠️ WHOIS API error {response.status_code} for {domain}")
                return None

        except requests.exceptions.Timeout:
            print(f"⏱️ WHOIS API timeout for {domain} (>{config.WHOIS_TIMEOUT_SECONDS}s)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️ WHOIS API network error for {domain}: {e}")
            return None
        except Exception as e:
            print(f"⚠️ WHOIS API unexpected error for {domain}: {e}")
            return None

    def _calculate_age_from_whois(self, whois_data: Dict) -> Optional[Dict]:
        """
        Extract and calculate domain age from WHOIS response with privacy and registrar analysis.

        Parsing strategy:
        - Try registryData.createdDate first (most reliable)
        - Fallback to createdDate if registryData missing
        - Extract registrar and check trust status
        - Detect privacy protection services
        - Extract registrant organization if public

        Args:
            whois_data (Dict): Full WHOIS API JSON response

        Returns:
            Dict with age, dates, privacy info, and registrar trust or None if parsing fails
        """
        try:
            record = whois_data.get('WhoisRecord', {})

            # Extract creation date (try multiple paths)
            created_date_str = None
            registry_data = record.get('registryData', {})

            if registry_data and 'createdDate' in registry_data:
                created_date_str = registry_data['createdDate']
            elif 'createdDate' in record:
                created_date_str = record['createdDate']

            if not created_date_str:
                print(f"⚠️ No creation date found in WHOIS data")
                return None

            # Parse date (handle various formats: 2024-01-15T00:00:00Z or 2024-01-15)
            created_date_str = created_date_str.split('T')[0]  # Remove time component
            created_date = datetime.strptime(created_date_str, '%Y-%m-%d')

            # Calculate age in years (with decimal precision)
            age_delta = datetime.now() - created_date
            age_years = age_delta.days / 365.25  # Account for leap years

            # Extract registrar
            registrar = record.get('registrarName', '') or registry_data.get('registrarName', '')

            # Check if registrar is trusted
            registrar_trusted = self._is_trusted_registrar(registrar)

            # Extract registrant information for privacy detection
            registrant = record.get('registrant', {}) or registry_data.get('registrant', {})
            registrant_org = registrant.get('organization', '') or registrant.get('name', '')

            # Detect privacy protection
            privacy_protected = self._detect_privacy_protection(registrant, registrant_org)

            # Clean up registrant org (don't show if privacy protected)
            if privacy_protected:
                registrant_org = ''

            status = 'active'  # Assume active if WHOIS returned data

            # === NEW: Extract 6 additional fields ===

            # 1. Expiry Information
            expires_date_str = registry_data.get('expiresDate', '') or record.get('expiresDate', '')
            expires_date = None
            days_until_expiry = None

            if expires_date_str:
                expires_date = expires_date_str.split('T')[0]  # Get YYYY-MM-DD
                try:
                    expires_dt = datetime.strptime(expires_date, '%Y-%m-%d')
                    days_until_expiry = (expires_dt - datetime.now()).days
                except:
                    pass

            # 2. Update Information
            updated_date_str = registry_data.get('updatedDate', '') or record.get('updatedDate', '')
            updated_date = None
            days_since_update = None

            if updated_date_str:
                updated_date = updated_date_str.split('T')[0]
                try:
                    updated_dt = datetime.strptime(updated_date, '%Y-%m-%d')
                    days_since_update = (datetime.now() - updated_dt).days
                except:
                    pass

            # 3. Nameserver / Hosting Information
            name_servers = []
            name_server_data = record.get('nameServers', {})

            if isinstance(name_server_data, dict):
                name_servers = name_server_data.get('hostNames', [])
            elif isinstance(name_server_data, list):
                name_servers = name_server_data

            # Extract hosting provider from nameservers
            hosting_provider = self._identify_hosting_provider(name_servers)

            # 4. DNSSEC Status
            dnssec_enabled = False
            dnssec_value = registry_data.get('dnssec', '') or record.get('dnssec', '')

            if dnssec_value:
                dnssec_lower = str(dnssec_value).lower()
                dnssec_enabled = dnssec_lower in ['signed', 'yes', 'true', 'enabled', 'signeddelegation']

            # 5. Domain Lock Status
            domain_locked = False
            status_codes = registry_data.get('status', []) or record.get('status', [])

            # Handle both string and list formats
            if isinstance(status_codes, str):
                status_codes = [status_codes]

            # Check for lock indicators
            lock_indicators = [
                'clientTransferProhibited',
                'serverTransferProhibited',
                'clientUpdateProhibited',
                'serverUpdateProhibited'
            ]

            for status_code in status_codes:
                if any(lock in status_code for lock in lock_indicators):
                    domain_locked = True
                    break

            # Build comprehensive result with all fields
            result = {
                # EXISTING FIELDS:
                'age_years': round(age_years, 2),
                'created_date': created_date_str,
                'source': 'whois',
                'verified': True,
                'registrar': registrar,
                'registrar_trusted': registrar_trusted,
                'privacy_protected': privacy_protected,
                'registrant_org': registrant_org,
                'status': status,

                # NEW FIELDS:
                'expires_date': expires_date,
                'days_until_expiry': days_until_expiry,
                'updated_date': updated_date,
                'days_since_update': days_since_update,
                'name_servers': name_servers[:3] if name_servers else [],  # Top 3
                'hosting_provider': hosting_provider,
                'dnssec_enabled': dnssec_enabled,
                'domain_locked': domain_locked,
                'status_codes': status_codes
            }

            return result

        except ValueError as e:
            print(f"⚠️ Date parsing error in WHOIS data: {e}")
            return None
        except Exception as e:
            print(f"⚠️ WHOIS data processing error: {e}")
            return None

    def _identify_hosting_provider(self, name_servers: list) -> str:
        """
        Identify hosting provider from nameserver hostnames.

        Returns friendly provider name or 'Unknown' if not recognized.

        Args:
            name_servers (list): List of nameserver hostnames

        Returns:
            str: Friendly provider name or 'Unknown'
        """
        if not name_servers:
            return 'Unknown'

        # Join all nameservers for pattern matching
        ns_string = ' '.join(name_servers).lower()

        # Check against known providers
        providers = {
            'cloudflare': 'Cloudflare',
            'awsdns': 'AWS Route 53',
            'googledomains': 'Google Domains',
            'google': 'Google Cloud DNS',
            'azure-dns': 'Microsoft Azure',
            'nsone': 'NS1',
            'dyn.com': 'Oracle Dyn',
            'ultradns': 'UltraDNS',
            'akamai': 'Akamai',
            'fastly': 'Fastly',
            'digitalocean': 'DigitalOcean',
            'linode': 'Linode',
            'vultr': 'Vultr'
        }

        for pattern, name in providers.items():
            if pattern in ns_string:
                return name

        return 'Unknown'

    def _is_trusted_registrar(self, registrar: str) -> bool:
        """
        Check if registrar is in the trusted list.

        Args:
            registrar (str): Registrar name from WHOIS

        Returns:
            bool: True if registrar is trusted
        """
        if not registrar:
            return False

        registrar_lower = registrar.lower()

        # Check against trusted registrars from config
        for trusted in config.TRUSTED_REGISTRARS:
            if trusted.lower() in registrar_lower:
                return True

        return False

    def _detect_privacy_protection(self, registrant: Dict, registrant_org: str) -> bool:
        """
        Detect if domain uses privacy protection service.

        Privacy indicators:
        - Registrant name/org contains: 'Privacy', 'WhoisGuard', 'Protected', 'Redacted'
        - Generic privacy service names
        - Contact information redacted

        Args:
            registrant (Dict): Registrant data from WHOIS
            registrant_org (str): Organization name

        Returns:
            bool: True if privacy protection detected
        """
        privacy_keywords = [
            'privacy', 'whoisguard', 'protected', 'redacted',
            'proxy', 'guard', 'shield', 'domains by proxy',
            'contact privacy', 'identity protect', 'private registration'
        ]

        # Check organization name
        if registrant_org:
            org_lower = registrant_org.lower()
            if any(keyword in org_lower for keyword in privacy_keywords):
                return True

        # Check registrant name
        registrant_name = registrant.get('name', '').lower()
        if any(keyword in registrant_name for keyword in privacy_keywords):
            return True

        # Check if contact info is redacted
        email = registrant.get('email', '').lower()
        if 'redacted' in email or 'privacy' in email:
            return True

        return False

    def _init_cache_table(self) -> None:
        """
        Initialize enhanced WHOIS cache table in SQLite.

        Table schema:
        - domain: Primary key, clean domain name
        - age_years: Decimal age in years
        - created_date: ISO date string (YYYY-MM-DD)
        - registrar: Registrar name
        - privacy_protected: Boolean flag for privacy protection
        - registrant_org: Organization name (if public)
        - verified: Boolean flag (always 1 for cached WHOIS data)
        - cached_at: Timestamp for freshness check

        Index on cached_at for efficient cleanup queries
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whois_cache (
                    domain TEXT PRIMARY KEY,
                    age_years REAL NOT NULL,
                    created_date TEXT,
                    registrar TEXT,
                    privacy_protected BOOLEAN DEFAULT 0,
                    registrant_org TEXT,
                    verified BOOLEAN DEFAULT 1,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Index for efficient date-based queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_whois_cache_date
                ON whois_cache(cached_at)
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"⚠️ WHOIS cache table initialization error: {e}")

    def _get_cached_age(self, domain: str) -> Optional[Dict]:
        """
        Retrieve cached comprehensive WHOIS data if fresh (<24 hours).

        Cache freshness logic:
        - WHOIS data rarely changes, so 24-hour cache is safe
        - Reduces API calls by ~95% for repeated domains
        - Critical for staying within 500 req/month free tier

        Args:
            domain (str): Domain to lookup

        Returns:
            Full result dict with all 13+ fields if cache hit and fresh, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Calculate freshness threshold (24 hours ago)
            threshold = datetime.now() - timedelta(hours=config.WHOIS_CACHE_HOURS)

            cursor.execute('''
                SELECT
                    age_years, created_date, registrar, verified,
                    privacy_protected, registrant_org,
                    expires_date, days_until_expiry, updated_date,
                    hosting_provider, dnssec_enabled, domain_locked,
                    cached_at
                FROM whois_cache
                WHERE domain = ? AND cached_at > ?
            ''', (domain, threshold.isoformat()))

            result = cursor.fetchone()
            conn.close()

            if result:
                # Cache hit - reconstruct full result dict with all fields
                return {
                    'age_years': result[0],
                    'created_date': result[1],
                    'source': 'whois',
                    'verified': bool(result[3]),
                    'registrar': result[2] or '',
                    'status': 'active',
                    'privacy_protected': bool(result[4]),
                    'registrant_org': result[5] or '',
                    'expires_date': result[6] or None,
                    'days_until_expiry': result[7],
                    'updated_date': result[8] or None,
                    'hosting_provider': result[9] or '',
                    'dnssec_enabled': bool(result[10]),
                    'domain_locked': bool(result[11]),
                    'registrar_trusted': self._is_trusted_registrar(result[2] or '')
                }

            return None  # Cache miss

        except Exception as e:
            print(f"⚠️ Cache lookup error for {domain}: {e}")
            return None

    def _cache_age(self, domain: str, whois_result: Dict) -> None:
        """
        Store comprehensive WHOIS result in cache for 24-hour reuse.

        Uses REPLACE to handle both INSERT and UPDATE cases.
        This prevents duplicate key errors on re-verification.

        Args:
            domain (str): Domain being cached
            whois_result (Dict): Full WHOIS result to store (includes all 13+ fields)
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            cursor.execute('''
                REPLACE INTO whois_cache (
                    domain, age_years, created_date, registrar,
                    privacy_protected, registrant_org, verified,
                    expires_date, days_until_expiry, updated_date,
                    hosting_provider, dnssec_enabled, domain_locked,
                    cached_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                domain,
                whois_result.get('age_years', 0),
                whois_result.get('created_date', ''),
                whois_result.get('registrar', ''),
                whois_result.get('privacy_protected', False),
                whois_result.get('registrant_org', ''),
                1,  # Always verified if we're caching WHOIS data
                whois_result.get('expires_date', ''),
                whois_result.get('days_until_expiry'),
                whois_result.get('updated_date', ''),
                whois_result.get('hosting_provider', ''),
                whois_result.get('dnssec_enabled', False),
                whois_result.get('domain_locked', False),
                datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"⚠️ Cache storage error for {domain}: {e}")


# Module-level convenience function for easy import
_whois_helper_instance = None

def get_domain_age(domain: str) -> Optional[Dict]:
    """
    Convenience function for getting domain age.
    Uses singleton pattern to avoid re-initializing cache table.

    Usage:
        from .whois_helper import get_domain_age

        result = get_domain_age('bbc.com')
        if result and result.get('verified'):
            age = result['age_years']  # 36.5
            created = result['created_date']  # '1987-06-15'
    """
    global _whois_helper_instance

    if _whois_helper_instance is None:
        _whois_helper_instance = WhoisHelper()

    return _whois_helper_instance.get_domain_age(domain)
