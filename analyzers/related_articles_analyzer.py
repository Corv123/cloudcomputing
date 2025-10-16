# analyzers/related_articles_analyzer.py
# Finds related articles from reputable sources using Google News RSS (Chart 6)
# FREE - No API key required!

from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from urllib.parse import urlparse, quote
import re
from datetime import datetime
import config  # ADD THIS IMPORT

class RelatedArticlesAnalyzer:
    """
    Finds related articles from reputable sources using:
    1. Google News RSS feed (primary - FREE, no API key)
    2. DuckDuckGo News search (fallback - FREE, no API key)

    Generates data for Chart 6 - Related Articles display with REAL working URLs.
    """

    def __init__(self):
        # Import from shared config
        self.reliable_sources = config.RELIABLE_NEWS_SOURCES

        print(f"📰 RelatedArticlesAnalyzer: Using {len(self.reliable_sources)} reliable sources from config")

        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def _is_reliable_source(self, domain: str) -> bool:
        """
        Check if domain is from a reliable source with FLEXIBLE matching.
        Handles subdomains, www prefixes, country-specific TLDs, etc.

        Examples that will match:
        - domain="uk.reuters.com" matches "reuters.com" ✓
        - domain="edition.cnn.com" matches "cnn.com" ✓
        - domain="bbc.co.uk" matches "bbc.com" ✓
        - domain="m.theguardian.com" matches "theguardian.com" ✓
        """
        # Clean domain for matching (remove common prefixes)
        clean_domain = domain.lower()
        clean_domain = clean_domain.replace('www.', '').replace('m.', '').replace('edition.', '')
        clean_domain = clean_domain.replace('uk.', '').replace('us.', '').replace('au.', '')

        # Method 1: Exact match
        if clean_domain in self.reliable_sources:
            return True

        # Method 2: Check if reliable source is substring of domain (handles subdomains)
        for reliable_source in self.reliable_sources:
            reliable_clean = reliable_source.replace('www.', '')

            # Forward match: reliable source in domain
            # e.g., "reuters.com" in "uk.reuters.com"
            if reliable_clean in clean_domain:
                return True

            # Backward match: domain in reliable source
            # e.g., "bbc" in "bbc.com" when domain is "bbc.co.uk"
            if clean_domain in reliable_clean:
                return True

            # Method 3: Base domain matching (e.g., "reuters" matches both "reuters.com" and "uk.reuters.com")
            base_reliable = reliable_clean.split('.')[0]
            base_domain = clean_domain.split('.')[0]

            # Only match if base is meaningful (4+ chars to avoid false matches like "cnn" in "cnna.com")
            if len(base_reliable) >= 4 and base_reliable == base_domain:
                return True

        return False

    def analyze(self, title: str, content: str, domain: str) -> Dict[str, Any]:
        """
        Find related articles from reputable sources

        Args:
            title: Article title (WILL BE USED AS SEARCH QUERY)
            content: Article content (for relevance scoring)
            domain: Original article domain

        Returns:
            dict with 'chart6_data' containing related articles
        """
        try:
            # CHANGE: Use title directly as search query instead of extracting keywords
            print(f"🔍 Searching for articles about: '{title}'")

            # Search for related articles using the FULL TITLE
            related_articles = self._search_related_articles(title, domain)

            # Calculate relevance scores
            if related_articles:
                related_articles = self._calculate_relevance(
                    content, related_articles
                )

            # Prepare Chart 6 data
            chart6_data = self._prepare_chart_data(related_articles)

            return {
                "chart6_data": chart6_data,
                "search_query": title,  # Changed from keywords_used
                "total_found": len(related_articles)
            }

        except Exception as e:
            print(f"Error in related articles analysis: {e}")
            import traceback
            traceback.print_exc()
            return {
                "chart6_data": {
                    "articles": [],
                    "message": "Unable to fetch related articles at this time"
                },
                "search_query": title,
                "total_found": 0
            }

    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """
        DEPRECATED: Now using full title instead
        Kept for backward compatibility only
        """
        return [title]

    def _search_related_articles(self, title: str, exclude_domain: str) -> List[Dict]:
        """
        Search for related articles using FREE sources (no API key needed!)

        Strategy:
        1. Try Google News RSS (primary) - using full title
        2. Try DuckDuckGo News (fallback) - using full title
        3. Return empty if both fail

        Returns:
            List of related articles with REAL, working URLs
        """
        related_articles = []

        try:
            # Try Google News RSS first
            print("Searching Google News RSS...")
            related_articles = self._search_google_news(title, exclude_domain)

            if related_articles:
                print(f"Google News RSS returned {len(related_articles)} articles")
                return related_articles

            # Fallback to DuckDuckGo
            print("Falling back to DuckDuckGo News...")
            related_articles = self._search_duckduckgo_news(title, exclude_domain)

            if related_articles:
                print(f"DuckDuckGo returned {len(related_articles)} articles")
                return related_articles

            print("No articles found from any source")
            return []

        except Exception as e:
            print(f"Error searching for articles: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _search_google_news(self, title: str, exclude_domain: str) -> List[Dict]:
        """
        Search Google News RSS feed using FULL ARTICLE TITLE
        NOW HANDLES GOOGLE NEWS REDIRECT URLS!
        Returns REAL, working article URLs
        FREE - No API key required!

        Args:
            title: Full article title to search for
            exclude_domain: Domain to exclude (original article)
        """
        try:
            import feedparser

            # Clean the title for searching
            # Remove special characters that might break the query
            clean_title = re.sub(r'[^\w\s]', ' ', title)  # Remove punctuation
            clean_title = ' '.join(clean_title.split())  # Normalize whitespace

            print(f"📰 Google News search: '{clean_title[:60]}...'")

            # Build multiple query variations (from most specific to least)
            queries_to_try = [
                clean_title,  # Full title
                ' '.join(clean_title.split()[:10]),  # First 10 words
                ' '.join(clean_title.split()[:6]),  # First 6 words (for long titles)
            ]

            for attempt, query in enumerate(queries_to_try):
                try:
                    search_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

                    print(f"  Attempt {attempt+1}: '{query[:50]}...'")

                    response = requests.get(search_url, timeout=15, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                        'Accept-Language': 'en-US,en;q=0.9'
                    })

                    if response.status_code != 200:
                        print(f"    HTTP {response.status_code}")
                        continue

                    feed = feedparser.parse(response.content)
                    print(f"    Feed entries: {len(feed.entries)}")

                    if len(feed.entries) == 0:
                        print(f"    No entries, trying shorter query...")
                        continue

                    articles = []
                    google_redirects = 0
                    no_source_url = 0
                    unreliable_count = 0

                    for entry in feed.entries[:40]:
                        try:
                            link = entry.link

                            # NEW: Extract real URL from Google News redirect
                            real_url = None

                            # Method 1: Check if it's a Google redirect
                            if 'news.google.com' in link:
                                google_redirects += 1

                                # Try to get real URL from entry.source
                                if hasattr(entry, 'source') and hasattr(entry.source, 'href'):
                                    real_url = entry.source.href

                                # Method 2: Try to extract from link tags
                                elif hasattr(entry, 'links'):
                                    for entry_link in entry.links:
                                        if entry_link.get('href') and 'news.google.com' not in entry_link.get('href', ''):
                                            real_url = entry_link['href']
                                            break

                                # Method 3: Follow the redirect (slower but works)
                                if not real_url:
                                    try:
                                        redirect_response = requests.head(link, allow_redirects=True, timeout=5)
                                        if redirect_response.url and 'news.google.com' not in redirect_response.url:
                                            real_url = redirect_response.url
                                    except:
                                        pass

                                # If we still don't have a real URL, skip
                                if not real_url:
                                    no_source_url += 1
                                    continue

                                # Use the real URL
                                link = real_url

                            # Now process with real URL
                            domain = self._extract_domain(link)

                            if not domain:
                                continue

                            # Skip same domain
                            if domain == exclude_domain:
                                continue

                            # Check reliability with flexible matching
                            if not self._is_reliable_source(domain):
                                unreliable_count += 1
                                continue

                            # SUCCESS!
                            print(f"    ✓ {domain}: {entry.title[:50]}")

                            # Get source name
                            source_name = domain
                            if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
                                source_name = entry.source.title

                            # Parse date
                            pub_date = ''
                            if hasattr(entry, 'published'):
                                try:
                                    from email.utils import parsedate_to_datetime
                                    dt = parsedate_to_datetime(entry.published)
                                    pub_date = dt.strftime('%Y-%m-%d')
                                except:
                                    pass

                            # Clean snippet
                            snippet = ''
                            if hasattr(entry, 'summary'):
                                snippet = self._clean_html(entry.summary)

                            articles.append({
                                'title': entry.title[:150],
                                'url': link,  # Real URL, not Google redirect
                                'domain': domain,
                                'source_name': source_name,
                                'snippet': snippet[:250],
                                'published_date': pub_date
                            })

                            if len(articles) >= 12:
                                break

                        except Exception as e:
                            continue

                    print(f"    Result: {len(articles)} articles (redirects: {google_redirects}, no source: {no_source_url}, unreliable: {unreliable_count})")

                    if articles:
                        print(f"  ✅ Found {len(articles)} articles")
                        return articles
                    else:
                        print(f"    No reliable sources, trying shorter query...")

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            print("  ❌ All query variations failed")
            return []

        except ImportError:
            print("feedparser not installed")
            return []
        except Exception as e:
            print(f"Critical error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _search_duckduckgo_news(self, title: str, exclude_domain: str) -> List[Dict]:
        """
        Search DuckDuckGo News using FULL ARTICLE TITLE
        FREE - No API key required!

        Args:
            title: Full article title to search for
            exclude_domain: Domain to exclude
        """
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            # Clean title
            clean_title = re.sub(r'[^\w\s]', ' ', title)
            clean_title = ' '.join(clean_title.split())

            # For very long titles, use first 10 words
            query_words = clean_title.split()
            query = ' '.join(query_words[:10]) if len(query_words) > 10 else clean_title

            print(f"🦆 DuckDuckGo search: '{query[:60]}...'")

            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=30))

                articles = []
                for result in results:
                    try:
                        domain = self._extract_domain(result['url'])

                        # Skip same domain
                        if domain == exclude_domain:
                            continue

                        # Check reliability
                        if not self._is_reliable_source(domain):
                            continue

                        print(f"  ✓ {domain}: {result['title'][:50]}")

                        # Parse date
                        pub_date = ''
                        if 'date' in result:
                            try:
                                if isinstance(result['date'], str):
                                    pub_date = result['date'][:10]
                                else:
                                    pub_date = result['date'].strftime('%Y-%m-%d')
                            except:
                                pass

                        articles.append({
                            'title': result['title'][:150],
                            'url': result['url'],
                            'domain': domain,
                            'source_name': result.get('source', domain),
                            'snippet': result.get('body', '')[:250],
                            'published_date': pub_date
                        })

                        if len(articles) >= 12:
                            break

                    except Exception as e:
                        continue

                print(f"  ✅ Found {len(articles)} articles")
                return articles

        except ImportError:
            print("DDG not installed. Install: pip install ddgs")
            return []
        except Exception as e:
            print(f"DuckDuckGo error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '').lower()
            return domain
        except:
            return ''

    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html_text).strip()

    def _calculate_relevance(self, original_content: str, articles: List[Dict]) -> List[Dict]:
        """
        Calculate relevance scores for found articles based on content similarity
        Uses TF-IDF cosine similarity between original content and article snippets
        """
        try:
            if not articles:
                return articles

            # Prepare texts for comparison
            # Use title + snippet for better relevance calculation
            texts = [original_content] + [
                (art.get('title', '') + ' ' + art.get('snippet', ''))
                for art in articles
            ]

            # Calculate TF-IDF similarities
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=200,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Calculate cosine similarities
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

            # Add relevance scores to articles
            for i, article in enumerate(articles):
                if i < len(similarities):
                    # Use actual similarity score
                    # Boost slightly to ensure minimum visibility (0.25-0.95 range)
                    base_score = float(similarities[i])
                    article['relevance'] = min(0.95, max(0.25, base_score + 0.15))
                else:
                    article['relevance'] = 0.5

            # Sort by relevance (highest first)
            articles.sort(key=lambda x: x.get('relevance', 0), reverse=True)

        except Exception as e:
            print(f"Error calculating relevance: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: assign decreasing scores
            for i, article in enumerate(articles):
                article['relevance'] = max(0.3, 0.85 - (i * 0.08))

        return articles

    def _prepare_chart_data(self, articles: List[Dict]) -> Dict[str, Any]:
        """
        Prepare data for Chart 6 display (article cards)
        Returns properly formatted data with REAL, working URLs
        """
        if not articles:
            return {
                "articles": [],
                "message": "No related articles found from reputable sources"
            }

        # Format articles for frontend display
        chart_articles = []
        for article in articles[:8]:  # Top 8 most relevant articles
            # Validate URL before including
            url = article.get('url', '')
            if not url or not url.startswith('http'):
                continue  # Skip articles with invalid URLs

            chart_articles.append({
                "title": article.get('title', 'Untitled')[:150],
                "source": article.get('source_name', article.get('domain', 'Unknown')),
                "url": url,  # REAL, validated URL
                "relevance": round(article.get('relevance', 0.5), 2),
                "snippet": article.get('snippet', '')[:250],
                "source_credibility": self._get_source_credibility(article.get('domain', '')),
                "published_date": article.get('published_date', ''),
                "url_to_image": article.get('url_to_image', '')
            })

        return {
            "articles": chart_articles,
            "total": len(chart_articles)
        }

    def _get_source_credibility(self, domain: str) -> float:
        """
        Get credibility score for a reliable source
        All sources in our list are reliable, but some are more established
        """
        # Tier 1: Most established and prestigious sources
        tier1 = [
            'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'npr.org',
            'nytimes.com', 'washingtonpost.com', 'wsj.com'
        ]

        # Tier 2: Highly credible major news outlets
        tier2 = [
            'cnn.com', 'theguardian.com', 'bloomberg.com',
            'time.com', 'newsweek.com', 'politico.com',
            'abcnews.go.com', 'cbsnews.com', 'nbcnews.com'
        ]

        # Tier 3: Credible sources
        tier3 = [
            'axios.com', 'usatoday.com', 'businessinsider.com',
            'fortune.com', 'channelnewsasia.com', 'straitstimes.com'
        ]

        if domain in tier1:
            return 0.95
        elif domain in tier2:
            return 0.90
        elif domain in tier3:
            return 0.85
        else:
            # Default for any other reliable source
            return 0.80
