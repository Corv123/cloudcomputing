# analyzers/crosscheck_analyzer.py
# Cross-checks article against web sources using credibility-weighted similarity (Chart 4)
# NEW PHILOSOPHY: High similarity from credible sources = High score (corroboration is good!)
# UPDATED: Uses DynamoDB database as PRIMARY source, web-based as FALLBACK

from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
import requests
from urllib.parse import urlparse, quote
import re
from bs4 import BeautifulSoup
import feedparser
import config

# Try to import boto3 for DynamoDB access
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None

# Try to import embedding utilities
try:
    from embedding_utils import (
        generate_embedding,
        cosine_similarity,
        find_top_similar,
        aggregate_similarity_scores
    )
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    generate_embedding = None
    cosine_similarity = None
    find_top_similar = None
    aggregate_similarity_scores = None

class CrossCheckAnalyzer:
    """
    Cross-checks article against web sources using credibility-weighted similarity.
    
    NEW PHILOSOPHY:
    - High similarity from credible sources = HIGH score (strong corroboration)
    - Low similarity = LOW score (no credible corroboration)
    - Source diversity matters (multiple independent sources = better)
    - Similarity matters MORE when from credible sources
    
    UPDATED BEHAVIOR:
    - PRIMARY: Uses DynamoDB database (from Spark news ingestion pipeline)
    - FALLBACK: Uses web-based progressive search (original logic)
    """
    
    def __init__(self, use_database: bool = True, dynamodb_table: str = "fakenews-scraped-news", region: str = "ap-southeast-2"):
        """
        Initialize CrossCheckAnalyzer
        
        Args:
            use_database: Whether to use DynamoDB database as primary source (default: True)
            dynamodb_table: DynamoDB table name for articles
            region: AWS region
        """
        self.reliable_sources = config.RELIABLE_NEWS_SOURCES
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.use_database = use_database and BOTO3_AVAILABLE
        self.dynamodb_table_name = dynamodb_table
        self.region = region
        
        if self.use_database:
            try:
                self.dynamodb = boto3.resource('dynamodb', region_name=region)
                self.articles_table = self.dynamodb.Table(dynamodb_table)
                print(f"🔍 CrossCheckAnalyzer: Using DynamoDB table '{dynamodb_table}' as PRIMARY source")
                print(f"🔍 CrossCheckAnalyzer: Web-based crosscheck will be used as FALLBACK")
            except Exception as e:
                print(f"⚠️ Could not initialize DynamoDB: {e}")
                print(f"🔍 CrossCheckAnalyzer: Falling back to web-based crosscheck only")
                self.use_database = False
        else:
            print(f"🔍 CrossCheckAnalyzer: Using web-based crosscheck only (database not available)")
        
        print(f"🔍 CrossCheckAnalyzer: Using {len(self.reliable_sources)} reliable sources from config")
    
    def _is_reliable_source(self, domain: str) -> bool:
        """Check if domain is from a reliable source (flexible matching)"""
        clean_domain = domain.lower()
        clean_domain = clean_domain.replace('www.', '').replace('m.', '').replace('edition.', '')
        clean_domain = clean_domain.replace('uk.', '').replace('us.', '').replace('au.', '')
        
        if clean_domain in self.reliable_sources:
            return True
        
        for reliable_source in self.reliable_sources:
            reliable_clean = reliable_source.replace('www.', '')
            if reliable_clean in clean_domain or clean_domain in reliable_clean:
                return True
            base_reliable = reliable_clean.split('.')[0]
            base_domain = clean_domain.split('.')[0]
            if len(base_reliable) >= 4 and base_reliable == base_domain:
                return True
        
        return False
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain.lower()
        except:
            return ""
    
    def _search_database_embeddings(self, content: str, exclude_url: str, top_k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Search database using vector embeddings (PRIMARY method).
        
        Args:
            content: Article content to find similar articles for
            exclude_url: URL to exclude from results
            top_k: Number of top similar articles to return
        
        Returns:
            List of (article_dict, similarity_score) tuples
        """
        if not self.use_database or not EMBEDDING_AVAILABLE:
            return []
        
        try:
            print(f"  🔍 [EMBEDDING] Searching database using vector embeddings...")
            
            # Generate embedding for query article
            query_embedding = generate_embedding(content)
            if not query_embedding:
                print(f"    ⚠️ Failed to generate embedding for query article")
                return []
            
            print(f"    [OK] Generated query embedding ({len(query_embedding)} dimensions)")
            
            # Scan database and collect articles with embeddings
            articles_with_embeddings = []
            articles = []
            response = self.articles_table.scan(Limit=100)
            articles.extend(response.get('Items', []))
            
            # Continue scanning to get ALL articles (no limit for embedding search)
            scan_count = 1
            while 'LastEvaluatedKey' in response:
                try:
                    response = self.articles_table.scan(
                        Limit=100,
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    new_items = response.get('Items', [])
                    articles.extend(new_items)
                    scan_count += 1
                    if scan_count % 10 == 0:
                        print(f"    [DEBUG] Scanned {scan_count} pages, {len(articles)} items so far...")
                except Exception as e:
                    print(f"    [WARN] Scan failed at page {scan_count}: {e}")
                    break
            
            print(f"    [OK] Retrieved {len(articles)} articles from database (scanned {scan_count} pages)")
            
            # Filter and extract embeddings
            exclude_domain = self._extract_domain(exclude_url)
            
            for article in articles:
                try:
                    # Skip if same URL
                    article_url = article.get('source_url', '')
                    if exclude_url and exclude_url in str(article_url):
                        continue
                    
                    # Skip if not scraped news
                    if str(article.get('article_type', '')) != 'scraped_news':
                        continue
                    
                    # Get embedding
                    embedding = article.get('embedding', None)
                    if not embedding:
                        continue  # Skip articles without embeddings
                    
                    # Convert DynamoDB list to Python list
                    if isinstance(embedding, list):
                        embedding_list = [float(x) for x in embedding]
                    else:
                        continue
                    
                    # Get article data
                    article_content = article.get('content', '')
                    if not article_content or len(str(article_content)) < 100:
                        continue
                    
                    article_domain = self._extract_domain(str(article_url))
                    
                    article_dict = {
                        'title': str(article.get('title', '')).strip(),
                        'url': str(article_url).strip(),
                        'domain': article_domain,
                        'content': str(article_content).strip(),
                        'is_reliable': self._is_reliable_source(article_domain),
                        'published_at': str(article.get('published_at', ''))
                    }
                    
                    articles_with_embeddings.append((article_dict, embedding_list))
                    
                except Exception as e:
                    print(f"    [WARN] Error processing article: {e}")
                    continue
            
            print(f"    [OK] Found {len(articles_with_embeddings)} articles with embeddings")
            
            # Find top similar using cosine similarity
            if articles_with_embeddings:
                top_similar = find_top_similar(query_embedding, articles_with_embeddings, top_k=top_k)
                print(f"    [OK] Found {len(top_similar)} similar articles (top-{top_k})")
                return top_similar
            
            return []
            
        except Exception as e:
            print(f"    ⚠️ Embedding search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _search_database(self, title: str, content: str, exclude_url: str, limit: int = 50) -> List[Dict]:
        """
        Search DynamoDB database for similar articles (PRIMARY method)
        
        Args:
            title: Article title for search
            content: Article content for similarity
            exclude_url: URL to exclude from results
            limit: Maximum number of articles to retrieve
        
        Returns:
            List of article dictionaries
        """
        if not self.use_database:
            return []
        
        try:
            print(f"  📊 Searching DynamoDB table: {self.dynamodb_table_name}")
            
            # Scan table (DynamoDB doesn't support full-text search natively)
            # Scan more items to get better coverage (increase limit for better results)
            articles = []
            try:
                response = self.articles_table.scan(Limit=100)  # Scan 100 items per page
                articles.extend(response.get('Items', []))
                print(f"    [DEBUG] First scan: {len(articles)} items")
            except Exception as e:
                print(f"    [ERROR] Failed to scan table: {e}")
                return []
            
            # Continue scanning to get more articles (up to 2000 total for better coverage)
            scan_count = 1
            max_scans = 20  # Scan up to 2000 items total (increased from 500)
            while 'LastEvaluatedKey' in response and scan_count < max_scans:
                try:
                    response = self.articles_table.scan(
                        Limit=100,
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    new_items = response.get('Items', [])
                    articles.extend(new_items)
                    scan_count += 1
                    print(f"    [DEBUG] Scan {scan_count}: {len(new_items)} items (total: {len(articles)})")
                except Exception as e:
                    print(f"    [WARN] Scan {scan_count} failed: {e}")
                    break
            
            print(f"    [OK] Retrieved {len(articles)} articles from database (scanned {scan_count} pages)")
            
            # Filter articles:
            # 1. Exclude the current article
            # 2. Only include scraped news articles (article_type='scraped_news')
            # 3. Must have content
            exclude_domain = self._extract_domain(exclude_url)
            filtered_articles = []
            
            article_type_count = {}
            content_count = 0
            
            for article in articles:
                try:
                    # Handle DynamoDB attribute format (boto3 resource auto-converts, but be safe)
                    # Get article_type
                    article_type = article.get('article_type', '')
                    if article_type:
                        article_type_count[str(article_type)] = article_type_count.get(str(article_type), 0) + 1
                    
                    # Skip if not a scraped news article
                    if str(article_type) != 'scraped_news':
                        continue
                    
                    # Get source_url
                    article_url = article.get('source_url', '')
                    if not article_url:
                        continue
                    
                    # Skip if same URL as current article
                    if exclude_url and exclude_url in str(article_url):
                        continue
                    
                    # Must have content
                    article_content = article.get('content', '')
                    if article_content:
                        content_count += 1
                    
                    if not article_content:
                        continue
                    
                    # Convert to string and check length
                    article_content_str = str(article_content).strip()
                    if len(article_content_str) < 100:
                        continue
                    
                    # Extract domain
                    article_domain = self._extract_domain(str(article_url))
                    
                    # Build article dict
                    article_dict = {
                        'title': str(article.get('title', '')).strip(),
                        'url': str(article_url).strip(),
                        'domain': article_domain,
                        'content': article_content_str,  # Already extracted content
                        'is_reliable': self._is_reliable_source(article_domain),
                        'published_at': str(article.get('published_at', ''))
                    }
                    
                    filtered_articles.append(article_dict)
                    
                except Exception as e:
                    print(f"    [WARN] Error processing article: {type(e).__name__}: {str(e)[:50]}")
                    continue
            
            print(f"    [DEBUG] Article types found: {article_type_count}")
            print(f"    [DEBUG] Articles with content: {content_count}")
            
            print(f"    [OK] Filtered to {len(filtered_articles)} relevant articles")
            
            return filtered_articles
            
        except Exception as e:
            print(f"    ⚠️ Database search failed: {e}")
            return []
    
    def _extract_article_content(self, url: str) -> Optional[str]:
        """
        Extract article content from URL with proper Google News redirect handling.
        Returns content text or None if extraction fails
        """
        try:
            # Handle Google News redirect URLs - extract real URL from redirect
            real_url = url
            if 'news.google.com/rss/articles' in url:
                # For Google News redirects, try to get the final URL
                # First, make a HEAD request to get redirect location
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml',
                        'Referer': 'https://www.google.com/'
                    }
                    # Use HEAD request first to get redirect (faster)
                    head_response = requests.head(url, timeout=4, allow_redirects=True, headers=headers)
                    real_url = head_response.url
                    # If still Google News URL, try GET with redirect
                    if 'news.google.com' in real_url:
                        get_response = requests.get(url, timeout=4, allow_redirects=True, headers=headers)
                        real_url = get_response.url
                except:
                    # If redirect extraction fails, proceed with original URL
                    pass
            
            # Now fetch the actual article content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Referer': 'https://www.google.com/',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(real_url, headers=headers, timeout=5)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(["script", "style", "meta", "link", "nav", "header", "footer", "aside", "form", "button"]):
                element.decompose()
            
            # Strategy 1: Try common article content selectors (prioritized)
            article_selectors = [
                'article',
                '[role="article"]',
                '.article-content',
                '.article-body',
                '.story-body',
                '.post-content',
                '.entry-content',
                '#content',
                '.content',
                'main article',
                '.article-text',
                '.article-main'
            ]
            
            for selector in article_selectors:
                try:
                    content_div = soup.select_one(selector)
                    if content_div:
                        text = content_div.get_text(separator=' ', strip=True)
                        # Clean up extra whitespace
                        text = ' '.join(text.split())
                        if len(text) > 300:  # Ensure substantial content
                            print(f"    ✅ Extracted {len(text)} chars using selector: {selector}")
                            return text
                except:
                    continue
            
            # Strategy 2: Get all paragraphs (common fallback)
            paragraphs = soup.find_all('p')
            if paragraphs:
                # Filter out very short paragraphs (likely navigation/menu items)
                meaningful_paragraphs = [p for p in paragraphs if len(p.get_text(strip=True)) > 50]
                if meaningful_paragraphs:
                    text = ' '.join([p.get_text(separator=' ', strip=True) for p in meaningful_paragraphs])
                    text = ' '.join(text.split())  # Clean whitespace
                    if len(text) > 300:
                        print(f"    ✅ Extracted {len(text)} chars from paragraphs")
                        return text
            
            # Strategy 3: Try main content area
            main = soup.find('main')
            if main:
                text = main.get_text(separator=' ', strip=True)
                text = ' '.join(text.split())
                if len(text) > 300:
                    print(f"    ✅ Extracted {len(text)} chars from main")
                    return text
            
            # Strategy 4: Last resort - body text (but filter out navigation)
            body = soup.find('body')
            if body:
                # Remove common navigation/header/footer elements
                for nav in body.find_all(['nav', 'header', 'footer', 'aside']):
                    nav.decompose()
                
                text = body.get_text(separator=' ', strip=True)
                text = ' '.join(text.split())
                # Filter out very short "words" (likely navigation fragments)
                words = text.split()
                meaningful_words = [w for w in words if len(w) > 2]
                text = ' '.join(meaningful_words)
                
                if len(text) > 300:
                    print(f"    ✅ Extracted {len(text)} chars from body (filtered)")
                    return text
            
            print(f"    ⚠️ Could not extract sufficient content from {real_url[:60]}...")
            return None
            
        except requests.exceptions.Timeout:
            print(f"    ⚠️ Timeout extracting content from {url[:60]}...")
            return None
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️ Request error extracting content from {url[:60]}...: {type(e).__name__}")
            return None
        except Exception as e:
            print(f"    ⚠️ Error extracting content from {url[:60]}...: {type(e).__name__}: {str(e)[:50]}")
            return None
    
    def _search_google_news_progressive(self, query: str, exclude_domain: str, target_count: int = 20) -> List[Dict]:
        """
        Progressive search: Start with full query, gradually reduce words if not enough results.
        This is the FALLBACK method when database search doesn't find enough articles.
        """
        all_articles = []
        seen_urls = set()
        words = query.split()
        
        # Progressive search: try full query, then reduce words
        for num_words in range(len(words), 0, -1):
            if len(all_articles) >= target_count:
                break
            
            search_query = ' '.join(words[:num_words])
            print(f"  Searching: '{search_query}' ({num_words} words)")
            
            try:
                search_url = f"https://news.google.com/rss/search?q={quote(search_query)}&hl=en-US&gl=US&ceid=US:en"
                response = requests.get(search_url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/rss+xml, application/xml, text/xml, */*'
                })
                
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)
                    for entry in feed.entries:
                        if len(all_articles) >= target_count:
                            break
                        
                        link = entry.link
                        if link in seen_urls:
                            continue
                        
                        # Extract domain
                        domain = self._extract_domain(link)
                        
                        # Skip if same domain as excluded
                        if exclude_domain and domain and exclude_domain in domain:
                            continue
                        
                        # Handle Google News redirect URLs
                        if 'news.google.com' in link:
                            # Try to get real URL from source
                            if hasattr(entry, 'source') and hasattr(entry.source, 'url'):
                                link = entry.source.url
                                domain = self._extract_domain(link)
                            else:
                                # Skip Google News redirects we can't resolve
                                continue
                        
                        seen_urls.add(link)
                        all_articles.append({
                            'url': link,
                            'title': entry.get('title', ''),
                            'domain': domain,
                            'is_reliable': self._is_reliable_source(domain) if domain else False
                        })
                
            except Exception as e:
                print(f"    ⚠️ Search error: {e}")
                continue
        
        print(f"  ✅ Found {len(all_articles)} articles from progressive search")
        return all_articles
    
    def _calculate_similarities(self, content: str, web_articles: List[Dict], extract_count: int = 5) -> List[tuple]:
        """
        Calculate similarity scores between content and web articles.
        Prioritizes credible sources for content extraction.
        
        Args:
            content: Current article content
            web_articles: List of web articles (should have at least extract_count)
            extract_count: Number of articles to extract content from (default: 5)
        
        Returns:
            List of (article_dict, similarity_score) tuples
        """
        try:
            if not content or len(content.strip()) < 50:
                print(f"  ⚠️ Article content too short for similarity calculation ({len(content) if content else 0} chars)")
                return []
            
            # If articles already have content (from database), use it directly
            articles_with_content = [art for art in web_articles if art.get('content') and len(str(art.get('content', '')).strip()) > 100]
            
            print(f"  📊 Articles with content: {len(articles_with_content)}")
            print(f"  📊 Total articles: {len(web_articles)}")
            
            # For articles without content (from web search), extract it
            articles_without_content = [art for art in web_articles if not art.get('content') or len(str(art.get('content', '')).strip()) < 100]
            
            # Prioritize credible sources for content extraction
            sorted_articles = sorted(
                articles_without_content,
                key=lambda x: (not x.get('is_reliable', False), -len(x.get('title', ''))),
                reverse=False  # False first (credible), then True
            )
            
            # Extract content from top articles (only if needed)
            target_extract = min(extract_count, len(sorted_articles))
            max_attempts = min(8, len(sorted_articles))
            
            for article in sorted_articles[:max_attempts]:
                if len(articles_with_content) >= extract_count:
                    break  # We have enough
                
                print(f"  Extracting content from: {article.get('domain', 'unknown')} (reliable: {article.get('is_reliable', False)})")
                article_content = self._extract_article_content(article['url'])
                if article_content:
                    article['content'] = article_content
                    articles_with_content.append(article)
                    print(f"    ✅ Success ({len(articles_with_content)}/{extract_count})")
                else:
                    print(f"    ⚠️ Failed to extract content")
            
            if not articles_with_content:
                print(f"  ❌ Could not get content from any articles")
                print(f"     - Articles with content: {len([a for a in web_articles if a.get('content')])}")
                print(f"     - Articles without content: {len(articles_without_content)}")
                return []
            
            print(f"  ✅ Using {len(articles_with_content)} articles for similarity calculation")
            
            # Prepare corpus - ensure all content is strings
            corpus = [str(content).strip()]
            for art in articles_with_content:
                art_content = str(art.get('content', '')).strip()
                if art_content and len(art_content) > 50:
                    corpus.append(art_content)
            
            if len(corpus) < 2:
                print(f"  ❌ Not enough valid content for similarity calculation")
                return []
            
            # Calculate TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate similarities (use sklearn version, not embedding_utils version)
            similarities = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # Pair with articles
            results = [
                (article, float(sim)) 
                for article, sim in zip(articles_with_content, similarities)
            ]
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
        
        except Exception as e:
            print(f"⚠️ Error calculating similarities: {e}")
            return []
    
    def _calculate_score_formula(self, similarities: List[tuple]) -> Dict[str, Any]:
        """
        Calculate cross-check score using FORMULA-BASED approach (not arbitrary thresholds).
        
        Formula:
        - base_score: Maximum similarity found (0.0 to 1.0)
        - credibility_weight: Average credibility of sources (weighted by similarity)
        - diversity_boost: Source diversity multiplier (more sources = better)
        - final_score = base_score * credibility_weight * (1 + diversity_boost)
        
        Args:
            similarities: List of (article_dict, similarity_score) tuples
        
        Returns:
            Dict with calculated score and metrics
        """
        if not similarities:
            return {
                "score": 0.30,
                "status": "No Corroboration",
                "description": "No similar articles found for comparison."
            }
        
        # Extract metrics
        similarity_values = [sim for _, sim in similarities]
        max_similarity = float(np.max(similarity_values))
        avg_similarity = float(np.mean(similarity_values))
        
        # Separate credible and non-credible sources
        credible_similarities = [(art, sim) for art, sim in similarities if art.get('is_reliable', False)]
        non_credible_similarities = [(art, sim) for art, sim in similarities if not art.get('is_reliable', False)]
        
        # Calculate credibility-weighted similarity
        if credible_similarities:
            credible_sims = [sim for _, sim in credible_similarities]
            credible_max_similarity = float(np.max(credible_sims))
            credible_avg_similarity = float(np.mean(credible_sims))
            credible_sources_count = len(credible_similarities)
        else:
            credible_max_similarity = 0.0
            credible_avg_similarity = 0.0
            credible_sources_count = 0
        
        # Calculate source diversity
        unique_domains = set()
        for art, _ in similarities:
            domain = art.get('domain', '')
            if domain:
                unique_domains.add(domain)
        
        source_diversity = len(unique_domains)
        diversity_boost = min(0.3, source_diversity * 0.05)  # Max 30% boost
        
        # Calculate credibility weight
        if credible_similarities:
            # Weight by similarity (higher similarity from credible sources = higher weight)
            total_credible_sim = sum(sim for _, sim in credible_similarities)
            total_sim = sum(similarity_values)
            credibility_weight = 0.7 + (0.3 * (total_credible_sim / total_sim if total_sim > 0 else 0))
        else:
            credibility_weight = 0.5  # Lower weight if no credible sources
        
        # Base score is max similarity
        base_score = max_similarity
        
        # Final score formula
        final_score = base_score * credibility_weight * (1 + diversity_boost)
        final_score = min(1.0, final_score)  # Cap at 1.0
        
        # Determine status
        if final_score >= 0.7:
            status = "Strong Corroboration"
            description = f"Strong similarity ({final_score*100:.1f}%) found from {source_diversity} sources, including {credible_sources_count} credible sources."
        elif final_score >= 0.5:
            status = "Moderate Corroboration"
            description = f"Moderate similarity ({final_score*100:.1f}%) found from {source_diversity} sources."
        elif final_score >= 0.3:
            status = "Weak Corroboration"
            description = f"Limited similarity ({final_score*100:.1f}%) found. Limited corroboration available."
        else:
            status = "No Corroboration"
            description = "No significant similarity found. Cannot verify story through cross-checking."
        
        return {
            "score": round(final_score, 3),
            "base_score": round(base_score, 3),
            "credibility_weight": round(credibility_weight, 3),
            "diversity_boost": round(diversity_boost, 3),
            "credible_max_similarity": round(credible_max_similarity, 3),
            "credible_sources_count": credible_sources_count,
            "source_diversity": source_diversity,
            "status": status,
            "description": description
        }
    
    def _get_article_from_db(self, url: str) -> Optional[Dict]:
        """
        Get the full article record from database by URL.
        Returns the raw DynamoDB item if found, None otherwise.
        """
        if not self.use_database:
            return None
        
        try:
            # Scan database to find exact match
            response = self.articles_table.scan(
                FilterExpression='contains(source_url, :url)',
                ExpressionAttributeValues={':url': url}
            )
            
            articles = response.get('Items', [])
            
            # Continue scanning if needed
            while 'LastEvaluatedKey' in response:
                response = self.articles_table.scan(
                    FilterExpression='contains(source_url, :url)',
                    ExpressionAttributeValues={':url': url},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                articles.extend(response.get('Items', []))
            
            # Find exact URL match
            url_normalized = url.strip().lower().rstrip('/')
            for article in articles:
                article_url = str(article.get('source_url', '')).strip().lower().rstrip('/')
                if article_url == url_normalized or url_normalized in article_url or article_url in url_normalized:
                    return article
            
            return None
        except Exception as e:
            print(f"  [WARN] Error getting article from database: {e}")
            return None
    
    def _check_article_in_database(self, url: str) -> Optional[Dict]:
        """
        Check if the current article exists in the database.
        Returns the article dict if found, None otherwise.
        """
        if not self.use_database:
            return None
        
        try:
            # Normalize URL for comparison
            url_normalized = url.strip().lower().rstrip('/')
            
            # Scan database to find exact match
            response = self.articles_table.scan(
                FilterExpression='contains(source_url, :url)',
                ExpressionAttributeValues={':url': url}
            )
            
            articles = response.get('Items', [])
            
            # Continue scanning if needed
            while 'LastEvaluatedKey' in response:
                response = self.articles_table.scan(
                    FilterExpression='contains(source_url, :url)',
                    ExpressionAttributeValues={':url': url},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                articles.extend(response.get('Items', []))
            
            # Find exact URL match
            for article in articles:
                article_url = str(article.get('source_url', '')).strip().lower().rstrip('/')
                if article_url == url_normalized or url_normalized in article_url or article_url in url_normalized:
                    # Found exact match
                    article_content = article.get('content', '')
                    if article_content and len(str(article_content)) > 100:
                        article_domain = self._extract_domain(str(article.get('source_url', '')))
                        return {
                            'title': str(article.get('title', '')).strip(),
                            'url': str(article.get('source_url', '')).strip(),
                            'domain': article_domain,
                            'content': str(article_content).strip(),
                            'is_reliable': self._is_reliable_source(article_domain),
                            'published_at': str(article.get('published_at', ''))
                        }
            
            return None
        except Exception as e:
            print(f"  [WARN] Error checking article in database: {e}")
            return None
    
    def analyze(self, title: str, content: str, url: str) -> Dict[str, Any]:
        """
        Cross-check analysis with CREDIBILITY-WEIGHTED similarity.
        
        NEW PHILOSOPHY:
        - High similarity from credible sources = HIGH score (corroboration is good!)
        - Low similarity = LOW score (no credible corroboration)
        - Source diversity matters (multiple sources = better)
        - If article exists in database, calculate similarity to itself (should be 100%)
        
        UPDATED BEHAVIOR:
        - PRIMARY: Uses DynamoDB database (from Spark news ingestion pipeline)
        - FALLBACK: Uses web-based progressive search if database doesn't have enough articles
        - SPECIAL: If article is in database, includes it in similarity calculation for accurate scoring
        
        Args:
            title: Article title (for search)
            content: Article content (for similarity calculation)
            url: Article URL (to exclude from search)
        
        Returns:
            dict with 'score' (0-1 scale) and 'chart4_data'
        """
        try:
            # FIRST: Check if this article exists in the database
            # If it does, we should include it in similarity calculation for accurate scoring
            current_article_in_db = None
            if self.use_database:
                print(f"🔍 [CHECK] Checking if article exists in database...")
                current_article_in_db = self._check_article_in_database(url)
                if current_article_in_db:
                    print(f"  ✅ Article found in database - will include in similarity calculation")
                else:
                    print(f"  ℹ️ Article not found in database (or no content)")
            
            # Extract domain to exclude from search (but we'll include the article itself)
            exclude_domain = self._extract_domain(url)
            
            # PRIMARY: Search database using embeddings (if available)
            embedding_similarities = []
            if self.use_database and EMBEDDING_AVAILABLE:
                print(f"🔍 [PRIMARY] Searching database using vector embeddings...")
                # Don't exclude URL - we want to find the article itself
                embedding_similarities = self._search_database_embeddings(content, "", top_k=20)
                
                if embedding_similarities:
                    print(f"  ✅ Found {len(embedding_similarities)} similar articles using embeddings")
                    # Extract articles from similarity tuples
                    database_articles = [art for art, _ in embedding_similarities]
                    # Filter out the current article from display, but keep it for similarity calculation
                    database_articles = [art for art in database_articles if art.get('url', '') != url]
                else:
                    print(f"  ⚠️ No similar articles found via embeddings, trying fallback search")
                    database_articles = []
            else:
                database_articles = []
            
            # FALLBACK: Use traditional database search if embeddings not available or didn't find results
            if not database_articles and self.use_database:
                print(f"🔍 [FALLBACK] Searching database using traditional method...")
                # Don't exclude URL - we want to find the article itself
                database_articles = self._search_database(title, content, "", limit=50)
                # Filter out the current article from display
                database_articles = [art for art in database_articles if art.get('url', '') != url]
                
                if database_articles:
                    print(f"  ✅ Found {len(database_articles)} articles in database")
                else:
                    print(f"  ⚠️ No articles found in database, will use web search")
            
            # FALLBACK: Use web-based search if database doesn't have enough articles
            web_articles = []
            if len(database_articles) < 10:
                print(f"🔍 [FALLBACK] Progressive web search for cross-check...")
                web_articles = self._search_google_news_progressive(title, exclude_domain, target_count=20)
                
                if web_articles:
                    print(f"  ✅ Found {len(web_articles)} articles from web search")
            
            # Combine articles (database first, then web)
            all_articles = database_articles + web_articles
            
            if not all_articles:
                # No similar articles found - low score (no corroboration)
                return {
                    "score": 0.30,
                    "chart4_data": {
                        "message": "No similar articles found on web. Limited corroboration available.",
                        "points": []
                    },
                    "metrics": {
                        "similar_count": 0,
                        "total_compared": 0,
                        "avg_similarity": 0,
                        "max_similarity": 0,
                        "avg_similarity_pct": 0,
                        "max_similarity_pct": 0,
                        "credible_sources_count": 0,
                        "source_diversity": 0,
                        "status": "No Corroboration Found",
                        "description": "No similar articles found from database or web sources. Cannot verify story through cross-checking.",
                        "source": "none"
                    }
                }
            
            # SPECIAL CASE: If current article is in database, include it in similarity calculation
            # This ensures accurate scoring regardless of position in database
            if current_article_in_db:
                print(f"📊 [SPECIAL] Current article found in database - including in similarity calculation...")
                # Add current article to the list for similarity calculation
                all_articles_for_similarity = [current_article_in_db] + all_articles
            else:
                all_articles_for_similarity = all_articles
            
            # Calculate similarities
            print(f"📊 Calculating similarities...")
            
            # If we have embedding similarities, use them directly
            if embedding_similarities:
                similarities = embedding_similarities
                # Check if current article is in the similarities (self-match)
                # Use normalized URL comparison for robust matching
                current_url_normalized = url.strip().lower().rstrip('/')
                for art, sim in similarities:
                    art_url_normalized = str(art.get('url', '')).strip().lower().rstrip('/')
                    # Check for exact match or substring match (handles URL variations)
                    if (art_url_normalized == current_url_normalized or 
                        current_url_normalized in art_url_normalized or 
                        art_url_normalized in current_url_normalized):
                        if sim > 0.95:  # Very high similarity = same article
                            print(f"  ✅ Found self-match with {sim*100:.1f}% similarity - returning 100% score")
                            print(f"     Current URL: {url}")
                            print(f"     Matched URL: {art.get('url')}")
                            return {
                                "score": 1.0,
                                "chart4_data": {
                                    "message": "Article found in database with perfect match.",
                                    "points": []
                                },
                                "metrics": {
                                    "similar_count": 1,
                                    "total_compared": 1,
                                    "avg_similarity": round(sim, 3),
                                    "max_similarity": round(sim, 3),
                                    "avg_similarity_pct": round(sim * 100, 1),
                                    "max_similarity_pct": round(sim * 100, 1),
                                    "credible_sources_count": 1 if art.get('is_reliable') else 0,
                                    "source_diversity": 1,
                                    "status": "Perfect Match",
                                    "description": f"Article found in database with {sim*100:.1f}% similarity (self-match).",
                                    "source": "database"
                                }
                            }
                # Filter out current article from similarities for display (but it was used in calculation)
                similarities = [(art, sim) for art, sim in similarities 
                               if str(art.get('url', '')).strip().lower().rstrip('/') != current_url_normalized]
                print(f"  ✅ Using {len(similarities)} similarity scores from embeddings")
            else:
                # Fallback to TF-IDF similarity calculation
                extract_count = 10 if database_articles else 5
                similarities = self._calculate_similarities(content, all_articles_for_similarity, extract_count=extract_count)
                print(f"  ✅ Calculated {len(similarities)} similarity scores using TF-IDF")
                
                # Check for self-match (if current article was included)
                # This ensures position-independent scoring
                if current_article_in_db and similarities:
                    current_url_normalized = url.strip().lower().rstrip('/')
                    for art, sim in similarities:
                        art_url_normalized = str(art.get('url', '')).strip().lower().rstrip('/')
                        if (art_url_normalized == current_url_normalized or 
                            current_url_normalized in art_url_normalized or 
                            art_url_normalized in current_url_normalized):
                            if sim > 0.95:  # Very high similarity = same article
                                print(f"  ✅ Found self-match with {sim*100:.1f}% similarity - returning 100% score")
                                print(f"     Current URL: {url}")
                                print(f"     Matched URL: {art.get('url')}")
                                return {
                                    "score": 1.0,
                                    "chart4_data": {
                                        "message": "Article found in database with perfect match.",
                                        "points": []
                                    },
                                    "metrics": {
                                        "similar_count": 1,
                                        "total_compared": 1,
                                        "avg_similarity": round(sim, 3),
                                        "max_similarity": round(sim, 3),
                                        "avg_similarity_pct": round(sim * 100, 1),
                                        "max_similarity_pct": round(sim * 100, 1),
                                        "credible_sources_count": 1 if art.get('is_reliable') else 0,
                                        "source_diversity": 1,
                                        "status": "Perfect Match",
                                        "description": f"Article found in database with {sim*100:.1f}% similarity (self-match).",
                                        "source": "database"
                                    }
                                }
                            break
                
                # Filter out current article from similarities for display (but it was used in calculation)
                current_url_normalized = url.strip().lower().rstrip('/')
                similarities = [(art, sim) for art, sim in similarities 
                               if str(art.get('url', '')).strip().lower().rstrip('/') != current_url_normalized]
            
            if not similarities:
                # Could not extract content or calculate similarity
                return {
                    "score": 0.40,
                    "chart4_data": {
                        "message": "Unable to compare with articles.",
                        "points": []
                    },
                    "metrics": {
                        "similar_count": 0,
                        "total_compared": len(all_articles),
                        "avg_similarity": 0,
                        "max_similarity": 0,
                        "avg_similarity_pct": 0,
                        "max_similarity_pct": 0,
                        "credible_sources_count": 0,
                        "source_diversity": 0,
                        "status": "Comparison Failed",
                        "description": "Found articles but could not extract content for comparison.",
                        "source": "database" if database_articles else "web"
                    }
                }
            
            # Calculate score using FORMULA (not arbitrary thresholds)
            print(f"🧮 Calculating score using formula-based approach...")
            score_result = self._calculate_score_formula(similarities)
            
            # Calculate additional metrics for display
            similarity_values = [sim for _, sim in similarities]
            avg_similarity = float(np.mean(similarity_values))
            max_similarity = float(np.max(similarity_values))
            similar_count = sum(1 for sim in similarity_values if sim > 0.6)
            
            # Count credible sources
            credible_articles = [art for art, sim in similarities if art.get('is_reliable', False)]
            credible_similar_count = sum(1 for art, sim in similarities 
                                        if art.get('is_reliable', False) and sim > 0.6)
            
            # Chart 4 Data: Include all articles for display
            chart4_data = self._prepare_chart_data(all_articles, similarities)
            chart4_data["description"] = score_result["description"]
            
            # Determine source
            source_used = "database" if database_articles and len(database_articles) >= len(web_articles) else "web"
            
            return {
                "score": score_result["score"],
                "chart4_data": chart4_data,
                "metrics": {
                    "similar_count": similar_count,
                    "credible_similar_count": credible_similar_count,
                    "total_compared": len(similarities),
                    "total_found": len(all_articles),
                    "database_count": len(database_articles),
                    "web_count": len(web_articles),
                    "avg_similarity": round(avg_similarity, 3),
                    "max_similarity": round(max_similarity, 3),
                    "avg_similarity_pct": round(avg_similarity * 100, 1),
                    "max_similarity_pct": round(max_similarity * 100, 1),
                    "similarity_percentage": round(max_similarity * 100, 1),  # For backward compatibility
                    "credible_sources_count": score_result["credible_sources_count"],
                    "source_diversity": score_result["source_diversity"],
                    "credible_max_similarity": round(score_result["credible_max_similarity"], 3),
                    "credible_max_similarity_pct": round(score_result["credible_max_similarity"] * 100, 1),
                    "base_score": score_result["base_score"],
                    "credibility_weight": score_result["credibility_weight"],
                    "diversity_boost": score_result["diversity_boost"],
                    "status": score_result["status"],
                    "description": score_result["description"],
                    "source": source_used
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error in cross-check analysis: {e}")
            import traceback
            traceback.print_exc()
            return {
                "score": 0.50,
                "chart4_data": {
                    "message": f"Error during cross-check: {str(e)}",
                    "points": []
                },
                "metrics": {
                    "status": "Error",
                    "description": f"Cross-check analysis failed: {str(e)}"
                }
            }
    
    def _prepare_chart_data(self, web_articles: List[Dict], similarities: List[tuple]) -> Dict[str, Any]:
        """
        Prepare data for Chart 4 (scatter plot showing similarity analysis).
        Shows all articles found, with similarity data for those that were calculated.
        """
        if not web_articles:
            return {
                "message": "No similarity data available",
                "points": []
            }

        # Create a map of URL to similarity for articles we calculated
        similarity_map = {art['url']: sim for art, sim in similarities}

        # Prepare points for all articles
        data_points = []
        for article in web_articles[:50]:  # Limit to 50 for display
            url = article.get('url', '')
            similarity = similarity_map.get(url, None)
            
            # If we have similarity data, use it; otherwise estimate based on title similarity
            if similarity is not None:
                similarity_pct = round(similarity * 100, 1)
            else:
                # No similarity data (article not in top extracted)
                similarity_pct = None  # Will be shown as 0 or estimated
            
            # Assign credibility score based on source reliability
            if article.get('is_reliable', False):
                credibility = 0.85  # High credibility for reliable sources
            else:
                credibility = 0.50  # Medium credibility for unknown sources

            data_points.append({
                "x": round(similarity, 3) if similarity is not None else 0.0,
                "y": round(credibility, 3),
                "similarity_pct": similarity_pct if similarity_pct is not None else 0.0,
                "credibility_pct": round(credibility * 100, 1),
                "title": article.get('title', 'Unknown')[:60],
                "source": article.get('domain', 'Unknown'),
                "url": url,
                "is_reliable": article.get('is_reliable', False),
                "has_similarity_data": similarity is not None
            })

        return {
            "points": data_points,
            "x_label": "Content Similarity",
            "y_label": "Source Credibility",
            "total_points": len(data_points)
        }
