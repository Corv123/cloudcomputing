"""
News Article Scraper for Google Colab
Scrapes news articles and stores them in CSV format with duplicate prevention
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from urllib.parse import urljoin, urlparse
import time
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Try to import Google Colab files (only available in Colab)
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    files = None

class NewsScraperWithTracking:
    def __init__(self, csv_filename='news_articles.csv', tracking_filename='scraper_tracking.json'):
        self.csv_filename = csv_filename
        self.tracking_filename = tracking_filename
        self.scraped_urls = set()
        self.last_scrape_date = None
        self.articles = []
        
        # Create a session for better cookie handling
        self.session = requests.Session()
        
        # Thread safety for concurrent scraping
        self.articles_lock = Lock()
        self.scraped_urls_lock = Lock()
        
        # Configuration
        self.max_workers = 5  # Concurrent threads for scraping
        self.request_delay = 0.2  # Reduced delay for faster scraping
        self.max_pages_per_site = 3  # Max pagination pages to scrape
        
        # Load existing data if available
        self.load_tracking_data()
        self.load_existing_articles()
    
    def get_headers(self):
        """Get browser-like headers to avoid blocking"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
    
    def load_tracking_data(self):
        """Load tracking information from previous runs"""
        if os.path.exists(self.tracking_filename):
            with open(self.tracking_filename, 'r') as f:
                data = json.load(f)
                self.scraped_urls = set(data.get('scraped_urls', []))
                last_date = data.get('last_scrape_date')
                if last_date:
                    self.last_scrape_date = datetime.fromisoformat(last_date)
            print(f"[OK] Loaded tracking data: {len(self.scraped_urls)} URLs previously scraped")
            if self.last_scrape_date:
                print(f"[OK] Last scrape was on: {self.last_scrape_date.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("-> First run - no tracking data found")
    
    def save_tracking_data(self):
        """Save tracking information for next run"""
        data = {
            'scraped_urls': list(self.scraped_urls),
            'last_scrape_date': datetime.now().isoformat()
        }
        with open(self.tracking_filename, 'w') as f:
            json.dump(data, f)
        print(f"[OK] Saved tracking data: {len(self.scraped_urls)} total URLs tracked")
    
    def load_existing_articles(self):
        """Load existing articles from CSV if available"""
        if os.path.exists(self.csv_filename):
            df = pd.read_csv(self.csv_filename)
            print(f"[OK] Loaded {len(df)} existing articles from CSV")
            self.articles = df.to_dict('records')
        else:
            print("-> No existing CSV found - will create new file")
    
    def save_articles(self):
        """Save all articles to CSV"""
        if self.articles:
            df = pd.DataFrame(self.articles)
            # Ensure columns are in the correct order
            df = df[['title', 'content', 'source_url', 'published_at']]
            df.to_csv(self.csv_filename, index=False)
            print(f"[OK] Saved {len(self.articles)} articles to {self.csv_filename}")
        else:
            print("⚠ No articles to save")
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_article_content(self, soup):
        """Extract article content - target 1000 characters, accept whatever is available"""
        target_length = 1000
        content_parts = []
        
        # Strategy 1: Try to find article body/main content area
        article_selectors = [
            'article',
            '[role="article"]',
            '.article-body',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.story-body',
            '.content-body',
            'main article',
            '.main-content'
        ]
        
        article_body = None
        for selector in article_selectors:
            article_body = soup.select_one(selector)
            if article_body:
                break
        
        if article_body:
            # Extract all paragraphs from article body
            paragraphs = article_body.find_all('p')
            for p in paragraphs:
                text = p.get_text()
                if text and len(text.strip()) > 20:  # Skip very short paragraphs
                    content_parts.append(text.strip())
                    # Check if we've reached target length
                    combined = ' '.join(content_parts)
                    if len(combined) >= target_length:
                        break
        
        # If we have content from article body, use it
        if content_parts:
            content = ' '.join(content_parts)
            content = self.clean_text(content)
            # Truncate to target length if longer
            if len(content) > target_length:
                content = content[:target_length].rsplit(' ', 1)[0] + '...'  # Cut at word boundary
            return content
        
        # Strategy 2: Try to collect paragraphs from common content containers
        content_containers = soup.select('div[class*="content"], div[class*="article"], div[class*="story"], div[class*="post"]')
        for container in content_containers[:3]:  # Try first 3 containers
            paragraphs = container.find_all('p')
            for p in paragraphs:
                text = p.get_text()
                if text and len(text.strip()) > 20:
                    content_parts.append(text.strip())
                    combined = ' '.join(content_parts)
                    if len(combined) >= target_length:
                        break
            if content_parts and len(' '.join(content_parts)) >= target_length:
                break
        
        if content_parts:
            content = ' '.join(content_parts)
            content = self.clean_text(content)
            if len(content) > target_length:
                content = content[:target_length].rsplit(' ', 1)[0] + '...'
            return content
        
        # Strategy 3: Fallback to meta description
        meta_desc_selectors = [
            'meta[name="description"]',
            'meta[property="og:description"]',
            'meta[name="twitter:description"]',
            'meta[property="description"]'
        ]
        
        for selector in meta_desc_selectors:
            meta = soup.select_one(selector)
            if meta and meta.get('content'):
                content = meta.get('content')
                if len(content) > 50:
                    return self.clean_text(content)
        
        # Strategy 4: Get first few paragraphs
        paragraphs = soup.find_all('p', limit=10)
        for p in paragraphs:
            text = p.get_text()
            if text and len(text.strip()) > 20:
                content_parts.append(text.strip())
                combined = ' '.join(content_parts)
                if len(combined) >= target_length:
                    break
        
        if content_parts:
            content = ' '.join(content_parts)
            content = self.clean_text(content)
            if len(content) > target_length:
                content = content[:target_length].rsplit(' ', 1)[0] + '...'
            return content
        
        return ""
    
    def extract_date(self, soup, url):
        """Extract publication date from page"""
        # Try common date selectors
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publishdate"]',
            'meta[name="date"]',
            'time[datetime]',
            '.publish-date',
            '.post-date',
            '[itemprop="datePublished"]'
        ]
        
        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                date_str = elem.get('content') or elem.get('datetime') or elem.get_text()
                try:
                    # Try to parse various date formats
                    for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                        try:
                            date_obj = datetime.strptime(date_str[:19], fmt)
                            return date_obj.strftime('%Y-%m-%d')
                        except:
                            continue
                except:
                    pass
        
        # Default to current date if not found
        return datetime.now().strftime('%Y-%m-%d')
    
    def scrape_article(self, url):
        """Scrape a single article"""
        # Thread-safe check if already scraped
        with self.scraped_urls_lock:
            if url in self.scraped_urls:
                return None
        
        try:
            headers = self.get_headers()
            headers['Referer'] = 'https://www.google.com/'  # Make it look like we came from Google
            
            # Special handling for NYT (has paywall/anti-scraping)
            if 'nytimes.com' in url:
                # NYT often blocks scrapers, skip silently
                return None
            
            response = self.session.get(url, headers=headers, timeout=6)  # Reduced timeout
            
            # Skip 403 Forbidden (paywall/access denied) - don't waste time retrying
            if response.status_code == 403:
                return None
            
            # Skip 429 Rate Limited
            if response.status_code == 429:
                return None
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title = title.get_text() if title else "No title"
            title = self.clean_text(title)
            
            # Extract content
            content = self.extract_article_content(soup)
            
            # Extract date
            published_at = self.extract_date(soup, url)
            
            if content and len(content) > 50:  # Ensure we have meaningful content
                article = {
                    'title': title,
                    'content': content,  # Target 1000 chars, accepts whatever is available
                    'source_url': url,
                    'published_at': published_at
                }
                
                # URL tracking is handled in thread-safe wrapper
                return article
            
        except Exception as e:
            print(f"[ERROR] Error scraping {url}: {str(e)[:100]}")
        
        return None
    
    def parse_rss_content(self, content):
        """Parse RSS/Atom feed content with better error handling"""
        article_urls = set()
        
        try:
            # Try to fix common XML issues
            content_str = content.decode('utf-8', errors='ignore')
            # Remove invalid XML characters
            content_str = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content_str)
            
            root = ET.fromstring(content_str)
            
            # Handle RSS 2.0
            for item in root.findall('.//item'):
                link_elem = item.find('link')
                if link_elem is not None and link_elem.text:
                    url = link_elem.text.strip()
                    if url and url.startswith('http'):
                        # Clean URL - remove query parameters for better deduplication
                        parsed = urlparse(url)
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        article_urls.add(clean_url)
            
            # Handle Atom feeds
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                link_elem = entry.find('{http://www.w3.org/2005/Atom}link')
                if link_elem is not None:
                    url = link_elem.get('href', '').strip()
                    if url and url.startswith('http'):
                        # Clean URL
                        parsed = urlparse(url)
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        article_urls.add(clean_url)
            
            return list(article_urls)
        except ET.ParseError:
            # Try using BeautifulSoup as fallback for malformed XML
            try:
                soup = BeautifulSoup(content, 'xml')
                # Find all links in RSS items
                for item in soup.find_all('item'):
                    link = item.find('link')
                    if link and link.text:
                        url = link.text.strip()
                        if url and url.startswith('http'):
                            parsed = urlparse(url)
                            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                            article_urls.add(clean_url)
                # Find all links in Atom entries
                for entry in soup.find_all('entry'):
                    link = entry.find('link')
                    if link:
                        url = link.get('href', '').strip()
                        if url and url.startswith('http'):
                            parsed = urlparse(url)
                            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                            article_urls.add(clean_url)
                return list(article_urls)
            except:
                return []
        except Exception:
            return []
    
    def find_rss_feeds(self, base_url):
        """Find and parse RSS feeds from a news site"""
        article_urls = set()
        
        # Site-specific RSS paths
        domain = urlparse(base_url).netloc
        base_domain = f"{urlparse(base_url).scheme}://{domain}"
        
        # Common RSS paths
        rss_paths = [
            '/rss', '/feed', '/feeds', '/rss.xml', '/feed.xml',
            '/rss/all', '/feed/all', '/rss/news', '/feed/news'
        ]
        
        # Site-specific RSS paths (prioritize these)
        site_specific = {
            'www.bbc.com': ['/news/rss.xml'],
            'www.cnn.com': ['/rss/latest.rss', '/rss/edition.rss'],
            'www.nytimes.com': ['/services/xml/rss/nyt/HomePage.xml'],
            'www.apnews.com': ['/apf.rss'],
            'www.washingtonpost.com': ['/rss'],
            'www.channelnewsasia.com': ['/rss.xml'],
            'www.ft.com': ['/rss', '/rss/home'],
            'www.economist.com': ['/rss', '/rss/all'],
            'www.wsj.com': ['/rss', '/rss.xml'],
            'www.forbes.com': ['/rss', '/feed'],
            'www.politico.com': ['/rss', '/rss/all'],
            'time.com': ['/rss', '/feed'],
            'www.latimes.com': ['/rss', '/rss.xml'],
            'www.usatoday.com': ['/rss', '/rss/all'],
            'www.theatlantic.com': ['/rss', '/feed'],
            'www.axios.com': ['/rss', '/feed'],
            'www.independent.co.uk': ['/rss', '/rss/all'],
        }
        
        # Add site-specific paths if available (try these first)
        if domain in site_specific:
            rss_paths = site_specific[domain] + rss_paths
        
        # Try only first 3 paths to save time
        for path in rss_paths[:3]:
            try:
                rss_url = urljoin(base_domain, path)
                headers = self.get_headers()
                response = self.session.get(rss_url, headers=headers, timeout=8)
                
                if response.status_code == 429:
                    raise requests.exceptions.HTTPError("429 Rate Limited")
                
                if response.status_code == 200:
                    urls = self.parse_rss_content(response.content)
                    if urls:
                        article_urls.update(urls)
                        return list(article_urls)  # Return immediately when found
                        
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response.status_code == 429:
                    raise  # Re-raise 429 to be handled by caller
            except requests.exceptions.Timeout:
                continue  # Try next path on timeout
            except Exception:
                continue
        
        return list(article_urls)
    
    def find_sitemap_urls(self, base_url):
        """Find article URLs from sitemap.xml"""
        article_urls = set()
        domain = urlparse(base_url).netloc
        base_domain = f"{urlparse(base_url).scheme}://{domain}"
        
        sitemap_urls = [
            f"{base_domain}/sitemap.xml",
            f"{base_domain}/sitemap_index.xml",
            f"{base_domain}/sitemap-news.xml"
        ]
        
        for sitemap_url in sitemap_urls[:2]:  # Only try first 2 to save time
            try:
                headers = self.get_headers()
                response = self.session.get(sitemap_url, headers=headers, timeout=8)
                if response.status_code == 200:
                    try:
                        root = ET.fromstring(response.content)
                        # Handle sitemap index
                        for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                            loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc is not None:
                                # Recursively parse nested sitemap (limit to avoid long waits)
                                nested_urls = self.parse_sitemap(loc.text)
                                article_urls.update(nested_urls)
                                if len(article_urls) >= 50:  # Stop if we have enough
                                    break
                        
                        # Handle regular sitemap
                        urls = self.parse_sitemap(sitemap_url)
                        article_urls.update(urls)
                        
                        if article_urls:
                            return list(article_urls)  # Return immediately when found
                    except ET.ParseError:
                        continue
            except requests.exceptions.Timeout:
                continue
            except:
                continue
        
        return list(article_urls)
    
    def parse_sitemap(self, sitemap_url):
        """Parse a sitemap and extract article URLs"""
        article_urls = set()
        try:
            headers = self.get_headers()
            response = self.session.get(sitemap_url, headers=headers, timeout=8)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                # Extract URLs from sitemap (limit to avoid processing huge sitemaps)
                for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url')[:200]:
                    loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None:
                        url = loc.text.strip()
                        # Filter for article-like URLs
                        if self.is_article_url(url):
                            article_urls.add(url)
                            if len(article_urls) >= 100:  # Stop if we have enough
                                break
        except requests.exceptions.Timeout:
            pass
        except:
            pass
        return article_urls
    
    def is_article_url(self, url):
        """Quick check if URL looks like an article"""
        path_lower = urlparse(url).path.lower()
        skip_patterns = ['/video/', '/podcast/', '/gallery/', '/tag/', '/author/', '/search']
        if any(p in path_lower for p in skip_patterns):
            return False
        # Check for article indicators
        if re.search(r'/202[0-9]|/2030|\d{6,}', path_lower):
            return True
        if len([s for s in path_lower.split('/') if s]) >= 2:
            return True
        return False
    
    def find_article_links(self, base_url):
        """Find article links from a news site homepage or section page"""
        article_urls = set()
        
        try:
            headers = self.get_headers()
            headers['Referer'] = 'https://www.google.com/'
            # Reduced timeout to prevent hanging
            response = self.session.get(base_url, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                raise requests.exceptions.HTTPError(f"429 Rate Limited")
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links
            links = soup.find_all('a', href=True)
            
            domain = urlparse(base_url).netloc
            base_domain = '.'.join(domain.split('.')[-2:])  # Get main domain
            
            for link in links:
                href = link['href']
                
                # Skip empty, anchor, or javascript links
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # Convert relative URLs to absolute
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                
                # Only include links from the same base domain
                link_domain = '.'.join(parsed.netloc.split('.')[-2:]) if parsed.netloc else ''
                if link_domain != base_domain:
                    continue
                
                # Remove query parameters and fragments for deduplication
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                # Skip obvious non-article pages
                path_lower = parsed.path.lower()
                skip_patterns = [
                    '/video/', '/videos/', '/live/', '/livetv/', '/tv/',
                    '/podcast/', '/audio/', '/gallery/', '/photo/',
                    '/tag/', '/tags/', '/topic/', '/category/',
                    '/author/', '/writers/', '/journalists/',
                    '/search', '/login', '/subscribe', '/newsletter',
                    '/about', '/contact', '/privacy', '/terms', '/help',
                    '.pdf', '.jpg', '.png', '.gif', '.mp4', '.mp3',
                    '/weather/', '/horoscope/', '/crossword/', '/puzzle/'
                ]
                
                if any(pattern in path_lower for pattern in skip_patterns):
                    continue
                
                # IMPROVED: Multiple detection strategies
                is_article = False
                path_segments = [s for s in parsed.path.split('/') if s]
                
                # Strategy 1: Contains year (2020-2030) - very reliable
                if re.search(r'/202[0-9]|/2030', parsed.path):
                    is_article = True
                
                # Strategy 2: Has article keywords in path
                article_keywords = [
                    'article', 'story', 'news', 'post', 'blog',
                    'opinion', 'analysis', 'commentary', 'feature',
                    'world', 'business', 'politics', 'tech', 'sports',
                    'singapore', 'asia', 'international', 'lifestyle'
                ]
                if any(keyword in path_lower for keyword in article_keywords):
                    # Must have sufficient path depth (not just /news/)
                    if len(path_segments) >= 2:
                        is_article = True
                
                # Strategy 3: Contains numeric ID (6+ digits)
                if re.search(r'\d{6,}', parsed.path):
                    is_article = True
                
                # Strategy 4: Path looks like article slug (multi-word with hyphens)
                if len(path_segments) > 0:
                    last_segment = path_segments[-1]
                    # Check: has multiple hyphens, reasonable length, alphanumeric
                    if (last_segment.count('-') >= 2 and 
                        15 <= len(last_segment) <= 200 and
                        re.search(r'[a-z]', last_segment)):
                        is_article = True
                
                # Strategy 5: Deep path structure (3+ segments, one being long)
                if len(path_segments) >= 3:
                    # Check if any segment is long enough to be an article slug
                    if any(len(seg) > 20 for seg in path_segments):
                        is_article = True
                
                # Strategy 6: Date-like structure (YYYY/MM/DD or YYYY-MM-DD)
                if re.search(r'/\d{4}[/-]\d{1,2}[/-]\d{1,2}', parsed.path):
                    is_article = True
                
                if is_article:
                    article_urls.add(clean_url)
            
            return list(article_urls)
            
        except requests.exceptions.Timeout:
            raise  # Re-raise timeout to be handled by caller
        except requests.exceptions.HTTPError as e:
            if '429' in str(e):
                raise  # Re-raise 429 to be handled by caller
            return []
        except Exception as e:
            error_msg = str(e)[:100]
            if 'timeout' in error_msg.lower() or '429' in error_msg:
                raise  # Re-raise to be handled by caller
            return []
    
    def discover_all_article_urls(self, base_url):
        """Use multiple strategies to discover article URLs - optimized for speed with timeouts"""
        all_urls = set()
        rate_limited = False
        discovery_start = time.time()
        max_discovery_time = 30  # Maximum 30 seconds for entire discovery process
        
        print(f"  -> Discovering articles...")
        
        # Strategy 1: RSS Feeds (usually has most articles - try this first!)
        try:
            if time.time() - discovery_start > max_discovery_time:
                print(f"  -> Discovery timeout ({max_discovery_time}s), stopping...")
                return list(all_urls)
            
            time.sleep(0.3)  # Small delay before first request
            rss_urls = self.find_rss_feeds(base_url)
            if rss_urls:
                all_urls.update(rss_urls)
                print(f"  -> RSS: {len(rss_urls)} articles")
                
                # If RSS found enough articles, skip other methods to save time
                if len(rss_urls) >= 20:
                    print(f"  -> RSS found enough articles, skipping other discovery methods")
                    return list(all_urls)
        except requests.exceptions.Timeout:
            print(f"  -> RSS: Timeout, skipping...")
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response.status_code == 429:
                rate_limited = True
                print(f"  -> RSS: Rate limited (429)")
            else:
                print(f"  -> RSS failed: {str(e)[:50]}")
        except Exception as e:
            error_msg = str(e)[:100]
            if '429' in error_msg:
                rate_limited = True
                print(f"  -> RSS: Rate limited (429)")
            elif 'timeout' in error_msg.lower():
                print(f"  -> RSS: Timeout")
            else:
                print(f"  -> RSS failed: {str(e)[:50]}")
        
        # If rate limited, skip all other methods
        if rate_limited:
            print(f"  -> Site is rate limited, skipping further discovery")
            return list(all_urls)
        
        # Strategy 2: Sitemap.xml (only if RSS didn't find enough)
        if len(all_urls) < 20 and (time.time() - discovery_start) < max_discovery_time:
            try:
                time.sleep(0.5)
                sitemap_urls = self.find_sitemap_urls(base_url)
                if sitemap_urls:
                    all_urls.update(sitemap_urls)
                    print(f"  -> Sitemap: {len(sitemap_urls)} articles")
                    # If sitemap found enough, stop here
                    if len(all_urls) >= 30:
                        return list(all_urls)
            except requests.exceptions.Timeout:
                print(f"  -> Sitemap: Timeout, skipping...")
            except Exception as e:
                error_msg = str(e)[:100]
                if '429' in error_msg:
                    rate_limited = True
                    print(f"  -> Sitemap: Rate limited (429)")
                elif 'timeout' in error_msg.lower():
                    print(f"  -> Sitemap: Timeout")
                else:
                    print(f"  -> Sitemap failed: {str(e)[:50]}")
        
        # Strategy 3: Homepage link discovery (only if RSS/sitemap didn't find enough)
        if len(all_urls) < 30 and not rate_limited and (time.time() - discovery_start) < max_discovery_time:
            try:
                time.sleep(0.8)  # Delay before homepage
                homepage_urls = self.find_article_links(base_url)
                if homepage_urls:
                    all_urls.update(homepage_urls)
                    print(f"  -> Homepage: {len(homepage_urls)} articles")
                    # If homepage found enough, stop here
                    if len(all_urls) >= 50:
                        return list(all_urls)
            except requests.exceptions.Timeout:
                print(f"  -> Homepage: Timeout, skipping...")
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response.status_code == 429:
                    rate_limited = True
                    print(f"  -> Homepage: Rate limited (429)")
                else:
                    print(f"  -> Homepage failed: {str(e)[:50]}")
            except Exception as e:
                error_msg = str(e)[:100]
                if '429' in error_msg or 'timeout' in error_msg.lower():
                    rate_limited = True
                    if '429' in error_msg:
                        print(f"  -> Homepage: Rate limited (429)")
                    else:
                        print(f"  -> Homepage: Timeout, skipping...")
                else:
                    print(f"  -> Homepage failed: {error_msg[:50]}")
        
        # Strategy 4: Section pages (ONLY if we have very few URLs and not rate limited)
        if len(all_urls) < 20 and not rate_limited and (time.time() - discovery_start) < max_discovery_time:
            domain = urlparse(base_url).netloc
            base_domain = f"{urlparse(base_url).scheme}://{domain}"
            section_paths = ['/news']
            
            for section in section_paths[:1]:  # Only try 1 section max
                if (time.time() - discovery_start) > max_discovery_time:
                    print(f"  -> Discovery timeout, stopping section discovery...")
                    break
                try:
                    time.sleep(1.0)
                    section_url = urljoin(base_domain, section)
                    section_urls = self.find_article_links(section_url)
                    if section_urls:
                        all_urls.update(section_urls)
                        print(f"  -> Section {section}: {len(section_urls)} articles")
                except requests.exceptions.Timeout:
                    print(f"  -> Section {section}: Timeout, stopping...")
                    break
                except Exception as e:
                    error_msg = str(e)[:100]
                    if '429' in error_msg or 'timeout' in error_msg.lower():
                        print(f"  -> Section {section}: Rate limited/timeout, stopping...")
                        break
                    continue
        
        elapsed = time.time() - discovery_start
        print(f"  -> Total: {len(all_urls)} unique article URLs (took {elapsed:.1f}s)")
        return list(all_urls)
    
    def scrape_article_thread_safe(self, url):
        """Thread-safe wrapper for scrape_article"""
        article = self.scrape_article(url)
        if article:
            with self.articles_lock:
                self.articles.append(article)
            with self.scraped_urls_lock:
                self.scraped_urls.add(url)
            return True
        return False
    
    def scrape_news_site(self, site_config):
        """Scrape a news site based on configuration"""
        name = site_config['name']
        url = site_config['url']
        
        # Skip NYT entirely (has aggressive paywall/anti-scraping)
        if 'nytimes.com' in url or 'NYT' in name or 'New York Times' in name:
            print(f"\n{'='*60}")
            print(f"Skipping: {name} (paywall/anti-scraping)")
            print(f"{'='*60}")
            return
        
        print(f"\n{'='*60}")
        print(f"Scraping: {name}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        
        # Use improved discovery method
        try:
            article_urls = self.discover_all_article_urls(url)
        except Exception as e:
            print(f"[ERROR] Discovery failed for {name}: {str(e)[:100]}")
            return
        
        print(f"-> Found {len(article_urls)} potential article URLs")
        
        # Filter out already scraped URLs and clean URLs
        with self.scraped_urls_lock:
            new_urls = []
            for u in article_urls:
                # Skip NYT URLs (safety check even if site was skipped)
                if 'nytimes.com' in u:
                    continue
                # Clean URL (remove query parameters for better deduplication)
                clean_url = urlparse(u)._replace(query='', fragment='').geturl()
                if clean_url not in self.scraped_urls:
                    new_urls.append(clean_url)
        
        print(f"-> {len(new_urls)} new URLs to scrape")
        
        if not new_urls:
            print(f"-> No new articles to scrape for {name}")
            return
        
        # Skip sites that are known to be rate-limited or problematic
        domain = urlparse(url).netloc
        if domain in ['www.apnews.com', 'apnews.com']:
            # Check if we got rate limited during discovery
            if len(new_urls) == 0 or len(article_urls) < 5:
                print(f"-> Site appears rate-limited, skipping scraping to avoid further blocks")
                return
        
        # Skip if we got very few URLs (likely rate limited or blocked)
        if len(new_urls) < 3 and len(article_urls) < 10:
            print(f"-> Very few articles found, likely rate-limited or blocked. Skipping...")
            return
        
        # Scrape articles concurrently for speed
        scraped_count = 0
        failed_count = 0
        start_time = time.time()
        max_scraping_time = 120  # Max 2 minutes per site
        
        print(f"-> Scraping {len(new_urls)} articles with {self.max_workers} concurrent workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {executor.submit(self.scrape_article_thread_safe, url): url 
                           for url in new_urls}
            
            # Process completed tasks with timeout
            for i, future in enumerate(as_completed(future_to_url), 1):
                # Check if we've exceeded max time
                if time.time() - start_time > max_scraping_time:
                    print(f"-> Scraping timeout ({max_scraping_time}s), stopping...")
                    # Cancel remaining tasks
                    for f in future_to_url:
                        f.cancel()
                    break
                
                url = future_to_url[future]
                try:
                    # Use shorter timeout to prevent hanging
                    success = future.result(timeout=8)  # 8s timeout per article
                    if success:
                        scraped_count += 1
                        print(f"  [{i}/{len(new_urls)}] [OK] {url[:60]}...")
                    else:
                        failed_count += 1
                        # Don't print every error to reduce noise
                        if failed_count <= 3 or i % 10 == 0:
                            print(f"  [{i}/{len(new_urls)}] [ERROR] {url[:60]}...")
                except TimeoutError:
                    failed_count += 1
                    if failed_count <= 3:
                        print(f"  [{i}/{len(new_urls)}] [TIMEOUT] {url[:50]}...")
                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)[:50]
                    # Only print first few errors or every 10th
                    if failed_count <= 3 or i % 10 == 0:
                        if '403' in error_msg or 'Forbidden' in error_msg:
                            print(f"  [{i}/{len(new_urls)}] [SKIP] Paywall (403): {url[:50]}...")
                        elif '429' in error_msg or 'Too Many Requests' in error_msg:
                            print(f"  [{i}/{len(new_urls)}] [SKIP] Rate limited (429): {url[:50]}...")
                        else:
                            print(f"  [{i}/{len(new_urls)}] [ERROR] {url[:50]}... {error_msg}")
        
        elapsed = time.time() - start_time
        print(f"[OK] Successfully scraped {scraped_count} articles from {name} in {elapsed:.1f}s")
        if failed_count > 0:
            print(f"  (Failed: {failed_count} articles)")
    
    def run(self, news_sites):
        """Run the scraper on all configured news sites"""
        print(f"\n{'#'*60}")
        print(f"Starting News Scraper")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")
        
        if self.last_scrape_date:
            print(f"\n-> This is an UPDATE run (collecting new articles since last run)")
        else:
            print(f"\n-> This is the FIRST run (collecting historical articles)")
        
        initial_count = len(self.articles)
        
        for site in news_sites:
            try:
                self.scrape_news_site(site)
            except Exception as e:
                print(f"[ERROR] Error with {site['name']}: {str(e)}")
        
        new_articles = len(self.articles) - initial_count
        
        print(f"\n{'='*60}")
        print(f"Scraping Complete!")
        print(f"New articles collected: {new_articles}")
        print(f"Total articles in database: {len(self.articles)}")
        print(f"{'='*60}\n")
        
        # Save data
        self.save_articles()
        self.save_tracking_data()
        
        # Download files (only in Colab)
        if IN_COLAB:
            print("\n-> Downloading files...")
            files.download(self.csv_filename)
            files.download(self.tracking_filename)
            print("[OK] Files downloaded to your computer")
        else:
            print(f"\n[OK] Files saved locally:")
            print(f"  - {self.csv_filename}")
            print(f"  - {self.tracking_filename}")


# ============================================================================
# CONFIGURATION - ADD YOUR NEWS SITES HERE
# ============================================================================

NEWS_SITES = [
    {
        'name': 'BBC News',
        'url': 'https://www.bbc.com/news'
    },
    {
        'name': 'CNN',
        'url': 'https://www.cnn.com'
    },
    {
        'name': 'Associated Press (AP)',
        'url': 'https://www.apnews.com'
    },
    {
        'name': 'Al Jazeera',
        'url': 'https://www.aljazeera.com'
    },
    {
        'name': 'The New York Times (NYT)',
        'url': 'https://www.nytimes.com'
    },
    {
        'name': 'The Guardian',
        'url': 'https://www.theguardian.com'
    },
    {
        'name': 'The Washington Post',
        'url': 'https://www.washingtonpost.com'
    },
    {
        'name': 'PBS NewsHour',
        'url': 'https://www.pbs.org/newshour/'
    },
    {
        'name': 'Business Times Singapore',
        'url': 'https://www.businesstimes.com.sg/'
    },
    {
        'name': 'Today Online',
        'url': 'https://www.todayonline.com/'
    },
    {
        'name': 'The Straits Times',
        'url': 'https://www.straitstimes.com/singapore'
    },
    {
        'name': 'Channel News Asia',
        'url': 'https://www.channelnewsasia.com/'
    },
    {
        'name': 'Yahoo News Singapore',
        'url': 'https://sg.news.yahoo.com/'
    },
    {
        'name': 'Financial Times',
        'url': 'https://www.ft.com'
    },
    {
        'name': 'The Economist',
        'url': 'https://www.economist.com'
    },
    {
        'name': 'The Wall Street Journal (WSJ)',
        'url': 'https://www.wsj.com'
    },
    {
        'name': 'Forbes',
        'url': 'https://www.forbes.com'
    },
    {
        'name': 'Politico',
        'url': 'https://www.politico.com'
    },
    {
        'name': 'Time Magazine',
        'url': 'https://time.com'
    },
    {
        'name': 'Los Angeles Times',
        'url': 'https://www.latimes.com'
    },
    {
        'name': 'USA Today',
        'url': 'https://www.usatoday.com'
    },
    {
        'name': 'The Atlantic',
        'url': 'https://www.theatlantic.com'
    },
    {
        'name': 'Axios',
        'url': 'https://www.axios.com'
    },
    {
        'name': 'The Independent (UK)',
        'url': 'https://www.independent.co.uk'
    },
]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Upload previous files if they exist (for subsequent runs - Colab only)
    if IN_COLAB:
        print("-> If you have previous tracking files, upload them now...")
        print("-> Otherwise, press cancel to start fresh\n")
        
        try:
            uploaded = files.upload()
            print("[OK] Previous files uploaded")
        except:
            print("-> No previous files uploaded - starting fresh")
    else:
        print("-> Running locally - will use existing tracking files if available\n")
    
    # Create scraper instance
    scraper = NewsScraperWithTracking()
    
    # Run the scraper
    scraper.run(NEWS_SITES)
    
    # Display sample of collected data
    if scraper.articles:
        print("\n" + "="*60)
        print("SAMPLE OF COLLECTED DATA (first 3 articles):")
        print("="*60)
        df = pd.DataFrame(scraper.articles)
        print(df.head(3).to_string())

        print("\n" + "="*60)
        print("SAMPLE OF COLLECTED DATA (first 3 articles):")
        print("="*60)
        df = pd.DataFrame(scraper.articles)
        print(df.head(3).to_string())

        print("\n" + "="*60)
        print("SAMPLE OF COLLECTED DATA (first 3 articles):")
        print("="*60)
        df = pd.DataFrame(scraper.articles)
        print(df.head(3).to_string())

        print("\n" + "="*60)
        print("SAMPLE OF COLLECTED DATA (first 3 articles):")
        print("="*60)
        df = pd.DataFrame(scraper.articles)
        print(df.head(3).to_string())
