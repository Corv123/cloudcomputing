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
        # Skip if already scraped
        if url in self.scraped_urls:
            return None
        
        try:
            headers = self.get_headers()
            headers['Referer'] = 'https://www.google.com/'  # Make it look like we came from Google
            response = self.session.get(url, headers=headers, timeout=8)
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
                
                self.scraped_urls.add(url)
                return article
            
        except Exception as e:
            print(f"[ERROR] Error scraping {url}: {str(e)[:100]}")
        
        return None
    
    def find_article_links(self, base_url):
        """Find article links from a news site homepage or section page"""
        print(f"  -> Fetching homepage...")
        try:
            headers = self.get_headers()
            headers['Referer'] = 'https://www.google.com/'  # Make it look like we came from Google
            response = self.session.get(base_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            print(f"  -> Parsing links...")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links
            links = soup.find_all('a', href=True)
            article_urls = set()
            
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
            
            print(f"  -> Found {len(article_urls)} potential articles")
            return list(article_urls)
            
        except Exception as e:
            print(f"[ERROR] Error finding links on {base_url}: {str(e)[:100]}")
            return []
    
    def scrape_news_site(self, site_config):
        """Scrape a news site based on configuration"""
        name = site_config['name']
        url = site_config['url']
        
        print(f"\n{'='*60}")
        print(f"Scraping: {name}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        
        # Find article links
        article_urls = self.find_article_links(url)
        print(f"-> Found {len(article_urls)} potential article URLs")
        
        # Filter out already scraped URLs
        new_urls = [u for u in article_urls if u not in self.scraped_urls]
        print(f"-> {len(new_urls)} new URLs to scrape")
        
        # Scrape all available articles (no limit)
        urls_to_scrape = new_urls
        
        scraped_count = 0
        failed_count = 0
        start_time = time.time()
        
        for i, article_url in enumerate(urls_to_scrape, 1):
            article_start = time.time()
            print(f"  [{i}/{len(urls_to_scrape)}] Scraping: {article_url[:70]}...", end='')
            
            article = self.scrape_article(article_url)
            if article:
                self.articles.append(article)
                scraped_count += 1
                print(f" [OK] ({time.time() - article_start:.1f}s)")
            else:
                failed_count += 1
                print(f" [ERROR] ({time.time() - article_start:.1f}s)")
            
            # Be polite - add delay between requests
            time.sleep(0.3)
        
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
        'url': 'https://www.pbs.org/newshour'
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
