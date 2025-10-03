# extractors/article_extractor.py
# Extracts article content from URLs

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import config

class ArticleExtractor:
    """Extracts article data from URLs"""
    
    def extract(self, url: str) -> dict:
        """
        Extract article from URL
        
        Args:
            url: Article URL
            
        Returns:
            dict with article data
        """
        # Try NewsAPI if key is available
        if config.NEWSAPI_KEY:
            article = self._extract_with_newsapi(url)
            if article:
                return article
        
        # Fallback to web scraping
        return self._extract_with_scraping(url)
    
    def _extract_with_newsapi(self, url: str) -> dict:
        """Extract using NewsAPI"""
        try:
            api_url = f"https://newsapi.org/v2/everything?q={url}&apiKey={config.NEWSAPI_KEY}"
            response = requests.get(api_url, timeout=10)
            data = response.json()
            
            if data.get('articles') and len(data['articles']) > 0:
                article = data['articles'][0]
                domain = urlparse(url).netloc.replace('www.', '')
                return {
                    'url': url,
                    'title': article.get('title', ''),
                    'content': article.get('content', article.get('description', '')),
                    'source': article.get('source', {}).get('name'),
                    'published_at': article.get('publishedAt'),
                    'category': None,
                    'domain': domain
                }
        except Exception as e:
            print(f"NewsAPI extraction failed: {e}")
        return None
    
    def _extract_with_scraping(self, url: str) -> dict:
        """Extract using web scraping"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract domain
        domain = urlparse(url).netloc.replace('www.', '')
        
        # Extract title
        title = ''
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Extract content
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text()) > 50])
        
        # Extract source
        source = ''
        meta_source = soup.find('meta', property='og:site_name')
        if meta_source:
            source = meta_source.get('content', '')
        else:
            source = domain
        
        # Extract published date
        published_at = ''
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta:
            published_at = date_meta.get('content', '')
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'source': source,
            'published_at': published_at,
            'category': None,
            'domain': domain
        }