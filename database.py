# database.py
# Database operations for storing and retrieving articles

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import config

class ArticleDatabase:
    def __init__(self, db_name: str = config.DATABASE_NAME):
        self.db_name = db_name
        self.init_database()
        self.create_whois_cache_table()  # Initialize WHOIS cache table
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                published_at TEXT,
                category TEXT,
                language_score REAL,
                credibility_score REAL,
                cross_check_score REAL,
                overall_score REAL,
                domain TEXT,
                word_count INTEGER,
                sensational_keyword_count INTEGER,
                analyzed_at TEXT,
                tfidf_vector TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_article(self, article_data: Dict) -> int:
        """Insert a new article into the database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            # Ensure domain exists
            if 'domain' not in article_data or not article_data['domain']:
                from urllib.parse import urlparse
                article_data['domain'] = urlparse(article_data['url']).netloc.replace('www.', '')
            
            cursor.execute("""
                INSERT INTO articles (
                    url, title, content, source, published_at, category,
                    language_score, credibility_score, cross_check_score,
                    overall_score, domain, word_count, sensational_keyword_count,
                    analyzed_at, tfidf_vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article_data['url'],
                article_data['title'],
                article_data['content'],
                article_data.get('source'),
                article_data.get('published_at'),
                article_data.get('category'),
                article_data.get('language_score', 0),
                article_data.get('credibility_score', 0),
                article_data.get('cross_check_score', 0),
                article_data.get('overall_score', 0),
                article_data.get('domain', ''),
                article_data.get('word_count', 0),
                article_data.get('sensational_keyword_count', 0),
                datetime.now().isoformat(),
                article_data.get('tfidf_vector', '{}')
            ))
            
            article_id = cursor.lastrowid
            conn.commit()
            return article_id
        
        except sqlite3.IntegrityError:
            # Article URL already exists, return existing ID
            cursor.execute("SELECT id FROM articles WHERE url = ?", (article_data['url'],))
            result = cursor.fetchone()
            return result[0] if result else None
        
        except Exception as e:
            print(f"Database insert error: {e}")
            print(f"Article data keys: {article_data.keys()}")
            raise
        
        finally:
            conn.close()
    
    def get_article_by_url(self, url: str) -> Optional[Dict]:
        """Retrieve article by URL"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM articles WHERE url = ?", (url,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None
    
    def get_all_articles(self) -> List[Dict]:
        """Retrieve all articles from database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM articles")
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_articles_for_comparison(self, exclude_id: Optional[int] = None) -> List[Dict]:
        """Get articles for similarity comparison"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        if exclude_id:
            cursor.execute("""
                SELECT id, title, content, overall_score, credibility_score, 
                       source, tfidf_vector, url
                FROM articles 
                WHERE id != ?
            """, (exclude_id,))
        else:
            cursor.execute("""
                SELECT id, title, content, overall_score, credibility_score,
                       source, tfidf_vector, url
                FROM articles
            """)
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_article_count(self) -> int:
        """Get total number of articles in database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def clear_database(self):
        """Clear all articles from database (use with caution!)"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM articles")
        conn.commit()
        conn.close()

    def delete_article(self, article_id: int) -> bool:
        """Delete specific article by ID"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Check if article exists
            cursor.execute("SELECT id FROM articles WHERE id = ?", (article_id,))
            if not cursor.fetchone():
                conn.close()
                return False

            # Delete the article
            cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
            conn.commit()
            conn.close()

            print(f"✅ Deleted article ID {article_id}")
            return True

        except Exception as e:
            print(f"❌ Delete error: {e}")
            return False

    def create_whois_cache_table(self):
        """
        Create enhanced WHOIS cache table for domain age verification.

        Table schema:
        - domain: Primary key, clean domain name
        - age_years: Decimal age in years from WHOIS data
        - created_date: ISO date string (YYYY-MM-DD)
        - registrar: Registrar name from WHOIS (optional)
        - verified: Boolean flag (always 1 for cached WHOIS data)
        - cached_at: Timestamp for 24-hour freshness check

        Index on cached_at enables efficient stale cache cleanup
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
                    expires_date TEXT,
                    days_until_expiry INTEGER,
                    updated_date TEXT,
                    hosting_provider TEXT,
                    dnssec_enabled BOOLEAN DEFAULT 0,
                    domain_locked BOOLEAN DEFAULT 0,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_whois_cache_date
                ON whois_cache(cached_at)
            ''')

            conn.commit()
            conn.close()

            print("✅ Enhanced WHOIS cache table ready")

        except Exception as e:
            print(f"⚠️ WHOIS cache table error: {e}")