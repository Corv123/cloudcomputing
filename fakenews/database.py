# database.py
# Database operations for storing and retrieving articles

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import config

class ArticleDatabase:
    def __init__(self, db_name: str = config.DATABASE_NAME):
        self.db_name = db_name
        self.init_database()
    
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
                domain TEXT,
                analyzed_at TEXT,
                
                -- Core Analysis Scores
                language_score REAL,
                credibility_score REAL,
                cross_check_score REAL,
                sensationalism_bias_likelihood REAL,
                overall_score REAL,
                
                -- Additional Metrics
                word_count INTEGER,
                sensational_keyword_count INTEGER,
                known_source_classification TEXT,
                
                -- Chart Data (JSON strings)
                chart1_data TEXT,  -- Language Quality Bar Chart
                chart2_data TEXT,  -- Sentiment Pie Chart
                chart3_data TEXT,  -- Credibility Radar Chart
                chart4_data TEXT,  -- Similarity Map
                chart6_data TEXT,  -- Related Articles
                
                -- Detailed Analysis Data
                detailed_metrics TEXT,
                enhanced_info TEXT,
                word_frequency_data TEXT,
                sentiment_flow_data TEXT,
                
                -- Technical Data
                tfidf_vector TEXT
            )
        """)
        
        # Add missing columns for existing databases
        missing_columns = [
            'sensationalism_bias_likelihood REAL',
            'known_source_classification TEXT',
            'detailed_metrics TEXT',
            'enhanced_info TEXT',
            'word_frequency_data TEXT',
            'sentiment_flow_data TEXT',
            'chart1_data TEXT',
            'chart2_data TEXT',
            'chart3_data TEXT',
            'chart4_data TEXT',
            'chart6_data TEXT'
        ]
        
        for column in missing_columns:
            try:
                cursor.execute(f"ALTER TABLE articles ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        
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
                    url, title, content, source, published_at, category, domain, analyzed_at,
                    language_score, credibility_score, cross_check_score, sensationalism_bias_likelihood, overall_score,
                    word_count, sensational_keyword_count, known_source_classification,
                    chart1_data, chart2_data, chart3_data, chart4_data, chart6_data,
                    detailed_metrics, enhanced_info, word_frequency_data, sentiment_flow_data,
                    tfidf_vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article_data['url'],
                article_data['title'],
                article_data['content'],
                article_data.get('source'),
                article_data.get('published_at'),
                article_data.get('category'),
                article_data.get('domain', ''),
                datetime.now().isoformat(),
                article_data.get('language_score', 0),
                article_data.get('credibility_score', 0),
                article_data.get('cross_check_score', 0),
                article_data.get('sensationalism_bias_likelihood', 0.5),
                article_data.get('overall_score', 0),
                article_data.get('word_count', 0),
                article_data.get('sensational_keyword_count', 0),
                article_data.get('known_source_classification'),
                json.dumps(article_data.get('chart1_data', {})),
                json.dumps(article_data.get('chart2_data', {})),
                json.dumps(article_data.get('chart3_data', {})),
                json.dumps(article_data.get('chart4_data', {})),
                json.dumps(article_data.get('chart6_data', {})),
                json.dumps(article_data.get('detailed_metrics', {})),
                json.dumps(article_data.get('enhanced_info', {})),
                json.dumps(article_data.get('word_frequency_data', {})),
                json.dumps(article_data.get('sentiment_flow_data', {})),
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
        """Retrieve article by URL with JSON deserialization"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM articles WHERE url = ?", (url,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            article = dict(zip(columns, result))
            
            # Deserialize JSON fields
            json_fields = ['chart1_data', 'chart2_data', 'chart3_data', 'chart4_data', 'chart6_data',
                          'detailed_metrics', 'enhanced_info', 'word_frequency_data', 'sentiment_flow_data']
            
            for field in json_fields:
                if article.get(field):
                    try:
                        article[field] = json.loads(article[field])
                    except (json.JSONDecodeError, TypeError):
                        article[field] = {}
                else:
                    article[field] = {}
            
            return article
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