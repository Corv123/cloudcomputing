"""
Stage 1: News Scraping
Uses news_scraper.py to collect new articles from news sites
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from typing import List, Dict, Optional
import sys
import os
import json
from datetime import datetime
from urllib.parse import urlparse

# Add path to import news_scraper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../analyzers'))
try:
    from news_scraper import NewsScraperWithTracking, NEWS_SITES
    SCRAPER_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] news_scraper not available: {e}")
    SCRAPER_AVAILABLE = False
    NewsScraperWithTracking = None
    NEWS_SITES = []


def scrape_new_articles(
    spark: SparkSession,
    tracking_file: str = "scraper_tracking.json",
    csv_file: str = "news_articles.csv",
    use_s3: bool = False,
    s3_tracking_path: Optional[str] = None,
    s3_csv_path: Optional[str] = None
) -> List[Dict]:
    """
    Scrape new articles using news_scraper.py
    
    Args:
        spark: SparkSession instance
        tracking_file: Local path to tracking JSON file
        csv_file: Local path to existing CSV file
        use_s3: Whether to read/write from S3
        s3_tracking_path: S3 path for tracking file (if use_s3=True)
        s3_csv_path: S3 path for CSV file (if use_s3=True)
    
    Returns:
        List of article dictionaries with: title, content, source_url, published_at
    """
    print("=" * 80)
    print("STAGE 1: NEWS SCRAPING")
    print("=" * 80)
    
    if not SCRAPER_AVAILABLE:
        print("[ERROR] news_scraper module not available")
        return []
    
    # Download tracking and CSV from S3 if needed
    if use_s3:
        import boto3
        s3_client = boto3.client('s3')
        
        # Download tracking file
        if s3_tracking_path:
            bucket, key = s3_tracking_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.download_file(bucket, key, tracking_file)
                print(f"[OK] Downloaded tracking file from S3")
            except Exception as e:
                print(f"[WARN] Could not download tracking file: {e}")
                # Create empty tracking file
                with open(tracking_file, 'w') as f:
                    json.dump({'scraped_urls': [], 'last_scrape_date': None}, f)
        
        # Download CSV file
        if s3_csv_path:
            bucket, key = s3_csv_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.download_file(bucket, key, csv_file)
                print(f"[OK] Downloaded CSV file from S3")
            except Exception as e:
                print(f"[WARN] Could not download CSV file: {e}")
                # Will create new CSV
    
    # Initialize scraper
    scraper = NewsScraperWithTracking(
        csv_filename=csv_file,
        tracking_filename=tracking_file
    )
    
    # Run scraper
    print(f"[INFO] Starting news scraping...")
    print(f"[INFO] Sites to scrape: {len(NEWS_SITES)}")
    
    scraper.run(NEWS_SITES)
    
    # Get newly scraped articles
    new_articles = scraper.articles
    
    print(f"[OK] Scraped {len(new_articles)} total articles")
    
    # Upload tracking and CSV back to S3 if needed
    if use_s3:
        import boto3
        s3_client = boto3.client('s3')
        
        # Upload tracking file
        if s3_tracking_path:
            bucket, key = s3_tracking_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.upload_file(tracking_file, bucket, key)
                print(f"[OK] Uploaded tracking file to S3")
            except Exception as e:
                print(f"[WARN] Could not upload tracking file: {e}")
        
        # Upload CSV file
        if s3_csv_path:
            bucket, key = s3_csv_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.upload_file(csv_file, bucket, key)
                print(f"[OK] Uploaded CSV file to S3")
            except Exception as e:
                print(f"[WARN] Could not upload CSV file: {e}")
    
    print("=" * 80)
    print()
    
    return new_articles


def load_existing_articles(
    spark: SparkSession,
    csv_path: str = "news_articles.csv",
    use_s3: bool = False
):
    """
    Load existing articles from CSV file
    
    Args:
        spark: SparkSession instance
        csv_path: Path to CSV file (local or S3)
        use_s3: Whether reading from S3
    
    Returns:
        Spark DataFrame with articles
    """
    print("  Loading existing articles from CSV...")
    
    try:
        if use_s3:
            df = spark.read.option("encoding", "utf-8").csv(
                csv_path,
                header=True,
                inferSchema=True
            )
        else:
            if not os.path.exists(csv_path):
                print(f"    [WARN] CSV file not found: {csv_path}")
                # Return empty DataFrame with correct schema
                schema = StructType([
                    StructField("title", StringType(), True),
                    StructField("content", StringType(), True),
                    StructField("source_url", StringType(), True),
                    StructField("published_at", StringType(), True)
                ])
                return spark.createDataFrame([], schema)
            
            df = spark.read.option("encoding", "utf-8").csv(
                csv_path,
                header=True,
                inferSchema=True
            )
        
        # Ensure required columns exist
        required_cols = ['title', 'content', 'source_url', 'published_at']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"    [ERROR] Missing columns: {missing_cols}")
            return None
        
        print(f"    [OK] Loaded {df.count()} existing articles")
        return df
        
    except Exception as e:
        print(f"    [ERROR] Failed to load CSV: {e}")
        return None

