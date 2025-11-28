"""
AWS Lambda handler for News Ingestion Pipeline
Runs the pipeline when triggered by EventBridge
NOTE: This is a Lambda-compatible version (no Spark dependencies)
"""

import json
import os
import sys
from datetime import datetime
from urllib.parse import urlparse
import hashlib

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../analyzers'))

# Import boto3
try:
    import boto3
    from decimal import Decimal
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None


def lambda_handler(event, context):
    """
    Lambda handler for news ingestion pipeline
    
    Args:
        event: EventBridge event (or manual invocation)
        context: Lambda context
    
    Returns:
        dict with status and results
    """
    print("=" * 80)
    print("NEWS INGESTION PIPELINE - LAMBDA HANDLER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Get configuration from environment variables
    tracking_file = os.environ.get('TRACKING_FILE', '/tmp/scraper_tracking.json')
    csv_file = os.environ.get('CSV_FILE', '/tmp/news_articles.csv')
    use_s3 = os.environ.get('USE_S3', 'true').lower() == 'true'
    s3_tracking_path = os.environ.get('S3_TRACKING_PATH')
    s3_csv_path = os.environ.get('S3_CSV_PATH')
    dynamodb_table = os.environ.get('DYNAMODB_TABLE', 'fakenews-scraped-news')
    # Get region from environment or Lambda context
    try:
        region = os.environ.get('REGION') or (context.aws_region if context and hasattr(context, 'aws_region') else 'ap-southeast-2')
    except:
        region = os.environ.get('REGION', 'ap-southeast-2')
    
    try:
        # Stage 1: Scrape new articles using news_scraper
        print("Stage 1: Scraping news articles...")
        
        # Import news_scraper directly (no Spark dependency)
        try:
            import news_scraper
            scraper = news_scraper.NewsScraperWithTracking(
                csv_filename=csv_file,
                tracking_filename=tracking_file
            )
        except ImportError as e:
            print(f"[ERROR] Could not import news_scraper: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': f'Failed to import news_scraper: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                })
            }
        
        # Download from S3 if needed
        if use_s3 and s3_tracking_path and BOTO3_AVAILABLE:
            s3_client = boto3.client('s3')
            bucket, key = s3_tracking_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.download_file(bucket, key, tracking_file)
                print(f"[OK] Downloaded tracking file from S3")
            except s3_client.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['404', 'NoSuchBucket', 'NoSuchKey']:
                    print(f"[WARN] S3 bucket '{bucket}' or file does not exist. Using local file.")
                    # Create empty tracking file
                    with open(tracking_file, 'w') as f:
                        json.dump({'scraped_urls': [], 'last_scrape_date': None}, f)
                else:
                    print(f"[WARN] Could not download tracking file: {e}")
                    # Create empty tracking file
                    with open(tracking_file, 'w') as f:
                        json.dump({'scraped_urls': [], 'last_scrape_date': None}, f)
            except Exception as e:
                print(f"[WARN] Could not download tracking file: {e}")
                # Create empty tracking file
                with open(tracking_file, 'w') as f:
                    json.dump({'scraped_urls': [], 'last_scrape_date': None}, f)
        
        if use_s3 and s3_csv_path and BOTO3_AVAILABLE:
            s3_client = boto3.client('s3')
            bucket, key = s3_csv_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.download_file(bucket, key, csv_file)
                print(f"[OK] Downloaded CSV file from S3")
            except s3_client.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['404', 'NoSuchBucket', 'NoSuchKey']:
                    print(f"[WARN] S3 bucket '{bucket}' or file does not exist. Using local file.")
                else:
                    print(f"[WARN] Could not download CSV file: {e}")
            except Exception as e:
                print(f"[WARN] Could not download CSV file: {e}")
        
        # Run scraper (limit to fewer sites to avoid timeout)
        print(f"[INFO] Running scraper on {len(news_scraper.NEWS_SITES)} news sites...")
        # Limit to first 5 sites to avoid Lambda timeout
        limited_sites = news_scraper.NEWS_SITES[:5]
        scraper.run(limited_sites)
        scraped_articles = scraper.articles
        
        print(f"[OK] Scraped {len(scraped_articles)} new articles")
        
        # Upload back to S3 if needed
        if use_s3 and s3_tracking_path and BOTO3_AVAILABLE:
            s3_client = boto3.client('s3')
            bucket, key = s3_tracking_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.upload_file(tracking_file, bucket, key)
                print(f"[OK] Uploaded tracking file to S3")
            except s3_client.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['404', 'NoSuchBucket']:
                    print(f"[WARN] S3 bucket '{bucket}' does not exist. Skipping upload.")
                else:
                    print(f"[WARN] Could not upload tracking file: {e}")
            except Exception as e:
                print(f"[WARN] Could not upload tracking file: {e}")
        
        if use_s3 and s3_csv_path and BOTO3_AVAILABLE:
            s3_client = boto3.client('s3')
            bucket, key = s3_csv_path.replace("s3://", "").split("/", 1)
            try:
                s3_client.upload_file(csv_file, bucket, key)
                print(f"[OK] Uploaded CSV file to S3")
            except s3_client.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['404', 'NoSuchBucket']:
                    print(f"[WARN] S3 bucket '{bucket}' does not exist. Skipping upload.")
                else:
                    print(f"[WARN] Could not upload CSV file: {e}")
            except Exception as e:
                print(f"[WARN] Could not upload CSV file: {e}")
        
        # Stage 2: Load existing articles and deduplicate
        print("Stage 2: Loading existing articles and deduplicating...")
        existing_articles = []
        if os.path.exists(csv_file):
            try:
                import csv
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_articles = list(reader)
                print(f"[OK] Loaded {len(existing_articles)} existing articles")
            except Exception as e:
                print(f"[WARN] Could not load existing CSV: {e}")
        
        # Combine and deduplicate
        all_articles = existing_articles + scraped_articles
        seen_urls = set()
        cleaned_articles = []
        
        for article in all_articles:
            url = article.get('source_url', '')
            if not url:
                continue
            
            # Clean URL
            parsed = urlparse(url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
            
            # Generate ID
            article_id = hashlib.md5(clean_url.encode()).hexdigest()
            
            if clean_url not in seen_urls:
                content = article.get('content', '')
                if content and len(content) > 50:  # Minimum content length
                    seen_urls.add(clean_url)
                    cleaned_article = {
                        'id': article_id,
                        'title': str(article.get('title', '')).strip(),
                        'content': str(content).strip(),
                        'source_url': clean_url,
                        'domain': parsed.netloc.replace('www.', '').lower(),
                        'published_at': str(article.get('published_at', '')),
                        'ingested_at': datetime.now().isoformat(),
                        'article_type': 'scraped_news',
                        'word_count': len(content.split())
                    }
                    cleaned_articles.append(cleaned_article)
        
        print(f"[OK] Cleaned: {len(cleaned_articles)} unique articles")
        
        # Stage 3: Store to DynamoDB
        print("Stage 3: Storing to DynamoDB...")
        stored_count = 0
        failed_count = 0
        
        if BOTO3_AVAILABLE:
            dynamodb = boto3.resource('dynamodb', region_name=region)
            table = dynamodb.Table(dynamodb_table)
            
            # Process in batches
            batch_size = 25
            for i in range(0, len(cleaned_articles), batch_size):
                batch = cleaned_articles[i:i + batch_size]
                
                try:
                    with table.batch_writer() as batch_writer:
                        for article in batch:
                            try:
                                item = {
                                    'id': article['id'],
                                    'title': article['title'],
                                    'content': article['content'],
                                    'source_url': article['source_url'],
                                    'domain': article['domain'],
                                    'published_at': article['published_at'],
                                    'ingested_at': article['ingested_at'],
                                    'article_type': article['article_type'],
                                    'word_count': Decimal(str(article['word_count']))
                                }
                                batch_writer.put_item(Item=item)
                                stored_count += 1
                            except Exception as e:
                                failed_count += 1
                                print(f"  [WARN] Failed to store article {article.get('id', 'unknown')}: {e}")
                except Exception as e:
                    print(f"  [ERROR] Batch write failed: {e}")
                    failed_count += len(batch)
        else:
            print("[ERROR] boto3 not available - cannot store to DynamoDB")
        
        print(f"[OK] Stored {stored_count} articles to DynamoDB")
        if failed_count > 0:
            print(f"[WARN] Failed to store {failed_count} articles")
        
        print()
        print("=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'News ingestion pipeline completed successfully',
                'articles_scraped': len(scraped_articles),
                'articles_stored': stored_count,
                'articles_failed': failed_count,
                'total_articles': len(cleaned_articles),
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Pipeline failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
        }
