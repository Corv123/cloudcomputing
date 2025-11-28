"""
Upload news_articles.csv to DynamoDB table fakenews-scraped-news
"""

import boto3
import pandas as pd
import hashlib
from urllib.parse import urlparse
from decimal import Decimal
from datetime import datetime
import sys
import os

# Add path for CSV file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../analyzers'))

def upload_csv_to_dynamodb(
    csv_file: str = "fakenews/analyzers/news_articles.csv",
    table_name: str = "fakenews-scraped-news",
    region: str = "ap-southeast-2"
):
    """
    Upload articles from CSV to DynamoDB table
    
    Args:
        csv_file: Path to CSV file
        table_name: DynamoDB table name
        region: AWS region
    """
    print("=" * 80)
    print("UPLOADING CSV TO DYNAMODB")
    print("=" * 80)
    print(f"CSV file: {csv_file}")
    print(f"Table: {table_name}")
    print()
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"[ERROR] CSV file not found: {csv_file}")
        return False
    
    # Read CSV
    print(f"Reading CSV file...")
    try:
        df = pd.read_csv(csv_file)
        print(f"[OK] Loaded {len(df)} articles from CSV")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return False
    
    # Connect to DynamoDB
    print(f"Connecting to DynamoDB...")
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    # Verify table exists
    try:
        table.load()
        print(f"[OK] Table exists and is ready")
    except Exception as e:
        print(f"[ERROR] Table does not exist or is not accessible: {e}")
        print(f"Please create the table first using create_table.ps1")
        return False
    
    print()
    print(f"Uploading articles to DynamoDB...")
    
    uploaded = 0
    skipped = 0
    failed = 0
    
    # Process in batches
    batch_size = 25
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        
        try:
            with table.batch_writer() as batch_writer:
                for _, row in batch.iterrows():
                    try:
                        # Get article data
                        title = str(row.get('title', '')).strip()
                        content = str(row.get('content', '')).strip()
                        source_url = str(row.get('source_url', '')).strip()
                        published_at = str(row.get('published_at', ''))
                        
                        # Skip if missing required fields
                        if not title or not content or not source_url:
                            skipped += 1
                            continue
                        
                        # Clean URL
                        parsed = urlparse(source_url)
                        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
                        
                        # Generate ID
                        article_id = hashlib.md5(clean_url.encode()).hexdigest()
                        
                        # Build item
                        item = {
                            'id': article_id,
                            'title': title,
                            'content': content,
                            'source_url': clean_url,
                            'domain': parsed.netloc.replace('www.', '').lower(),
                            'published_at': published_at if published_at else datetime.now().strftime('%Y-%m-%d'),
                            'ingested_at': datetime.now().isoformat(),
                            'article_type': 'scraped_news',
                            'word_count': Decimal(str(len(content.split())))
                        }
                        
                        batch_writer.put_item(Item=item)
                        uploaded += 1
                        
                    except Exception as e:
                        failed += 1
                        print(f"  [WARN] Failed to upload article: {e}")
            
            if (i + batch_size) % 100 == 0:
                print(f"  Progress: {min(i + batch_size, len(df))}/{len(df)}")
                
        except Exception as e:
            print(f"  [ERROR] Batch write failed: {e}")
            failed += len(batch)
    
    print()
    print("=" * 80)
    print("UPLOAD COMPLETE")
    print("=" * 80)
    print(f"Uploaded: {uploaded} articles")
    print(f"Skipped: {skipped} articles (missing data)")
    print(f"Failed: {failed} articles")
    print()
    
    return uploaded > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload CSV to DynamoDB")
    parser.add_argument("--csv_file", default="fakenews/analyzers/news_articles.csv",
                       help="Path to CSV file")
    parser.add_argument("--table_name", default="fakenews-scraped-news",
                       help="DynamoDB table name")
    parser.add_argument("--region", default="ap-southeast-2",
                       help="AWS region")
    
    args = parser.parse_args()
    
    success = upload_csv_to_dynamodb(
        csv_file=args.csv_file,
        table_name=args.table_name,
        region=args.region
    )
    
    sys.exit(0 if success else 1)

