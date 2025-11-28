"""
Stage 3: Store Articles to Database
Stores cleaned articles to DynamoDB for fast cross-checking
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from typing import Optional
import boto3
from decimal import Decimal
from datetime import datetime
import json


def store_to_dynamodb(
    spark: SparkSession,
    articles_df,
    table_name: str = "fakenews-scraped-news",
    region: str = "ap-southeast-2",
    batch_size: int = 25  # DynamoDB batch write limit
):
    """
    Store articles to DynamoDB for fast cross-checking
    
    Args:
        spark: SparkSession instance
        articles_df: Cleaned articles DataFrame
        table_name: DynamoDB table name
        region: AWS region
        batch_size: Batch size for DynamoDB writes (max 25)
    
    Returns:
        Number of articles stored
    """
    print("=" * 80)
    print("STAGE 3: STORE TO DATABASE")
    print("=" * 80)
    
    if articles_df is None:
        print("[ERROR] No articles to store")
        return 0
    
    # Convert to list for batch processing
    print(f"  Collecting articles for DynamoDB storage...")
    articles_list = articles_df.collect()
    total_articles = len(articles_list)
    
    print(f"    [OK] Collected {total_articles} articles")
    
    # Initialize DynamoDB client
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    # Convert articles to DynamoDB format
    print(f"  Converting to DynamoDB format...")
    dynamodb_items = []
    
    for article in articles_list:
        # Convert to DynamoDB item format
        item = {
            'id': article['id'],
            'title': article['title'],
            'content': article['content'],
            'source_url': article['source_url'],
            'domain': article.get('domain', ''),
            'published_at': article.get('published_at', ''),
            'ingested_at': article.get('ingested_at', datetime.now().isoformat()),
            'article_type': 'scraped_news',  # Mark as scraped news
            'word_count': len(article['content'].split()) if article.get('content') else 0
        }
        
        # Add embedding if available (from embedding generation stage)
        if 'embedding' in article and article.get('embedding') is not None:
            # Convert embedding list to DynamoDB format (list of Decimal)
            embedding_list = article['embedding']
            if embedding_list:
                item['embedding'] = [Decimal(str(float(x))) for x in embedding_list]
        
        # Convert to DynamoDB-compatible format (Decimal for numbers)
        dynamodb_item = {}
        for key, value in item.items():
            if isinstance(value, (int, float)):
                dynamodb_item[key] = Decimal(str(value))
            elif value is None:
                continue  # Skip None values
            else:
                dynamodb_item[key] = str(value)
        
        dynamodb_items.append(dynamodb_item)
    
    print(f"    [OK] Converted {len(dynamodb_items)} items")
    
    # Batch write to DynamoDB
    print(f"  Writing to DynamoDB table: {table_name}...")
    stored_count = 0
    failed_count = 0
    
    # Process in batches
    for i in range(0, len(dynamodb_items), batch_size):
        batch = dynamodb_items[i:i + batch_size]
        
        try:
            # Prepare batch write request
            with table.batch_writer() as batch_writer:
                for item in batch:
                    try:
                        batch_writer.put_item(Item=item)
                        stored_count += 1
                    except Exception as e:
                        # Check if it's a duplicate (conditional check failed)
                        if 'ConditionalCheckFailedException' in str(e):
                            # Item already exists, skip
                            pass
                        else:
                            failed_count += 1
                            print(f"    [WARN] Failed to store item {item.get('id', 'unknown')}: {e}")
            
            if (i + batch_size) % 100 == 0:
                print(f"    Progress: {min(i + batch_size, len(dynamodb_items))}/{len(dynamodb_items)}")
                
        except Exception as e:
            print(f"    [ERROR] Batch write failed: {e}")
            failed_count += len(batch)
    
    print(f"    [OK] Stored {stored_count} articles")
    if failed_count > 0:
        print(f"    [WARN] Failed to store {failed_count} articles")
    
    print("=" * 80)
    print()
    
    return stored_count


def store_to_s3_parquet(
    spark: SparkSession,
    articles_df,
    s3_path: str = "s3://fakenews-news-database/articles/",
    partition_by: Optional[str] = "domain"
):
    """
    Store articles to S3 as Parquet for backup and analytics
    
    Args:
        spark: SparkSession instance
        articles_df: Cleaned articles DataFrame
        s3_path: S3 path for Parquet files
        partition_by: Column to partition by (optional)
    
    Returns:
        Number of articles stored
    """
    print("=" * 80)
    print("STAGE 3: STORE TO S3 (PARQUET)")
    print("=" * 80)
    
    if articles_df is None:
        print("[ERROR] No articles to store")
        return 0
    
    count = articles_df.count()
    print(f"  Writing {count} articles to S3...")
    
    try:
        writer = articles_df.write.mode("append")
        
        if partition_by and partition_by in articles_df.columns:
            writer = writer.partitionBy(partition_by)
        
        writer.parquet(s3_path)
        
        print(f"    [OK] Stored {count} articles to {s3_path}")
        print("=" * 80)
        print()
        
        return count
        
    except Exception as e:
        print(f"    [ERROR] Failed to store to S3: {e}")
        print("=" * 80)
        print()
        return 0

