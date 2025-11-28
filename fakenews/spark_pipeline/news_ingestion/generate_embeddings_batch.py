"""
Standalone script to generate embeddings for all existing articles in DynamoDB.
This can be run once to backfill embeddings for articles that were added before embedding support.
"""

import argparse
import boto3
from decimal import Decimal
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../analyzers'))

from embedding_utils import generate_embedding

def generate_embeddings_for_existing_articles(
    table_name: str = "fakenews-scraped-news",
    region: str = "ap-southeast-2",
    batch_size: int = 25
):
    """
    Generate embeddings for all existing articles in DynamoDB that don't have embeddings yet.
    
    Args:
        table_name: DynamoDB table name
        region: AWS region
        batch_size: Batch size for updates
    """
    print("=" * 80)
    print("GENERATING EMBEDDINGS FOR EXISTING ARTICLES")
    print("=" * 80)
    
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    # Scan all articles
    print("Scanning DynamoDB table...")
    response = table.scan()
    articles = response.get('Items', [])
    
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        articles.extend(response.get('Items', []))
    
    print(f"[OK] Found {len(articles)} articles")
    
    # Filter articles without embeddings
    articles_needing_embeddings = []
    for article in articles:
        if article.get('article_type') != 'scraped_news':
            continue
        if 'embedding' in article and article.get('embedding'):
            continue  # Already has embedding
        content = article.get('content', '')
        if not content or len(str(content)) < 100:
            continue
        articles_needing_embeddings.append(article)
    
    print(f"[OK] {len(articles_needing_embeddings)} articles need embeddings")
    
    if len(articles_needing_embeddings) == 0:
        print("[OK] All articles already have embeddings")
        return
    
    # Generate embeddings
    print("Generating embeddings...")
    updated_count = 0
    failed_count = 0
    
    for i, article in enumerate(articles_needing_embeddings):
        try:
            article_id = article['id']
            content = str(article.get('content', ''))
            
            # Generate embedding
            embedding = generate_embedding(content)
            if not embedding:
                failed_count += 1
                continue
            
            # Convert to DynamoDB format
            embedding_decimal = [Decimal(str(float(x))) for x in embedding]
            
            # Update item
            table.update_item(
                Key={'id': article_id},
                UpdateExpression='SET embedding = :emb',
                ExpressionAttributeValues={':emb': embedding_decimal}
            )
            
            updated_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(articles_needing_embeddings)}")
        
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Show first 5 errors for debugging
                import traceback
                print(f"  [ERROR] Failed to generate embedding for {article.get('id', 'unknown')}: {e}")
                traceback.print_exc()
            elif failed_count == 6:
                print(f"  [WARN] Suppressing further error messages...")
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Updated: {updated_count} articles")
    print(f"Failed: {failed_count} articles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for existing articles")
    parser.add_argument("--table_name", default="fakenews-scraped-news",
                       help="DynamoDB table name")
    parser.add_argument("--region", default="ap-southeast-2",
                       help="AWS region")
    
    args = parser.parse_args()
    
    generate_embeddings_for_existing_articles(
        table_name=args.table_name,
        region=args.region
    )

