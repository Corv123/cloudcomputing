"""
Generate vector embeddings for articles using sentence-transformers.
This stage processes articles in batch to create embeddings for similarity search.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import ArrayType, FloatType
from typing import List, Optional
import sys
import os

# Add path for embedding utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../analyzers'))

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


# Global model instance (loaded once per worker)
_embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or create embedding model instance.
    Uses singleton pattern to avoid reloading model on each call.
    
    Args:
        model_name: Name of the sentence-transformer model
    
    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        print(f"  Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        print(f"  [OK] Model loaded successfully")
    
    return _embedding_model


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
    """
    Generate embedding vector for a text string.
    
    Args:
        text: Text to embed
        model_name: Name of the sentence-transformer model
    
    Returns:
        List of floats representing the embedding vector, or None if error
    """
    try:
        if not text or len(text.strip()) < 10:
            return None
        
        model = get_embedding_model(model_name)
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        
        # Convert numpy array to list of floats
        return embedding.tolist()
    
    except Exception as e:
        print(f"  [WARN] Error generating embedding: {e}")
        return None


def generate_embeddings_batch(
    spark: SparkSession,
    articles_df,
    model_name: str = "all-MiniLM-L6-v2",
    text_column: str = "content",
    embedding_column: str = "embedding"
):
    """
    Generate embeddings for all articles in a DataFrame.
    
    Args:
        spark: SparkSession
        articles_df: DataFrame with articles (must have 'content' column)
        model_name: Name of the sentence-transformer model
        text_column: Column name containing text to embed
        embedding_column: Column name for storing embeddings
    
    Returns:
        DataFrame with added 'embedding' column
    """
    print("=" * 80)
    print("STAGE 4: GENERATING EMBEDDINGS")
    print("=" * 80)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("[ERROR] sentence-transformers not available")
        print("Install with: pip install sentence-transformers")
        return articles_df
    
    if articles_df.count() == 0:
        print("[WARN] No articles to generate embeddings for")
        return articles_df
    
    print(f"Generating embeddings for {articles_df.count()} articles...")
    print(f"Model: {model_name}")
    print()
    
    # Create UDF for embedding generation
    # Note: This will load the model on each worker node
    def generate_embedding_udf(text):
        if not text:
            return None
        try:
            embedding = generate_embedding(str(text), model_name)
            return embedding
        except Exception as e:
            print(f"  [WARN] Embedding generation failed: {e}")
            return None
    
    embedding_udf = udf(
        generate_embedding_udf,
        ArrayType(FloatType())
    )
    
    # Generate embeddings
    print("Processing articles...")
    articles_with_embeddings = articles_df.withColumn(
        embedding_column,
        embedding_udf(col(text_column))
    )
    
    # Filter out articles where embedding generation failed
    articles_with_embeddings = articles_with_embeddings.filter(
        col(embedding_column).isNotNull()
    )
    
    print(f"[OK] Generated embeddings for {articles_with_embeddings.count()} articles")
    
    return articles_with_embeddings


def update_embeddings_in_dynamodb(
    spark: SparkSession,
    articles_df,
    table_name: str = "fakenews-scraped-news",
    region: str = "ap-southeast-2",
    batch_size: int = 25
):
    """
    Update DynamoDB table with embeddings for articles.
    
    Args:
        spark: SparkSession
        articles_df: DataFrame with articles and embeddings
        table_name: DynamoDB table name
        region: AWS region
        batch_size: Batch write size
    
    Returns:
        Number of articles updated
    """
    print("=" * 80)
    print("UPDATING EMBEDDINGS IN DYNAMODB")
    print("=" * 80)
    
    if articles_df.count() == 0:
        print("[WARN] No articles to update")
        return 0
    
    try:
        import boto3
        from decimal import Decimal
    except ImportError:
        print("[ERROR] boto3 not available")
        return 0
    
    # Collect articles to driver
    articles_data = articles_df.select("id", "embedding").collect()
    print(f"[OK] Collected {len(articles_data)} articles for embedding update")
    
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    updated_count = 0
    failed_count = 0
    
    for i in range(0, len(articles_data), batch_size):
        batch = articles_data[i:i + batch_size]
        
        for article in batch:
            try:
                article_id = article.id
                embedding = article.embedding
                
                if not embedding:
                    continue
                
                # Convert embedding list to DynamoDB format (list of Decimal)
                embedding_decimal = [Decimal(str(float(x))) for x in embedding]
                
                # Update item with embedding
                table.update_item(
                    Key={'id': article_id},
                    UpdateExpression='SET embedding = :emb',
                    ExpressionAttributeValues={':emb': embedding_decimal}
                )
                
                updated_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"  [WARN] Failed to update embedding for {article.id}: {e}")
        
        if (i + batch_size) % 100 == 0:
            print(f"  Progress: {min(i + batch_size, len(articles_data))}/{len(articles_data)}")
    
    print(f"[OK] Updated embeddings for {updated_count} articles")
    if failed_count > 0:
        print(f"[WARN] Failed to update {failed_count} articles")
    
    return updated_count


