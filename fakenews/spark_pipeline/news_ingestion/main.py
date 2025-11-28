"""
Main Orchestrator for News Ingestion Pipeline
Runs hourly to scrape and store news articles for cross-checking
"""

import argparse
import os
import sys
import importlib.util
from datetime import datetime
from typing import Optional
from pyspark.sql import SparkSession

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../../analyzers'))

# Import pipeline stages
# Files are in same directory, so use direct imports
from scrape_news import scrape_new_articles, load_existing_articles
from clean_and_deduplicate import clean_and_deduplicate, filter_credible_sources
from store_to_database import store_to_dynamodb, store_to_s3_parquet

# Import embedding stage (file has numeric prefix, use importlib)
try:
    import importlib.util
    embedding_file = os.path.join(current_dir, '04_generate_embeddings.py')
    if os.path.exists(embedding_file):
        spec = importlib.util.spec_from_file_location("generate_embeddings", embedding_file)
        generate_embeddings_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_embeddings_module)
        generate_embeddings_batch = generate_embeddings_module.generate_embeddings_batch
        update_embeddings_in_dynamodb = generate_embeddings_module.update_embeddings_in_dynamodb
        EMBEDDING_STAGE_AVAILABLE = True
    else:
        EMBEDDING_STAGE_AVAILABLE = False
        generate_embeddings_batch = None
        update_embeddings_in_dynamodb = None
except (ImportError, FileNotFoundError, AttributeError) as e:
    EMBEDDING_STAGE_AVAILABLE = False
    generate_embeddings_batch = None
    update_embeddings_in_dynamodb = None
    print(f"[WARN] Embedding generation stage not available: {e}")


def create_spark_session(app_name: str = "NewsIngestionPipeline") -> SparkSession:
    """Create Spark session with appropriate configuration"""
    builder = SparkSession.builder.appName(app_name)
    
    # Windows-specific configuration
    if sys.platform == 'win32':
        import os
        hadoop_home = os.path.join(os.path.dirname(__file__), '../../..')
        os.environ['HADOOP_HOME'] = hadoop_home
        builder.config("spark.sql.warehouse.dir", "file:///C:/temp/spark-warehouse")
    
    # AWS configuration
    builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    builder.config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                   "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def run_pipeline(
    spark: SparkSession,
    tracking_file: str = "scraper_tracking.json",
    csv_file: str = "news_articles.csv",
    use_s3: bool = False,
    s3_tracking_path: Optional[str] = None,
    s3_csv_path: Optional[str] = None,
    s3_parquet_path: Optional[str] = None,
    dynamodb_table: str = "fakenews-scraped-news",
    region: str = "ap-southeast-2",
    store_to_db: bool = True,
    store_to_s3: bool = True
):
    """
    Run the complete news ingestion pipeline
    
    Args:
        spark: SparkSession instance
        tracking_file: Local tracking file path
        csv_file: Local CSV file path
        use_s3: Whether to use S3 for storage
        s3_tracking_path: S3 path for tracking file
        s3_csv_path: S3 path for CSV file
        s3_parquet_path: S3 path for Parquet backup
        dynamodb_table: DynamoDB table name
        region: AWS region
        store_to_db: Whether to store to DynamoDB
        store_to_s3: Whether to store to S3 Parquet
    """
    print("=" * 80)
    print("NEWS INGESTION PIPELINE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Stage 1: Scrape new articles
    scraped_articles = scrape_new_articles(
        spark,
        tracking_file=tracking_file,
        csv_file=csv_file,
        use_s3=use_s3,
        s3_tracking_path=s3_tracking_path,
        s3_csv_path=s3_csv_path
    )
    
    # Load existing articles
    existing_df = load_existing_articles(
        spark,
        csv_path=csv_file if not use_s3 else s3_csv_path,
        use_s3=use_s3
    )
    
    # Stage 2: Clean and deduplicate
    cleaned_df = clean_and_deduplicate(
        spark,
        articles_df=existing_df,
        scraped_articles=scraped_articles
    )
    
    if cleaned_df is None:
        print("[ERROR] Pipeline failed at cleaning stage")
        return
    
    # Stage 3: Store to database
    if store_to_db:
        stored_count = store_to_dynamodb(
            spark,
            articles_df=cleaned_df,
            table_name=dynamodb_table,
            region=region
        )
        print(f"[OK] Stored {stored_count} articles to DynamoDB")
    
    # Stage 4: Generate embeddings for new articles (if available)
    if EMBEDDING_STAGE_AVAILABLE:
        print()
        articles_with_embeddings = generate_embeddings_batch(
            spark,
            articles_df=cleaned_df,
            model_name="all-MiniLM-L6-v2"
        )
        
        # Update embeddings in DynamoDB
        if store_to_db and articles_with_embeddings.count() > 0:
            updated_count = update_embeddings_in_dynamodb(
                spark,
                articles_df=articles_with_embeddings,
                table_name=dynamodb_table,
                region=region
            )
            print(f"[OK] Updated embeddings for {updated_count} articles")
    else:
        print("[WARN] Skipping embedding generation (sentence-transformers not available)")
    
    # Store to S3 Parquet (backup)
    if store_to_s3 and s3_parquet_path:
        parquet_count = store_to_s3_parquet(
            spark,
            articles_df=cleaned_df,
            s3_path=s3_parquet_path
        )
        print(f"[OK] Stored {parquet_count} articles to S3 Parquet")
    
    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Ingestion Pipeline")
    parser.add_argument("--tracking_file", default="scraper_tracking.json",
                       help="Path to tracking JSON file")
    parser.add_argument("--csv_file", default="news_articles.csv",
                       help="Path to CSV file")
    parser.add_argument("--use_s3", action="store_true",
                       help="Use S3 for storage")
    parser.add_argument("--s3_tracking", 
                       help="S3 path for tracking file (e.g., s3://bucket/path/tracking.json)")
    parser.add_argument("--s3_csv",
                       help="S3 path for CSV file (e.g., s3://bucket/path/articles.csv)")
    parser.add_argument("--s3_parquet",
                       help="S3 path for Parquet backup (e.g., s3://bucket/articles/)")
    parser.add_argument("--dynamodb_table", default="fakenews-scraped-news",
                       help="DynamoDB table name")
    parser.add_argument("--region", default="ap-southeast-2",
                       help="AWS region")
    parser.add_argument("--no_db", action="store_true",
                       help="Skip DynamoDB storage")
    parser.add_argument("--no_s3", action="store_true",
                       help="Skip S3 Parquet storage")
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = create_spark_session("NewsIngestionPipeline")
    
    try:
        # Run pipeline
        run_pipeline(
            spark,
            tracking_file=args.tracking_file,
            csv_file=args.csv_file,
            use_s3=args.use_s3,
            s3_tracking_path=args.s3_tracking,
            s3_csv_path=args.s3_csv,
            s3_parquet_path=args.s3_parquet,
            dynamodb_table=args.dynamodb_table,
            region=args.region,
            store_to_db=not args.no_db,
            store_to_s3=not args.no_s3
        )
    finally:
        spark.stop()

