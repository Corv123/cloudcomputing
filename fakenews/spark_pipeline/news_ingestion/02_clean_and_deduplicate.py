"""
Stage 2: Clean and Deduplicate Articles
Removes duplicates, cleans data, and prepares for storage
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, lower, regexp_replace, udf, length
from pyspark.sql.types import StringType, TimestampType
from typing import List, Dict
from urllib.parse import urlparse
import hashlib


def clean_article_url(url: str) -> str:
    """Clean URL by removing query parameters and fragments for deduplication"""
    try:
        parsed = urlparse(url)
        # Keep only scheme, netloc, and path
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return clean_url.rstrip('/')
    except:
        return url


def generate_article_id(url: str) -> str:
    """Generate unique ID from URL"""
    clean_url = clean_article_url(url)
    return hashlib.md5(clean_url.encode()).hexdigest()


def clean_and_deduplicate(
    spark: SparkSession,
    articles_df,
    scraped_articles: List[Dict]
) -> 'DataFrame':
    """
    Clean and deduplicate articles
    
    Args:
        spark: SparkSession instance
        articles_df: Existing articles DataFrame (from CSV)
        scraped_articles: Newly scraped articles (list of dicts)
    
    Returns:
        Cleaned and deduplicated DataFrame
    """
    print("=" * 80)
    print("STAGE 2: CLEAN AND DEDUPLICATE")
    print("=" * 80)
    
    from pyspark.sql.types import StructType, StructField, StringType
    
    # Convert scraped articles to DataFrame
    if scraped_articles:
        print(f"  Converting {len(scraped_articles)} scraped articles to DataFrame...")
        schema = StructType([
            StructField("title", StringType(), True),
            StructField("content", StringType(), True),
            StructField("source_url", StringType(), True),
            StructField("published_at", StringType(), True)
        ])
        new_df = spark.createDataFrame(scraped_articles, schema)
        print(f"    [OK] Created DataFrame from scraped articles")
    else:
        print(f"  [WARN] No new articles scraped")
        new_df = None
    
    # Combine existing and new articles
    if articles_df is not None and new_df is not None:
        print(f"  Combining existing and new articles...")
        combined_df = articles_df.union(new_df)
        print(f"    [OK] Combined: {combined_df.count()} total articles")
    elif new_df is not None:
        combined_df = new_df
        print(f"    [OK] Using only new articles: {combined_df.count()} articles")
    elif articles_df is not None:
        combined_df = articles_df
        print(f"    [OK] Using only existing articles: {combined_df.count()} articles")
    else:
        print(f"    [ERROR] No articles available")
        return None
    
    # Register UDFs
    clean_url_udf = udf(clean_article_url, StringType())
    generate_id_udf = udf(generate_article_id, StringType())
    
    # Clean data
    print(f"  Cleaning article data...")
    cleaned_df = combined_df.select(
        generate_id_udf(col("source_url")).alias("id"),
        trim(col("title")).alias("title"),
        trim(col("content")).alias("content"),
        clean_url_udf(col("source_url")).alias("source_url"),
        col("published_at").alias("published_at"),
        # Extract domain for filtering
        regexp_replace(
            regexp_replace(col("source_url"), "https?://", ""),
            "/.*", ""
        ).alias("domain")
    ).filter(
        col("title").isNotNull() &
        col("content").isNotNull() &
        (col("content") != "") &
        (length(col("content")) > 50)  # Minimum content length
    )
    
    print(f"    [OK] Cleaned: {cleaned_df.count()} articles after filtering")
    
    # Deduplicate by URL (keep first occurrence)
    print(f"  Deduplicating by URL...")
    before_count = cleaned_df.count()
    deduplicated_df = cleaned_df.dropDuplicates(["source_url"])
    after_count = deduplicated_df.count()
    duplicates_removed = before_count - after_count
    
    print(f"    [OK] Removed {duplicates_removed} duplicates")
    print(f"    [OK] Final count: {after_count} unique articles")
    
    # Add timestamp
    from pyspark.sql.functions import current_timestamp, lit
    final_df = deduplicated_df.withColumn(
        "ingested_at",
        current_timestamp()
    )
    
    print("=" * 80)
    print()
    
    return final_df


def filter_credible_sources(df, credible_domains: List[str]):
    """
    Filter articles to only include credible sources
    
    Args:
        df: Articles DataFrame
        credible_domains: List of credible domain names
    
    Returns:
        Filtered DataFrame
    """
    from pyspark.sql.functions import lower, col
    
    credible_domains_lower = [d.lower().replace('www.', '') for d in credible_domains]
    
    # Create filter condition
    conditions = None
    for domain in credible_domains_lower:
        condition = lower(col("domain")).contains(domain)
        if conditions is None:
            conditions = condition
        else:
            conditions = conditions | condition
    
    if conditions is not None:
        filtered_df = df.filter(conditions)
        return filtered_df
    else:
        return df

