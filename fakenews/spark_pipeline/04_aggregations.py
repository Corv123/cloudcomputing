"""
Stage 4: Aggregations & Statistics
Perform aggregations on cleaned and feature-engineered data
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import count, avg, min, max, desc, stddev


def perform_aggregations(df: DataFrame) -> dict:
    """
    Perform aggregations on the dataset.
    
    Args:
        df: DataFrame with cleaned data
    
    Returns:
        Dictionary with aggregated DataFrames
    """
    print("=" * 80)
    print("STAGE 4: AGGREGATIONS")
    print("=" * 80)
    
    aggregations = {}
    
    # Label distribution
    print("  Computing label distribution...")
    label_dist = df.groupBy("label").agg(
        count("*").alias("count")
    ).orderBy("label")
    aggregations["label_distribution"] = label_dist
    
    # Use collect() instead of toPandas() for compatibility
    label_rows = label_dist.collect()
    print("    Label Distribution:")
    for row in label_rows:
        label_name = "Real" if row["label"] == 0 else "Fake"
        print(f"      {label_name} (label={row['label']}): {row['count']:,}")
    
    # Source distribution
    print("  Computing source distribution...")
    source_dist = df.groupBy("source").agg(
        count("*").alias("count")
    ).orderBy(desc("count"))
    aggregations["source_distribution"] = source_dist
    
    # Use collect() instead of toPandas() for compatibility
    source_rows = source_dist.collect()
    print("    Source Distribution:")
    for row in source_rows:
        print(f"      {row['source']}: {row['count']:,}")
    
    # Text statistics
    print("  Computing text statistics...")
    text_stats = df.agg(
        avg("text_length").alias("avg_text_length"),
        min("text_length").alias("min_text_length"),
        max("text_length").alias("max_text_length"),
        stddev("text_length").alias("std_text_length"),
        avg("word_count").alias("avg_word_count"),
        min("word_count").alias("min_word_count"),
        max("word_count").alias("max_word_count"),
        stddev("word_count").alias("std_word_count")
    )
    aggregations["text_statistics"] = text_stats
    
    # Use collect() instead of toPandas() for compatibility
    stats_row = text_stats.collect()[0]
    print("    Text Statistics:")
    print(f"      Average text length: {stats_row['avg_text_length']:.0f} characters")
    print(f"      Min text length: {stats_row['min_text_length']} characters")
    print(f"      Max text length: {stats_row['max_text_length']:,} characters")
    print(f"      Std text length: {stats_row['std_text_length']:.0f} characters")
    print(f"      Average word count: {stats_row['avg_word_count']:.0f} words")
    print(f"      Min word count: {stats_row['min_word_count']} words")
    print(f"      Max word count: {stats_row['max_word_count']:,} words")
    print(f"      Std word count: {stats_row['std_word_count']:.0f} words")
    
    # Label by source
    print("  Computing label distribution by source...")
    label_by_source = df.groupBy("source", "label").agg(
        count("*").alias("count")
    ).orderBy("source", "label")
    aggregations["label_by_source"] = label_by_source
    
    print("=" * 80)
    print()
    
    return aggregations


def get_aggregation_summary(aggregations: dict) -> dict:
    """
    Convert aggregations to summary dictionary for easy access.
    
    Args:
        aggregations: Dictionary with aggregated DataFrames
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # Label distribution
    if "label_distribution" in aggregations:
        label_rows = aggregations["label_distribution"].collect()
        summary["label_distribution"] = {
            int(row["label"]): int(row["count"]) for row in label_rows
        }
    
    # Source distribution
    if "source_distribution" in aggregations:
        source_rows = aggregations["source_distribution"].collect()
        summary["source_distribution"] = {
            row["source"]: int(row["count"]) for row in source_rows
        }
    
    # Text statistics
    if "text_statistics" in aggregations:
        stats_row = aggregations["text_statistics"].collect()[0]
        summary["text_statistics"] = {
            "avg_text_length": float(stats_row["avg_text_length"]),
            "min_text_length": int(stats_row["min_text_length"]),
            "max_text_length": int(stats_row["max_text_length"]),
            "std_text_length": float(stats_row["std_text_length"]),
            "avg_word_count": float(stats_row["avg_word_count"]),
            "min_word_count": int(stats_row["min_word_count"]),
            "max_word_count": int(stats_row["max_word_count"]),
            "std_word_count": float(stats_row["std_word_count"])
        }
    
    return summary

