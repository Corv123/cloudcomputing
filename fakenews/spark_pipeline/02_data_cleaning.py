"""
Stage 2: Data Cleaning & Enrichment
Apply transformations to clean and enrich the data
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, concat, lit, length, split, size, lower, regexp_replace
)
from pyspark.sql.types import StringType, IntegerType
from typing import Optional


def clean_and_enrich_dataframe(
    df: DataFrame,
    source_name: str,
    label_value: Optional[int] = None
) -> DataFrame:
    """
    Apply cleaning and enrichment transformations to a DataFrame.
    
    Args:
        df: Input DataFrame
        source_name: Name of the data source (e.g., "buzzfeed_fake")
        label_value: Explicit label value (0 or 1) if not in DataFrame
    
    Returns:
        Cleaned and enriched DataFrame
    """
    print(f"  Cleaning dataset: {source_name}")
    
    # Step 1: Handle missing values
    # Fill missing title and text columns
    if "title" in df.columns:
        df_clean = df.fillna({"title": "", "text": ""})
    else:
        # Some datasets only have "text" column
        df_clean = df.fillna({"text": ""})
        # Add empty title column
        df_clean = df_clean.withColumn("title", lit(""))
    
    # Step 2: Combine title + text into text_full
    df_clean = df_clean.withColumn(
        "text_full",
        concat(
            col("title").cast(StringType()),
            lit("\n\n"),
            col("text").cast(StringType())
        )
    )
    
    # Step 3: Normalize label column
    if label_value is not None:
        # Explicit label (True.csv, Fake.csv, BuzzFeed, PolitiFact)
        df_clean = df_clean.withColumn("label", lit(label_value).cast(IntegerType()))
    elif "label" in df_clean.columns:
        # WELFake_Dataset.csv: map "true"/"false" to 0/1
        df_clean = df_clean.withColumn(
            "label_normalized",
            when(
                lower(col("label").cast(StringType())).isin(["true", "1", "1.0"]), 0
            ).when(
                lower(col("label").cast(StringType())).isin(["false", "0", "0.0"]), 1
            ).otherwise(None)
        )
        df_clean = df_clean.withColumn(
            "label",
            col("label_normalized").cast(IntegerType())
        )
    else:
        # No label column - this shouldn't happen, but handle gracefully
        print(f"    ⚠️ Warning: No label column found for {source_name}")
        return None
    
    # Step 4: Add source column
    df_clean = df_clean.withColumn("source", lit(source_name))
    
    # Step 5: Add metadata columns
    df_clean = df_clean.withColumn(
        "text_length",
        length(col("text_full"))
    )
    df_clean = df_clean.withColumn(
        "word_count",
        size(split(col("text_full"), " "))
    )
    
    # Step 6: Remove invalid rows
    df_clean = df_clean.filter(
        col("text_full").isNotNull() &
        (col("text_full") != "") &
        (col("text_full") != "\n\n") &  # Empty after concatenation
        col("label").isNotNull()
    )
    
    # Step 7: Select required columns
    df_clean = df_clean.select(
        "text_full",
        "label",
        "source",
        "text_length",
        "word_count"
    )
    
    row_count = df_clean.count()
    print(f"    ✓ Cleaned dataset: {row_count} rows")
    
    return df_clean


def union_all_datasets(cleaned_datasets: list) -> DataFrame:
    """
    Union all cleaned datasets into a single DataFrame.
    
    Args:
        cleaned_datasets: List of cleaned DataFrames
    
    Returns:
        Combined DataFrame
    """
    if not cleaned_datasets:
        raise ValueError("No cleaned datasets to union")
    
    print("  Combining all datasets...")
    
    # Union all DataFrames
    combined_df = cleaned_datasets[0]
    for df in cleaned_datasets[1:]:
        combined_df = combined_df.union(df)
    
    total_rows = combined_df.count()
    print(f"    ✓ Combined dataset: {total_rows} total rows")
    
    return combined_df


def get_data_summary(df: DataFrame) -> dict:
    """
    Get summary statistics of the cleaned dataset.
    
    Args:
        df: Combined DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    from pyspark.sql.functions import count, avg, min, max
    
    summary = {}
    
    # Total count
    summary["total_rows"] = df.count()
    
    # Label distribution
    label_dist = df.groupBy("label").agg(count("*").alias("count")).collect()
    summary["label_distribution"] = {row["label"]: row["count"] for row in label_dist}
    
    # Source distribution
    source_dist = df.groupBy("source").agg(count("*").alias("count")).collect()
    summary["source_distribution"] = {row["source"]: row["count"] for row in source_dist}
    
    # Text statistics
    stats = df.agg(
        avg("text_length").alias("avg_text_length"),
        min("text_length").alias("min_text_length"),
        max("text_length").alias("max_text_length"),
        avg("word_count").alias("avg_word_count"),
        min("word_count").alias("min_word_count"),
        max("word_count").alias("max_word_count")
    ).collect()[0]
    
    summary["text_statistics"] = {
        "avg_text_length": stats["avg_text_length"],
        "min_text_length": stats["min_text_length"],
        "max_text_length": stats["max_text_length"],
        "avg_word_count": stats["avg_word_count"],
        "min_word_count": stats["min_word_count"],
        "max_word_count": stats["max_word_count"]
    }
    
    return summary


def print_data_summary(summary: dict):
    """
    Print formatted summary statistics.
    
    Args:
        summary: Summary dictionary from get_data_summary()
    """
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total rows: {summary['total_rows']:,}")
    print()
    print("Label Distribution:")
    for label, count in summary["label_distribution"].items():
        label_name = "Real" if label == 0 else "Fake"
        print(f"  {label_name} (label={label}): {count:,} ({count/summary['total_rows']*100:.1f}%)")
    print()
    print("Source Distribution:")
    for source, count in sorted(summary["source_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,} ({count/summary['total_rows']*100:.1f}%)")
    print()
    print("Text Statistics:")
    stats = summary["text_statistics"]
    print(f"  Average text length: {stats['avg_text_length']:.0f} characters")
    print(f"  Min text length: {stats['min_text_length']} characters")
    print(f"  Max text length: {stats['max_text_length']:,} characters")
    print(f"  Average word count: {stats['avg_word_count']:.0f} words")
    print(f"  Min word count: {stats['min_word_count']} words")
    print(f"  Max word count: {stats['max_word_count']:,} words")
    print("=" * 80)
    print()

