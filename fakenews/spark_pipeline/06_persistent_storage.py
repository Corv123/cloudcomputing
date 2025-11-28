"""
Stage 6: Persistent Storage
Write cleaned data, features, and aggregations to S3 or local storage
"""

from pyspark.sql import DataFrame
from typing import Optional
import os


def write_cleaned_data(
    df: DataFrame,
    output_path: str = "output/cleaned-data/",
    use_s3: bool = False,
    format: str = "parquet"
):
    """
    Write cleaned data to persistent storage.
    
    Args:
        df: Cleaned DataFrame
        output_path: Output path (local or S3)
        use_s3: Whether to write to S3
        format: Output format ("parquet" or "csv")
    """
    print("=" * 80)
    print("STAGE 6: PERSISTENT STORAGE")
    print("=" * 80)
    
    if format == "parquet":
        print(f"  Writing cleaned data to Parquet: {output_path}")
        df.write.mode("overwrite").parquet(output_path)
        print(f"    ✓ Cleaned data written to {output_path}")
    elif format == "csv":
        print(f"  Writing cleaned data to CSV: {output_path}")
        df.select("text_full", "label", "source", "text_length", "word_count").write.mode("overwrite").csv(
            output_path,
            header=True
        )
        print(f"    ✓ Cleaned data written to {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print("=" * 80)
    print()


def write_aggregations(
    aggregations: dict,
    output_path: str = "output/statistics/",
    use_s3: bool = False
):
    """
    Write aggregated statistics to persistent storage.
    
    Args:
        aggregations: Dictionary with aggregated DataFrames
        output_path: Output path (local or S3)
        use_s3: Whether to write to S3
    """
    print("  Writing aggregations to storage...")
    
    for name, df in aggregations.items():
        if isinstance(df, DataFrame):
            agg_path = os.path.join(output_path, f"{name}.parquet")
            df.write.mode("overwrite").parquet(agg_path)
            print(f"    ✓ {name} written to {agg_path}")
    
    print()


def write_features(
    df: DataFrame,
    output_path: str = "output/features/",
    use_s3: bool = False
):
    """
    Write feature-engineered data to persistent storage.
    
    Args:
        df: DataFrame with features
        output_path: Output path (local or S3)
        use_s3: Whether to write to S3
    """
    print("  Writing feature-engineered data to storage...")
    
    # Write full DataFrame with features
    df.write.mode("overwrite").parquet(output_path)
    print(f"    ✓ Features written to {output_path}")
    print()


def write_visualizations(
    plot_paths: list,
    output_path: str = "output/visualizations/",
    use_s3: bool = False
):
    """
    Upload visualizations to S3 or copy to output directory.
    
    Args:
        plot_paths: List of local plot file paths
        output_path: Output path (local or S3)
        use_s3: Whether to upload to S3
    """
    if not plot_paths:
        return
    
    print("  Writing visualizations to storage...")
    
    if use_s3:
        import boto3
        s3_client = boto3.client('s3')
        
        # Extract bucket and prefix from S3 path
        if output_path.startswith("s3://"):
            parts = output_path[5:].split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            for plot_path in plot_paths:
                filename = os.path.basename(plot_path)
                s3_key = f"{prefix}{filename}" if prefix else filename
                
                s3_client.upload_file(plot_path, bucket, s3_key)
                print(f"    ✓ Uploaded {filename} to s3://{bucket}/{s3_key}")
        else:
            print("    ⚠️ Warning: use_s3=True but output_path is not S3 path")
    else:
        # Copy to local output directory
        os.makedirs(output_path, exist_ok=True)
        
        for plot_path in plot_paths:
            filename = os.path.basename(plot_path)
            dest_path = os.path.join(output_path, filename)
            
            # Check if source and destination are the same file
            import shutil
            import os.path as ospath
            if ospath.abspath(plot_path) == ospath.abspath(dest_path):
                print(f"    ✓ {filename} already in correct location")
            else:
                shutil.copy2(plot_path, dest_path)
                print(f"    ✓ Copied {filename} to {dest_path}")
    
    print()

