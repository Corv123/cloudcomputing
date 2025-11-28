"""
Stage 1: Data Ingestion
Read messy datasets from multiple sources with different schemas and encodings
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from typing import List, Tuple, Optional
import os


def read_all_datasets(
    spark: SparkSession,
    datasets_dir: str = "datasets/",
    use_s3: bool = False,
    s3_bucket: Optional[str] = None
) -> List[Tuple]:
    """
    Read all CSV datasets with different schemas and encoding.
    
    Args:
        spark: SparkSession instance
        datasets_dir: Local directory path (if use_s3=False)
        use_s3: Whether to read from S3 (default: False, reads from local)
        s3_bucket: S3 bucket path (e.g., "s3://fakenews-ml-pipeline/raw-data/")
    
    Returns:
        List of tuples: (DataFrame, source_name, label_value or None)
    """
    datasets = []
    base_path = s3_bucket if use_s3 else datasets_dir
    
    print("=" * 80)
    print("STAGE 1: DATA INGESTION")
    print("=" * 80)
    print(f"Reading datasets from: {base_path}")
    print()
    
    # BuzzFeed datasets
    buzzfeed_files = [
        ("BuzzFeed_fake_news_content.csv", 1, "buzzfeed_fake"),
        ("BuzzFeed_real_news_content.csv", 0, "buzzfeed_real"),
    ]
    
    for file, label, source in buzzfeed_files:
        file_path = os.path.join(base_path, file) if not use_s3 else f"{base_path}{file}"
        
        if use_s3 or os.path.exists(file_path):
            try:
                print(f"  Reading {file}...")
                # Try UTF-8 first
                try:
                    df = spark.read.option("encoding", "utf-8").csv(
                        file_path,
                        header=True,
                        inferSchema=True
                    )
                except Exception as e:
                    print(f"    UTF-8 failed, trying latin-1: {e}")
                    # Fallback to latin-1
                    df = spark.read.option("encoding", "latin-1").csv(
                        file_path,
                        header=True,
                        inferSchema=True
                    )
                
                # Ensure required columns exist
                if "title" not in df.columns or "text" not in df.columns:
                    print(f"    ⚠️ Warning: {file} missing required columns. Found: {df.columns}")
                    continue
                
                datasets.append((df, source, label))
                print(f"    ✓ Loaded {df.count()} rows")
            except Exception as e:
                print(f"    ✗ Error reading {file}: {e}")
        else:
            print(f"  ⚠️ {file} not found, skipping")
    
    # PolitiFact datasets (if they exist)
    politifact_files = [
        ("PolitiFact_fake_news_content.csv", 1, "politifact_fake"),
        ("PolitiFact_real_news_content.csv", 0, "politifact_real"),
    ]
    
    for file, label, source in politifact_files:
        file_path = os.path.join(base_path, file) if not use_s3 else f"{base_path}{file}"
        
        if use_s3 or os.path.exists(file_path):
            try:
                print(f"  Reading {file}...")
                try:
                    df = spark.read.option("encoding", "utf-8").csv(
                        file_path,
                        header=True,
                        inferSchema=True
                    )
                except Exception as e:
                    print(f"    UTF-8 failed, trying latin-1: {e}")
                    df = spark.read.option("encoding", "latin-1").csv(
                        file_path,
                        header=True,
                        inferSchema=True
                    )
                
                if "title" not in df.columns or "text" not in df.columns:
                    print(f"    ⚠️ Warning: {file} missing required columns")
                    continue
                
                datasets.append((df, source, label))
                print(f"    ✓ Loaded {df.count()} rows")
            except Exception as e:
                print(f"    ✗ Error reading {file}: {e}")
    
    # True.csv (Real content - label as 0)
    true_path = os.path.join(base_path, "True.csv") if not use_s3 else f"{base_path}True.csv"
    if use_s3 or os.path.exists(true_path):
        try:
            print(f"  Reading True.csv...")
            # Try UTF-8 first
            try:
                df = spark.read.option("encoding", "utf-8").csv(
                    true_path,
                    header=True,
                    inferSchema=True
                )
            except Exception as e:
                print(f"    UTF-8 failed, trying latin-1: {e}")
                # Fallback to latin-1
                df = spark.read.option("encoding", "latin-1").csv(
                    true_path,
                    header=True,
                    inferSchema=True
                )
            
            if "text" in df.columns:
                datasets.append((df, "true_corrected", 0))
                print(f"    ✓ Loaded {df.count()} rows")
            else:
                print(f"    ⚠️ Warning: True.csv missing 'text' column")
        except Exception as e:
            print(f"    ✗ Error reading True.csv: {e}")
    
    # Fake.csv (Fake content - label as 1)
    fake_path = os.path.join(base_path, "Fake.csv") if not use_s3 else f"{base_path}Fake.csv"
    if use_s3 or os.path.exists(fake_path):
        try:
            print(f"  Reading Fake.csv...")
            # Try UTF-8 first
            try:
                df = spark.read.option("encoding", "utf-8").csv(
                    fake_path,
                    header=True,
                    inferSchema=True
                )
            except Exception as e:
                print(f"    UTF-8 failed, trying latin-1: {e}")
                # Fallback to latin-1
                df = spark.read.option("encoding", "latin-1").csv(
                    fake_path,
                    header=True,
                    inferSchema=True
                )
            
            if "text" in df.columns:
                datasets.append((df, "fake_corrected", 1))
                print(f"    ✓ Loaded {df.count()} rows")
            else:
                print(f"    ⚠️ Warning: Fake.csv missing 'text' column")
        except Exception as e:
            print(f"    ✗ Error reading Fake.csv: {e}")
    
    # WELFake_Dataset.csv
    wel_path = os.path.join(base_path, "WELFake_Dataset.csv") if not use_s3 else f"{base_path}WELFake_Dataset.csv"
    if use_s3 or os.path.exists(wel_path):
        try:
            print(f"  Reading WELFake_Dataset.csv...")
            try:
                df = spark.read.option("encoding", "utf-8").csv(
                    wel_path,
                    header=True,
                    inferSchema=True
                )
            except Exception as e:
                print(f"    UTF-8 failed, trying latin-1: {e}")
                df = spark.read.option("encoding", "latin-1").csv(
                    wel_path,
                    header=True,
                    inferSchema=True
                )
            
            if "title" in df.columns and "text" in df.columns and "label" in df.columns:
                datasets.append((df, "welfake", None))  # Label will be processed in cleaning
                print(f"    ✓ Loaded {df.count()} rows")
            else:
                print(f"    ⚠️ Warning: WELFake_Dataset.csv missing required columns")
        except Exception as e:
            print(f"    ✗ Error reading WELFake_Dataset.csv: {e}")
    
    print()
    print(f"✓ Total datasets loaded: {len(datasets)}")
    print("=" * 80)
    print()
    
    return datasets


def validate_datasets(datasets: List[Tuple]) -> bool:
    """
    Validate that datasets have required columns.
    
    Args:
        datasets: List of (DataFrame, source_name, label) tuples
    
    Returns:
        True if all datasets are valid, False otherwise
    """
    required_columns = {"title", "text"}  # At least one of these should exist
    
    for df, source, label in datasets:
        columns = set(df.columns)
        
        # Check if either 'title' and 'text' exist, or just 'text' exists
        if not (("title" in columns and "text" in columns) or "text" in columns):
            print(f"✗ Dataset {source} missing required columns. Found: {columns}")
            return False
    
    return True

