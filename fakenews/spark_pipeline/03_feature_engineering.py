"""
Stage 3: Feature Engineering
Create Spark ML Pipeline for feature engineering (TF-IDF + Linguistic Features)
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF,
    VectorAssembler, StandardScaler
)
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, array
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField
import sys
import os


def extract_linguistic_features_udf():
    """
    Create UDF for extracting 28 linguistic features.
    This wraps the existing features_enhanced module.
    
    Returns:
        UDF function that takes text and returns array of 28 float features
    """
    def extract_features(text):
        """
        Extract 28 linguistic features using features_enhanced module.
        
        Args:
            text: Article text (string)
        
        Returns:
            List of 28 float values
        """
        try:
            # Try to import features_enhanced module
            # First try from current directory (if copied)
            try:
                from features_enhanced import extract_enhanced_linguistic_features, features_to_array
            except ImportError:
                # Try from same directory
                current_dir = os.path.dirname(__file__)
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                try:
                    from features_enhanced import extract_enhanced_linguistic_features, features_to_array
                except ImportError:
                    # Try from parent src directory
                    features_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
                    if features_path not in sys.path:
                        sys.path.insert(0, features_path)
                    from features_enhanced import extract_enhanced_linguistic_features, features_to_array
            import numpy as np
            
            # Extract features
            feats = extract_enhanced_linguistic_features(text)
            features_array = features_to_array(feats)
            
            # Convert to list
            return features_array.tolist()
        except Exception as e:
            # Return zeros if extraction fails
            print(f"    ⚠️ Warning: Linguistic feature extraction failed: {e}")
            return [0.0] * 28
    
    return udf(extract_features, ArrayType(FloatType()))


def split_array_to_columns(df: DataFrame, array_col: str, num_features: int = 28) -> DataFrame:
    """
    Split an array column into individual columns.
    
    Args:
        df: DataFrame with array column
        array_col: Name of array column
        num_features: Number of features in array
    
    Returns:
        DataFrame with individual feature columns
    """
    for i in range(num_features):
        df = df.withColumn(f"ling_feat_{i}", col(array_col)[i])
    
    return df


def create_tfidf_pipeline(max_features: int = 50000, ngram_range: tuple = (1, 2)):
    """
    Create TF-IDF feature engineering pipeline.
    
    Args:
        max_features: Maximum number of TF-IDF features (default: 50000)
        ngram_range: N-gram range (default: (1, 2) for unigrams and bigrams)
    
    Returns:
        Pipeline with tokenizer, stopwords remover, HashingTF, and IDF
    """
    # Stage 1: Tokenization
    tokenizer = Tokenizer(
        inputCol="text_full",
        outputCol="words"
    )
    
    # Stage 2: Stop words removal
    stopwords_remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )
    
    # Stage 3: TF (Term Frequency) using HashingTF
    # Note: HashingTF doesn't support ngram_range directly
    # We'll use unigrams only, or create ngrams separately if needed
    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=max_features
    )
    
    # Stage 4: IDF (Inverse Document Frequency)
    idf = IDF(
        inputCol="raw_features",
        outputCol="tfidf_features"
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[
        tokenizer,
        stopwords_remover,
        hashing_tf,
        idf
    ])
    
    return pipeline


def create_linguistic_features_pipeline():
    """
    Create pipeline for extracting linguistic features.
    
    Returns:
        UDF for linguistic feature extraction
    """
    linguistic_udf = extract_linguistic_features_udf()
    return linguistic_udf


def create_complete_feature_pipeline(
    max_features: int = 50000,
    include_linguistic: bool = True
) -> Pipeline:
    """
    Create complete feature engineering pipeline (TF-IDF + Linguistic Features).
    
    Args:
        max_features: Maximum number of TF-IDF features
        include_linguistic: Whether to include 28 linguistic features
    
    Returns:
        Complete feature engineering pipeline
    """
    stages = []
    
    # TF-IDF pipeline stages
    tokenizer = Tokenizer(inputCol="text_full", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=max_features
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    
    stages.extend([tokenizer, stopwords_remover, hashing_tf, idf])
    
    # Note: Linguistic features require UDF which can't be directly in Pipeline
    # We'll handle this separately in the main pipeline
    
    return Pipeline(stages=stages)


def apply_linguistic_features(df: DataFrame) -> DataFrame:
    """
    Apply linguistic feature extraction to DataFrame.
    This must be done separately as UDFs can't be in Pipeline stages.
    
    Args:
        df: DataFrame with text_full column
    
    Returns:
        DataFrame with linguistic_features_array column
    """
    print("  Extracting linguistic features (28 features)...")
    linguistic_udf = extract_linguistic_features_udf()
    
    df = df.withColumn("linguistic_features_array", linguistic_udf(col("text_full")))
    
    # Split array into individual columns
    df = split_array_to_columns(df, "linguistic_features_array", num_features=28)
    
    print("    ✓ Linguistic features extracted")
    
    return df


def scale_linguistic_features(df: DataFrame) -> DataFrame:
    """
    Scale linguistic features using StandardScaler.
    
    Args:
        df: DataFrame with ling_feat_0 through ling_feat_27 columns
    
    Returns:
        DataFrame with scaled linguistic features
    """
    print("  Scaling linguistic features...")
    
    # Assemble linguistic features into vector
    linguistic_cols = [f"ling_feat_{i}" for i in range(28)]
    assembler = VectorAssembler(
        inputCols=linguistic_cols,
        outputCol="linguistic_features_vector"
    )
    
    df = assembler.transform(df)
    
    # Scale features
    scaler = StandardScaler(
        inputCol="linguistic_features_vector",
        outputCol="linguistic_features_scaled",
        withMean=True,
        withStd=True
    )
    
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    print("    ✓ Linguistic features scaled")
    
    return df, scaler_model


def combine_all_features(df: DataFrame) -> DataFrame:
    """
    Combine TF-IDF features and scaled linguistic features.
    
    Args:
        df: DataFrame with tfidf_features and linguistic_features_scaled
    
    Returns:
        DataFrame with combined features column
    """
    print("  Combining TF-IDF and linguistic features...")
    
    # Combine features using VectorAssembler
    assembler = VectorAssembler(
        inputCols=["tfidf_features", "linguistic_features_scaled"],
        outputCol="features"
    )
    
    df = assembler.transform(df)
    
    print("    ✓ Features combined")
    
    return df


def get_feature_summary(df: DataFrame) -> dict:
    """
    Get summary of feature engineering results.
    
    Args:
        df: DataFrame with features column
    
    Returns:
        Dictionary with feature summary
    """
    summary = {}
    
    # Count rows
    summary["total_rows"] = df.count()
    
    # Check if features column exists
    if "features" in df.columns:
        # Get feature vector size (sample first row)
        sample = df.select("features").first()
        if sample:
            feature_size = len(sample["features"].toArray())
            summary["feature_vector_size"] = feature_size
        else:
            summary["feature_vector_size"] = 0
    else:
        summary["feature_vector_size"] = 0
    
    return summary

