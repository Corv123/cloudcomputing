"""
Stage 8: Model Export
Export trained model for deployment (Spark Pipeline and scikit-learn format)
"""

from pyspark.ml import Pipeline
from typing import Optional
import os
import joblib
import numpy as np
from pyspark.ml.linalg import Vectors


def save_spark_pipeline(
    pipeline: Pipeline,
    output_path: str = "output/models/spark_pipeline/",
    use_s3: bool = False
):
    """
    Save complete Spark ML Pipeline.
    
    Args:
        pipeline: Fitted Spark ML Pipeline
        output_path: Output path (local or S3)
        use_s3: Whether to save to S3
    """
    print("=" * 80)
    print("STAGE 8: MODEL EXPORT")
    print("=" * 80)
    
    print(f"  Saving Spark ML Pipeline to: {output_path}")
    
    if use_s3:
        # For S3, we need to save locally first, then upload
        local_path = "/tmp/spark_pipeline_temp"
        pipeline.write().overwrite().save(local_path)
        
        # Upload to S3 (would need boto3)
        print("    ⚠️ S3 upload not implemented - save locally first")
    else:
        # Save locally
        os.makedirs(output_path, exist_ok=True)
        pipeline.write().overwrite().save(output_path)
        print(f"    ✓ Spark pipeline saved to {output_path}")
    
    print()


def export_to_sklearn(
    pipeline: Pipeline,
    tfidf_pipeline,
    scaler_model,
    output_path: str = "output/models/sklearn_export/",
    use_s3: bool = False
):
    """
    Export model components to scikit-learn format for Lambda deployment.
    
    Note: This is a simplified export. Full conversion requires:
    1. Extracting model weights from Spark LinearSVC
    2. Converting HashingTF to scikit-learn TfidfVectorizer (complex)
    3. Exporting scaler
    
    For now, we'll export what we can and note the limitations.
    
    Args:
        pipeline: Fitted Spark ML Pipeline
        tfidf_pipeline: TF-IDF pipeline (for reference)
        scaler_model: Scaler model (for export)
        output_path: Output path
        use_s3: Whether to save to S3
    """
    print("  Exporting to scikit-learn format...")
    print("    ⚠️ Note: Full conversion requires manual steps")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Extract model from pipeline
    model = pipeline.stages[-1]  # Last stage is the model
    
    # Export scaler (if available)
    if scaler_model:
        scaler_path = os.path.join(output_path, "scaler_comprehensive.joblib")
        # Note: Spark StandardScalerModel needs conversion to scikit-learn StandardScaler
        # For now, we'll save a note about this
        print(f"    ⚠️ Scaler conversion needed - save manually")
    
    # Extract model weights
    try:
        coefficients = model.coefficients.toArray()
        intercept = model.intercept
        
        # Save model info (for reference)
        model_info = {
            "coefficients_shape": coefficients.shape,
            "intercept": float(intercept),
            "num_features": len(coefficients),
            "note": "This is Spark LinearSVC model. Full conversion to scikit-learn requires: "
                   "1. Converting HashingTF to TfidfVectorizer (vocabulary mapping) "
                   "2. Converting Spark StandardScaler to scikit-learn StandardScaler "
                   "3. Recreating the model with scikit-learn LinearSVC"
        }
        
        import json
        info_path = os.path.join(output_path, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"    ✓ Model info saved to {info_path}")
        print(f"    ⚠️ Full scikit-learn export requires manual conversion")
        
    except Exception as e:
        print(f"    ✗ Error extracting model: {e}")
    
    print()
    print("=" * 80)
    print()
    print("IMPORTANT: For Lambda deployment, you need to:")
    print("  1. Train model using current pandas/scikit-learn approach")
    print("  2. OR manually convert Spark model to scikit-learn format")
    print("  3. Export vectorizer, scaler, and model as .joblib files")
    print()


def load_spark_pipeline(
    input_path: str,
    use_s3: bool = False
) -> Pipeline:
    """
    Load Spark ML Pipeline from storage.
    
    Args:
        input_path: Path to saved pipeline
        use_s3: Whether to load from S3
    
    Returns:
        Loaded Spark ML Pipeline
    """
    if use_s3:
        # Download from S3 first
        print("    ⚠️ S3 loading not implemented - load locally")
        return None
    
    pipeline = Pipeline.load(input_path)
    return pipeline

