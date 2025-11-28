"""
Main Orchestrator for Spark ML Pipeline
End-to-end machine learning workflow for sensationalism model training
"""

import argparse
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

# Import pipeline stages
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import using importlib to handle numeric module names
import importlib.util

# Stage 1: Data Ingestion
spec1 = importlib.util.spec_from_file_location("data_ingestion", os.path.join(current_dir, "01_data_ingestion.py"))
data_ingestion = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(data_ingestion)
read_all_datasets = data_ingestion.read_all_datasets
validate_datasets = data_ingestion.validate_datasets

# Stage 2: Data Cleaning
spec2 = importlib.util.spec_from_file_location("data_cleaning", os.path.join(current_dir, "02_data_cleaning.py"))
data_cleaning = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(data_cleaning)
clean_and_enrich_dataframe = data_cleaning.clean_and_enrich_dataframe
union_all_datasets = data_cleaning.union_all_datasets
get_data_summary = data_cleaning.get_data_summary
print_data_summary = data_cleaning.print_data_summary

# Stage 3: Feature Engineering
spec3 = importlib.util.spec_from_file_location("feature_engineering", os.path.join(current_dir, "03_feature_engineering.py"))
feature_engineering = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(feature_engineering)
create_tfidf_pipeline = feature_engineering.create_tfidf_pipeline
apply_linguistic_features = feature_engineering.apply_linguistic_features
scale_linguistic_features = feature_engineering.scale_linguistic_features
combine_all_features = feature_engineering.combine_all_features

# Stage 4: Aggregations
spec4 = importlib.util.spec_from_file_location("aggregations", os.path.join(current_dir, "04_aggregations.py"))
aggregations = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(aggregations)
perform_aggregations = aggregations.perform_aggregations

# Stage 5: Visualizations
spec5 = importlib.util.spec_from_file_location("visualizations", os.path.join(current_dir, "05_visualizations.py"))
visualizations = importlib.util.module_from_spec(spec5)
spec5.loader.exec_module(visualizations)
create_visualizations = visualizations.create_visualizations

# Stage 6: Persistent Storage
spec6 = importlib.util.spec_from_file_location("persistent_storage", os.path.join(current_dir, "06_persistent_storage.py"))
persistent_storage = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(persistent_storage)
write_cleaned_data = persistent_storage.write_cleaned_data
write_aggregations = persistent_storage.write_aggregations
write_features = persistent_storage.write_features
write_visualizations = persistent_storage.write_visualizations

# Stage 7: Model Training
spec7 = importlib.util.spec_from_file_location("model_training", os.path.join(current_dir, "07_model_training.py"))
model_training = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(model_training)
train_model = model_training.train_model

# Stage 8: Model Export
spec8 = importlib.util.spec_from_file_location("model_export", os.path.join(current_dir, "08_model_export.py"))
model_export = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(model_export)
save_spark_pipeline = model_export.save_spark_pipeline
export_to_sklearn = model_export.export_to_sklearn


def create_spark_session(app_name: str = "FakeNewsSensationalismTraining") -> SparkSession:
    """
    Create SparkSession with appropriate configuration.
    Handles Windows-specific issues.
    
    Args:
        app_name: Application name
    
    Returns:
        SparkSession instance
    """
    import platform
    
    # Base configuration
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    # Windows-specific fixes
    if platform.system() == "Windows":
        # Fix for Windows: Set HADOOP_HOME if not set
        import os
        if "HADOOP_HOME" not in os.environ:
            # Try to find winutils or set to a temp directory
            # For now, we'll use a workaround
            os.environ["HADOOP_HOME"] = os.path.join(os.path.expanduser("~"), "hadoop")
        
        # Additional Windows configurations
        builder = builder.config("spark.sql.warehouse.dir", "file:///C:/temp/spark-warehouse") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    
    # Create SparkSession
    try:
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")  # Reduce log verbosity
        return spark
    except Exception as e:
        print(f"✗ Error creating SparkSession: {e}")
        print("\nWindows Spark Setup Issues Detected!")
        print("\nTo fix this, you have several options:")
        print("\n1. Use WSL (Windows Subsystem for Linux):")
        print("   - Install WSL2")
        print("   - Run Spark inside WSL")
        print("\n2. Use Docker:")
        print("   - docker run -it apache/spark-py:latest")
        print("\n3. Use AWS EMR (Recommended for production):")
        print("   - Deploy to EMR cluster")
        print("   - No local setup needed")
        print("\n4. Install Hadoop/winutils for Windows:")
        print("   - Download winutils.exe")
        print("   - Set HADOOP_HOME environment variable")
        print("\nFor now, the pipeline is designed for Linux/EMR environments.")
        raise


def main():
    """
    Main orchestrator function.
    """
    parser = argparse.ArgumentParser(
        description="Spark ML Pipeline for Sensationalism Model Training"
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="datasets/",
        help="Directory containing dataset CSV files (default: datasets/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Output directory for results (default: output/)"
    )
    parser.add_argument(
        "--use_s3",
        action="store_true",
        help="Use S3 for input/output (requires S3 paths)"
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket path (e.g., s3://fakenews-ml-pipeline/)"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=50000,
        help="Maximum TF-IDF features (default: 50000)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip model training (only do data processing)"
    )
    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SPARK ML PIPELINE - SENSATIONALISM MODEL TRAINING")
    print("=" * 80)
    print()
    
    # Create SparkSession
    print("Initializing SparkSession...")
    spark = create_spark_session()
    print(f"✓ SparkSession created: {spark.version}")
    print()
    
    try:
        # =====================================================================
        # STAGE 1: DATA INGESTION
        # =====================================================================
        datasets = read_all_datasets(
            spark=spark,
            datasets_dir=args.datasets_dir,
            use_s3=args.use_s3,
            s3_bucket=args.s3_bucket
        )
        
        if not datasets:
            raise ValueError("No datasets loaded!")
        
        if not validate_datasets(datasets):
            raise ValueError("Dataset validation failed!")
        
        # =====================================================================
        # STAGE 2: DATA CLEANING & ENRICHMENT
        # =====================================================================
        print("=" * 80)
        print("STAGE 2: DATA CLEANING & ENRICHMENT")
        print("=" * 80)
        
        cleaned_datasets = []
        for df, source, label in datasets:
            cleaned_df = clean_and_enrich_dataframe(df, source, label)
            if cleaned_df is not None:
                cleaned_datasets.append(cleaned_df)
        
        if not cleaned_datasets:
            raise ValueError("No cleaned datasets!")
        
        # Union all datasets
        combined_df = union_all_datasets(cleaned_datasets)
        
        # Get and print summary
        summary = get_data_summary(combined_df)
        print_data_summary(summary)
        
        # =====================================================================
        # STAGE 3: FEATURE ENGINEERING
        # =====================================================================
        print("=" * 80)
        print("STAGE 3: FEATURE ENGINEERING")
        print("=" * 80)
        
        # Apply TF-IDF pipeline
        print("  Applying TF-IDF feature engineering...")
        tfidf_pipeline = create_tfidf_pipeline(max_features=args.max_features)
        tfidf_model = tfidf_pipeline.fit(combined_df)
        df_with_tfidf = tfidf_model.transform(combined_df)
        print("    ✓ TF-IDF features created")
        
        # Apply linguistic features
        df_with_ling = apply_linguistic_features(df_with_tfidf)
        
        # Scale linguistic features
        df_scaled, scaler_model = scale_linguistic_features(df_with_ling)
        
        # Combine all features
        df_final = combine_all_features(df_scaled)
        
        print(f"    ✓ Feature engineering complete")
        print(f"    Total features: {len(df_final.select('features').first()['features'].toArray())}")
        print()
        
        # =====================================================================
        # STAGE 4: AGGREGATIONS
        # =====================================================================
        aggregations = perform_aggregations(combined_df)
        
        # =====================================================================
        # STAGE 5: VISUALIZATIONS
        # =====================================================================
        plot_paths = []
        if not args.skip_visualizations:
            viz_output = os.path.join(args.output_dir, "visualizations/")
            plot_paths = create_visualizations(
                aggregations,
                output_dir=viz_output,
                show_plots=False
            )
        
        # =====================================================================
        # STAGE 6: PERSISTENT STORAGE
        # =====================================================================
        # Write cleaned data
        cleaned_output = os.path.join(args.output_dir, "cleaned-data/")
        write_cleaned_data(
            combined_df,
            output_path=cleaned_output,
            use_s3=args.use_s3,
            format="parquet"
        )
        
        # Write aggregations
        stats_output = os.path.join(args.output_dir, "statistics/")
        write_aggregations(
            aggregations,
            output_path=stats_output,
            use_s3=args.use_s3
        )
        
        # Write features
        features_output = os.path.join(args.output_dir, "features/")
        write_features(
            df_final,
            output_path=features_output,
            use_s3=args.use_s3
        )
        
        # Write visualizations
        if plot_paths:
            viz_output = os.path.join(args.output_dir, "visualizations/")
            write_visualizations(
                plot_paths,
                output_path=viz_output,
                use_s3=args.use_s3
            )
        
        # =====================================================================
        # STAGE 7: MODEL TRAINING
        # =====================================================================
        if not args.skip_training:
            fitted_pipeline, test_transformed, metrics = train_model(
                train_df=df_final,
                test_df=None,
                test_size=args.test_size,
                seed=args.seed
            )
            
            # =====================================================================
            # STAGE 8: MODEL EXPORT
            # =====================================================================
            # Save Spark pipeline
            model_output = os.path.join(args.output_dir, "models/spark_pipeline/")
            save_spark_pipeline(
                fitted_pipeline,
                output_path=model_output,
                use_s3=args.use_s3
            )
            
            # Export to scikit-learn (with limitations)
            sklearn_output = os.path.join(args.output_dir, "models/sklearn_export/")
            export_to_sklearn(
                fitted_pipeline,
                tfidf_model,
                scaler_model,
                output_path=sklearn_output,
                use_s3=args.use_s3
            )
            
            print("=" * 80)
            print("PIPELINE COMPLETE!")
            print("=" * 80)
            print()
            print("Final Metrics:")
            for metric, value in metrics.items():
                if value is not None and metric != "confusion_matrix":
                    print(f"  {metric}: {value:.4f}")
        else:
            print("=" * 80)
            print("PIPELINE COMPLETE (Training Skipped)")
            print("=" * 80)
            print()
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Stop SparkSession
        spark.stop()
        print("\n✓ SparkSession stopped")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

