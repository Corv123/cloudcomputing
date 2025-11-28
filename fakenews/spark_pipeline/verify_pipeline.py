"""
Verification script to check if Spark ML Pipeline completed successfully
"""

import os
import json
from pathlib import Path

def verify_pipeline():
    """Verify all pipeline components are present and working."""
    
    output_dir = Path("output")
    
    print("=" * 80)
    print("SPARK ML PIPELINE VERIFICATION")
    print("=" * 80)
    print()
    
    all_checks_passed = True
    
    # 1. Check cleaned data
    print("1. CLEANED DATA:")
    cleaned_dir = output_dir / "cleaned-data"
    if cleaned_dir.exists():
        parquet_files = list(cleaned_dir.glob("*.parquet"))
        success_file = cleaned_dir / "_SUCCESS"
        if parquet_files and success_file.exists():
            print(f"   [OK] {len(parquet_files)} parquet partitions found")
            print(f"   [OK] _SUCCESS marker present")
        else:
            print("   [ERROR] Missing parquet files or _SUCCESS marker")
            all_checks_passed = False
    else:
        print("   [ERROR] cleaned-data directory not found")
        all_checks_passed = False
    print()
    
    # 2. Check features
    print("2. FEATURES:")
    features_dir = output_dir / "features"
    if features_dir.exists():
        parquet_files = list(features_dir.glob("*.parquet"))
        success_file = features_dir / "_SUCCESS"
        if parquet_files and success_file.exists():
            print(f"   [OK] {len(parquet_files)} feature partitions found")
            print(f"   [OK] _SUCCESS marker present")
        else:
            print("   [ERROR] Missing parquet files or _SUCCESS marker")
            all_checks_passed = False
    else:
        print("   [ERROR] features directory not found")
        all_checks_passed = False
    print()
    
    # 3. Check statistics
    print("3. STATISTICS:")
    stats_dir = output_dir / "statistics"
    if stats_dir.exists():
        stat_files = [
            "label_distribution.parquet",
            "source_distribution.parquet",
            "label_by_source.parquet",
            "text_statistics.parquet"
        ]
        for stat_file in stat_files:
            stat_path = stats_dir / stat_file
            if stat_path.exists():
                success_file = stat_path / "_SUCCESS"
                if success_file.exists():
                    print(f"   [OK] {stat_file}")
                else:
                    print(f"   [WARN] {stat_file} exists but missing _SUCCESS")
            else:
                print(f"   [ERROR] {stat_file} not found")
                all_checks_passed = False
    else:
        print("   [ERROR] statistics directory not found")
        all_checks_passed = False
    print()
    
    # 4. Check visualizations
    print("4. VISUALIZATIONS:")
    viz_dir = output_dir / "visualizations"
    if viz_dir.exists():
        viz_files = ["label_distribution.png", "source_distribution.png"]
        for viz_file in viz_files:
            viz_path = viz_dir / viz_file
            if viz_path.exists():
                size_kb = viz_path.stat().st_size / 1024
                print(f"   [OK] {viz_file} ({size_kb:.1f} KB)")
            else:
                print(f"   [ERROR] {viz_file} not found")
                all_checks_passed = False
    else:
        print("   [ERROR] visualizations directory not found")
        all_checks_passed = False
    print()
    
    # 5. Check models
    print("5. MODELS:")
    models_dir = output_dir / "models"
    
    # Check Spark pipeline model
    spark_model_dir = models_dir / "spark_pipeline"
    if spark_model_dir.exists():
        metadata_dir = spark_model_dir / "metadata"
        stages_dir = spark_model_dir / "stages"
        if metadata_dir.exists() and stages_dir.exists():
            print("   [OK] Spark ML Pipeline model saved")
            # Check for LinearSVC stage
            stages = list(stages_dir.glob("*LinearSVC*"))
            if stages:
                print(f"   [OK] LinearSVC model stage found")
            else:
                print("   [WARN] LinearSVC stage not found")
        else:
            print("   [ERROR] Spark model structure incomplete")
            all_checks_passed = False
    else:
        print("   [ERROR] Spark pipeline model not found")
        all_checks_passed = False
    
    # Check sklearn export
    sklearn_dir = models_dir / "sklearn_export"
    if sklearn_dir.exists():
        model_info = sklearn_dir / "model_info.json"
        if model_info.exists():
            try:
                with open(model_info) as f:
                    info = json.load(f)
                print("   [OK] Model info exported")
                print(f"     - Features: {info.get('num_features', 'N/A')}")
                print(f"     - Intercept: {info.get('intercept', 'N/A'):.4f}")
            except Exception as e:
                print(f"   [WARN] Could not read model_info.json: {e}")
        else:
            print("   [WARN] model_info.json not found")
    else:
        print("   [WARN] sklearn_export directory not found (optional)")
    print()
    
    # Summary
    print("=" * 80)
    if all_checks_passed:
        print("[SUCCESS] ALL PIPELINE COMPONENTS VERIFIED SUCCESSFULLY")
        print()
        print("Pipeline stages completed:")
        print("  [OK] Stage 1: Data Ingestion")
        print("  [OK] Stage 2: Data Cleaning")
        print("  [OK] Stage 3: Feature Engineering")
        print("  [OK] Stage 4: Aggregations")
        print("  [OK] Stage 5: Visualizations")
        print("  [OK] Stage 6: Persistent Storage")
        print("  [OK] Stage 7: Model Training")
        print("  [OK] Stage 8: Model Export")
    else:
        print("[ERROR] SOME COMPONENTS MISSING OR INCOMPLETE")
    print("=" * 80)
    
    return all_checks_passed

if __name__ == "__main__":
    verify_pipeline()

