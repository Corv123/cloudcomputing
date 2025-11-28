"""
Stage 7: Model Training
Train ML model using Spark ML Pipeline
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.sql import DataFrame
from typing import Tuple, Optional


def create_ml_pipeline() -> Pipeline:
    """
    Create complete ML pipeline (feature engineering + model).
    Note: Feature engineering stages are applied separately,
    so this pipeline only includes the model.
    
    Returns:
        ML Pipeline with LinearSVC model
    """
    # Model stage
    svc = LinearSVC(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.1
    )
    
    # Pipeline (only model, features already created)
    pipeline = Pipeline(stages=[svc])
    
    return pipeline


def train_model(
    train_df: DataFrame,
    test_df: Optional[DataFrame] = None,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[Pipeline, DataFrame, dict]:
    """
    Train the ML model.
    
    Args:
        train_df: Training DataFrame with features
        test_df: Optional test DataFrame (if None, will split from train_df)
        test_size: Test set size (if test_df is None)
        seed: Random seed
    
    Returns:
        Tuple of (fitted_pipeline, test_transformed, metrics)
    """
    print("=" * 80)
    print("STAGE 7: MODEL TRAINING")
    print("=" * 80)
    
    # Split if test_df not provided
    if test_df is None:
        print(f"  Splitting data (test_size={test_size})...")
        train_df, test_df = train_df.randomSplit([1.0 - test_size, test_size], seed=seed)
        print(f"    Training set: {train_df.count():,} rows")
        print(f"    Test set: {test_df.count():,} rows")
    
    # Create pipeline
    print("  Creating ML pipeline...")
    pipeline = create_ml_pipeline()
    
    # Fit pipeline
    print("  Training model (this may take a while)...")
    fitted_pipeline = pipeline.fit(train_df)
    print("    ✓ Model trained")
    
    # Transform test data
    print("  Evaluating on test set...")
    test_transformed = fitted_pipeline.transform(test_df)
    
    # Evaluate
    metrics = evaluate_model(test_transformed)
    
    print("=" * 80)
    print()
    
    return fitted_pipeline, test_transformed, metrics


def evaluate_model(test_transformed: DataFrame) -> dict:
    """
    Evaluate model performance.
    
    Args:
        test_transformed: Test DataFrame with predictions
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Binary classification metrics
    try:
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction"
        )
        auc = evaluator.evaluate(test_transformed)
        metrics["roc_auc"] = auc
        print(f"    ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"    ⚠️ ROC-AUC calculation failed: {e}")
        metrics["roc_auc"] = None
    
    # Accuracy
    try:
        accuracy_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = accuracy_eval.evaluate(test_transformed)
        metrics["accuracy"] = accuracy
        print(f"    Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"    ⚠️ Accuracy calculation failed: {e}")
        metrics["accuracy"] = None
    
    # Precision
    try:
        precision_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        precision = precision_eval.evaluate(test_transformed)
        metrics["precision"] = precision
        print(f"    Precision: {precision:.4f}")
    except Exception as e:
        print(f"    ⚠️ Precision calculation failed: {e}")
        metrics["precision"] = None
    
    # Recall
    try:
        recall_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        recall = recall_eval.evaluate(test_transformed)
        metrics["recall"] = recall
        print(f"    Recall: {recall:.4f}")
    except Exception as e:
        print(f"    ⚠️ Recall calculation failed: {e}")
        metrics["recall"] = None
    
    # F1-Score
    try:
        f1_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        f1 = f1_eval.evaluate(test_transformed)
        metrics["f1"] = f1
        print(f"    F1-Score: {f1:.4f}")
    except Exception as e:
        print(f"    ⚠️ F1-Score calculation failed: {e}")
        metrics["f1"] = None
    
    # Confusion matrix
    try:
        from pyspark.sql.functions import col
        import pandas as pd
        confusion = test_transformed.groupBy("label", "prediction").count().orderBy("label", "prediction")
        # Use collect() instead of toPandas() for compatibility
        confusion_rows = confusion.collect()
        confusion_pd = pd.DataFrame([row.asDict() for row in confusion_rows])
        print("    Confusion Matrix:")
        print("      Predicted")
        print("      Real  Fake")
        for _, row in confusion_pd.iterrows():
            label_name = "Real" if row["label"] == 0 else "Fake"
            pred_name = "Real" if row["prediction"] == 0 else "Fake"
            print(f"      {label_name} ({row['label']}) {pred_name} ({row['prediction']}): {row['count']:,}")
        metrics["confusion_matrix"] = confusion_pd.to_dict('records')
    except Exception as e:
        print(f"    ⚠️ Confusion matrix calculation failed: {e}")
        metrics["confusion_matrix"] = None
    
    return metrics

