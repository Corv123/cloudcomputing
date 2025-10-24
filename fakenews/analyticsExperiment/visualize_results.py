"""
Results Visualization Script
Reads experiment JSON files and generates readable reports
"""
import os
import json
import glob
from datetime import datetime

RESULTS_DIR = "./results"

def find_latest_results():
    """Find the most recent result files"""
    result_files = {
        "training": glob.glob(os.path.join(RESULTS_DIR, "training_results_*.json")),
        "baseline": glob.glob(os.path.join(RESULTS_DIR, "baseline_comparison_*.json")),
        "ablation": glob.glob(os.path.join(RESULTS_DIR, "ablation_study_*.json"))
    }
    
    latest = {}
    for exp_type, files in result_files.items():
        if files:
            latest[exp_type] = max(files, key=os.path.getctime)
    
    return latest

def print_training_results(filepath: str):
    """Print formatted training results"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("FULL TRAINING RESULTS")
    print("=" * 80)
    print(f"File: {os.path.basename(filepath)}")
    print(f"Date: {data['experiment_info']['date']}")
    
    print(f"\nDataset:")
    print(f"  Total samples:   {data['dataset_info']['total_samples']}")
    print(f"  Real samples:    {data['dataset_info']['real_samples']}")
    print(f"  Fake samples:    {data['dataset_info']['fake_samples']}")
    print(f"  Training split:  {data['dataset_info']['train_samples']}")
    print(f"  Test split:      {data['dataset_info']['test_samples']}")
    
    print(f"\nFeatures:")
    print(f"  TF-IDF features: {data['feature_info']['tfidf_features']}")
    print(f"  Enhanced features: {data['feature_info']['enhanced_features']}")
    print(f"  Total features:  {data['feature_info']['total_features']}")
    
    print(f"\nTraining:")
    print(f"  Model type:      {data['training_info']['model_type']}")
    print(f"  Training time:   {data['training_info']['training_time_seconds']:.2f}s "
          f"({data['training_info']['training_time_minutes']:.2f} min)")
    
    perf = data['test_performance']
    print(f"\nTest Performance:")
    print(f"  Accuracy:        {perf['accuracy']:.4f}")
    print(f"  Precision:       {perf['precision']:.4f}")
    print(f"  Recall:          {perf['recall']:.4f}")
    print(f"  F1-Score:        {perf['f1_score']:.4f}")
    print(f"  ROC-AUC:         {perf['roc_auc']:.4f}")
    
    cm = perf['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Negative:   {cm['true_negative']:5d} (Credible correctly identified)")
    print(f"  False Positive:  {cm['false_positive']:5d} (Credible misclassified)")
    print(f"  False Negative:  {cm['false_negative']:5d} (Sensational misclassified)")
    print(f"  True Positive:   {cm['true_positive']:5d} (Sensational correctly identified)")
    
    cv = data['cross_validation']
    print(f"\nCross-Validation (5-fold):")
    print(f"  Fold scores:     {', '.join([f'{s:.4f}' for s in cv['cv_scores']])}")
    print(f"  Mean F1:         {cv['cv_mean']:.4f}")
    print(f"  Std Dev:         {cv['cv_std']:.4f}")
    print(f"  Range:           [{cv['cv_min']:.4f}, {cv['cv_max']:.4f}]")

def print_baseline_results(filepath: str):
    """Print formatted baseline comparison results"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)
    print(f"File: {os.path.basename(filepath)}")
    
    # Sort by F1 score
    sorted_data = sorted(data, key=lambda x: x['f1_score'], reverse=True)
    
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    print("-" * 85)
    
    for result in sorted_data:
        roc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] is not None else "N/A"
        print(f"{result['model_name']:<25} {result['accuracy']:<10.4f} "
              f"{result['precision']:<10.4f} {result['recall']:<10.4f} "
              f"{result['f1_score']:<10.4f} {roc_str:<10}")
    
    print("\nBest Model: " + sorted_data[0]['model_name'])
    print(f"  F1-Score: {sorted_data[0]['f1_score']:.4f}")

def print_ablation_results(filepath: str):
    """Print formatted ablation study results"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"File: {os.path.basename(filepath)}")
    
    # Sort by F1 score
    sorted_data = sorted(data, key=lambda x: x['f1_score'], reverse=True)
    
    print(f"\n{'Configuration':<35} {'Accuracy':<10} {'F1':<10} {'Δ F1':<10} {'Δ F1 %':<10}")
    print("-" * 75)
    
    for result in sorted_data:
        delta_str = f"{result.get('delta_f1', 0):.4f}" if 'delta_f1' in result else "baseline"
        delta_pct = f"{result.get('delta_f1_pct', 0):+.2f}%" if 'delta_f1_pct' in result else "--"
        
        print(f"{result['experiment_name']:<35} {result['accuracy']:<10.4f} "
              f"{result['f1_score']:<10.4f} {delta_str:<10} {delta_pct:<10}")
    
    # Find baseline
    baseline = next((r for r in data if 'Full Model' in r['experiment_name']), None)
    if baseline:
        print(f"\nBaseline (Full Model) F1: {baseline['f1_score']:.4f}")
        
        # Find biggest drops
        drops = [(r['experiment_name'], r.get('delta_f1', 0)) 
                 for r in data if 'delta_f1' in r and r['delta_f1'] < 0]
        drops.sort(key=lambda x: x[1])
        
        if drops:
            print("\nMost Important Feature Groups (by F1 drop when removed):")
            for i, (name, delta) in enumerate(drops[:3], 1):
                print(f"  {i}. {name}: {delta:.4f} ({delta/baseline['f1_score']*100:.1f}%)")

def generate_summary_report(latest_files: dict):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {RESULTS_DIR}")
    
    print("\nAvailable Results:")
    for exp_type, filepath in latest_files.items():
        print(f"  ✓ {exp_type.capitalize()}: {os.path.basename(filepath)}")
    
    if not latest_files:
        print("\n⚠️  No result files found. Please run experiments first.")
        return
    
    # Print each result type
    if 'training' in latest_files:
        print_training_results(latest_files['training'])
    
    if 'baseline' in latest_files:
        print_baseline_results(latest_files['baseline'])
    
    if 'ablation' in latest_files:
        print_ablation_results(latest_files['ablation'])
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

def main():
    """Main execution"""
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        print("Please run experiments first using run_all_experiments.py")
        return
    
    latest_files = find_latest_results()
    
    if not latest_files:
        print("No experiment results found.")
        print("Please run experiments first using run_all_experiments.py")
        return
    
    generate_summary_report(latest_files)

if __name__ == "__main__":
    main()

