"""
Master Script to Run All Experiments
Executes full training, baseline comparisons, and ablation studies
"""
import os
import sys
import subprocess
from datetime import datetime

def run_script(script_name: str, description: str):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Script: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with exception: {e}")
        return False

def main():
    """Run all experiments in sequence"""
    print("\n" + "=" * 80)
    print("MASTER EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    experiments = [
        ("run_full_training.py", "Full Model Training with Detailed Metrics"),
        ("run_baseline_comparisons.py", "Baseline Model Comparisons"),
        ("run_ablation_study.py", "Ablation Study")
    ]
    
    results = {}
    
    for script, description in experiments:
        success = run_script(script, description)
        results[description] = "SUCCESS" if success else "FAILED"
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for description, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {description}: {status}")
    
    print("\n" + "=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check if any failed
    if "FAILED" in results.values():
        print("\n⚠️  Some experiments failed. Check logs above for details.")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")
        print("✓ Results saved in ./results/ directory")

if __name__ == "__main__":
    main()

