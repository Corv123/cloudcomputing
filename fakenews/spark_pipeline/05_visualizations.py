"""
Stage 5: Visualizations
Generate visualizations from aggregated data
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from typing import Optional
import os


def create_visualizations(
    aggregations: dict,
    output_dir: str = "visualizations/",
    show_plots: bool = False
) -> list:
    """
    Create visualizations from aggregated data.
    
    Args:
        aggregations: Dictionary with aggregated DataFrames
        output_dir: Directory to save plots
        show_plots: Whether to display plots (default: False)
    
    Returns:
        List of saved plot file paths
    """
    print("=" * 80)
    print("STAGE 5: VISUALIZATIONS")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = []
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Label Distribution
    if "label_distribution" in aggregations:
        print("  Creating label distribution plot...")
        # Use collect() instead of toPandas() for compatibility
        import pandas as pd
        label_rows = aggregations["label_distribution"].collect()
        label_pd = pd.DataFrame([row.asDict() for row in label_rows])
        
        plt.figure(figsize=(10, 6))
        plt.bar(label_pd['label'], label_pd['count'], color=['#2ecc71', '#e74c3c'])
        plt.title('Label Distribution (0=Real, 1=Fake)', fontsize=16, fontweight='bold')
        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks([0, 1], ['Real (0)', 'Fake (1)'])
        plt.grid(axis='y', alpha=0.3)
        
        for i, row in label_pd.iterrows():
            plt.text(row['label'], row['count'], f"{row['count']:,}", 
                    ha='center', va='bottom', fontsize=10)
        
        plot_path = os.path.join(output_dir, "label_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot_path)
        print(f"    ✓ Saved: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 2. Source Distribution
    if "source_distribution" in aggregations:
        print("  Creating source distribution plot...")
        # Use collect() instead of toPandas() for compatibility
        import pandas as pd
        source_rows = aggregations["source_distribution"].collect()
        source_pd = pd.DataFrame([row.asDict() for row in source_rows])
        
        plt.figure(figsize=(12, 8))
        plt.barh(source_pd['source'], source_pd['count'], color='#3498db')
        plt.title('Source Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Source', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        for i, row in source_pd.iterrows():
            plt.text(row['count'], i, f"{row['count']:,}", 
                    ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "source_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot_path)
        print(f"    ✓ Saved: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 3. Text Length Distribution (if we have the data)
    # Note: This would require sampling from the full DataFrame
    # For now, we'll skip this or implement it separately
    
    print(f"✓ Generated {len(saved_plots)} visualizations")
    print("=" * 80)
    print()
    
    return saved_plots

