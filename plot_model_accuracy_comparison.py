#!/usr/bin/env python3
"""
Model Accuracy Comparison Plotter
=================================

This script creates a comparison plot between FedAvg and DAGFLTrust models
showing their accuracy performance under different percentages of malicious attackers.

The plot follows the same visual style as the DAGTrustFL analysis suite.

Author: DAGTrustFL Analysis Suite
Date: 2025-09-01
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib with Roboto font and better looking plots
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

class ModelAccuracyPlotter:
    """Plotter for model accuracy comparison."""
    
    def __init__(self, output_dir="model_accuracy_comparison"):
        """
        Initialize the plotter.
        
        Args:
            output_dir (str): Directory to save the plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define malicious attacker percentages (x-axis)
        self.malicious_percentages = [0, 2, 4, 10, 20, 30, 40, 50]
        
        # Define FedAvg accuracy values (y-axis) - RED LINE
        self.fedavg_accuracy = [92.0, 91.1, 90.2, 89.3, 88.15, 88.05, 87.8, 85.8]

        # Define DAGFLTrust accuracy values (y-axis) - GREEN LINE
        # For now all set to 100% as requested
        self.dagfltrust_accuracy = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        
        # Color scheme - same as DAGTrustFL suite
        self.colors = {
            'fedavg': '#e74c3c',      # Red
            'dagfltrust': '#27ae60',   # Green
            'grid': '#bdc3c7',        # Light gray
            'background': '#ecf0f1'   # Very light gray
        }
        
    def create_accuracy_comparison_plot(self):
        """Create the main accuracy comparison plot."""
        # Set up the plot with 8x6 dimensions like lambda_metrics
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot FedAvg line (red) with larger markers and lines like lambda_metrics
        ax.plot(self.malicious_percentages, self.fedavg_accuracy, 
               'o-', linewidth=4, markersize=14, color=self.colors['fedavg'], 
               label='FedAvg', alpha=0.9)
        
        # Plot DAGFLTrust line (green) with larger markers and lines like lambda_metrics
        ax.plot(self.malicious_percentages, self.dagfltrust_accuracy, 
               's-', linewidth=4, markersize=14, color=self.colors['dagfltrust'], 
               label='DAGFLTrust', alpha=0.9)
        
        # Customize the plot
        ax.set_xlabel('Malicious Attackers Percentage (%)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Model Accuracy (%)', fontsize=20, fontweight='bold')
        
        # Add grid like lambda_metrics
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set axis limits and ticks
        ax.set_xlim(-2, 52)
        ax.set_ylim(80, 102)
        ax.set_xticks(self.malicious_percentages)
        ax.set_xticklabels([f'{p}%' for p in self.malicious_percentages])
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Add legend like lambda_metrics
        ax.legend(fontsize=16, loc='lower right', frameon=True, fancybox=True, shadow=True)
        
        # Set background color
        ax.set_facecolor(self.colors['background'])
        fig.patch.set_facecolor('white')
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        plot_filename = self.output_dir / 'model_accuracy_comparison.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Model accuracy comparison plot saved: {plot_filename}")
        
        # Also save as PDF
        pdf_filename = self.output_dir / 'model_accuracy_comparison.pdf'
        plt.savefig(pdf_filename, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Model accuracy comparison PDF saved: {pdf_filename}")
        
        # Show the plot
        plt.show()
        plt.close()
    
    def create_detailed_analysis_plot(self):
        """Create a more detailed analysis plot with additional information."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Main comparison
        ax1.plot(self.malicious_percentages, self.fedavg_accuracy, 
                'o-', color=self.colors['fedavg'], linewidth=3, markersize=8,
                label='FedAvg', markerfacecolor='white', markeredgewidth=2)
        
        ax1.plot(self.malicious_percentages, self.dagfltrust_accuracy, 
                'o-', color=self.colors['dagfltrust'], linewidth=3, markersize=8,
                label='DAGFLTrust', markerfacecolor='white', markeredgewidth=2)
        
        ax1.set_xlabel('Malicious Attackers Percentage (%)', fontweight='bold')
        ax1.set_ylabel('Model Accuracy (%)', fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_xlim(-2, 52)
        ax1.set_ylim(80, 102)
        ax1.set_xticks(self.malicious_percentages)
        ax1.set_xticklabels([f'{p}%' for p in self.malicious_percentages])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_facecolor(self.colors['background'])
        
        # Right plot: Performance difference
        accuracy_diff = [d - f for d, f in zip(self.dagfltrust_accuracy, self.fedavg_accuracy)]
        
        bars = ax2.bar(self.malicious_percentages, accuracy_diff, 
                      color=self.colors['dagfltrust'], alpha=0.7, 
                      edgecolor=self.colors['dagfltrust'], linewidth=2)
        
        ax2.set_xlabel('Malicious Attackers Percentage (%)', fontweight='bold')
        ax2.set_ylabel('Accuracy Advantage (%)', fontweight='bold')
        ax2.set_title('DAGFLTrust Advantage over FedAvg', fontweight='bold')
        ax2.set_xticks(self.malicious_percentages)
        ax2.set_xticklabels([f'{p}%' for p in self.malicious_percentages])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor(self.colors['background'])
        
        # Add value labels on bars
        for bar, diff in zip(bars, accuracy_diff):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'+{diff:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Model Performance Analysis: FedAvg vs DAGFLTrust', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the detailed plot
        detailed_filename = self.output_dir / 'detailed_model_comparison.png'
        plt.savefig(detailed_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Detailed analysis plot saved: {detailed_filename}")
        plt.close()
    
    def print_summary_statistics(self):
        """Print summary statistics about the comparison."""
        print("\n" + "="*60)
        print("MODEL ACCURACY COMPARISON SUMMARY")
        print("="*60)
        
        print("\nFedAvg Performance:")
        for perc, acc in zip(self.malicious_percentages, self.fedavg_accuracy):
            print(f"  {perc}% malicious: {acc}% accuracy")
        
        print("\nDAGFLTrust Performance:")
        for perc, acc in zip(self.malicious_percentages, self.dagfltrust_accuracy):
            print(f"  {perc}% malicious: {acc}% accuracy")
        
        print("\nPerformance Analysis:")
        avg_fedavg = sum(self.fedavg_accuracy) / len(self.fedavg_accuracy)
        avg_dagfltrust = sum(self.dagfltrust_accuracy) / len(self.dagfltrust_accuracy)
        
        print(f"  Average FedAvg accuracy: {avg_fedavg:.1f}%")
        print(f"  Average DAGFLTrust accuracy: {avg_dagfltrust:.1f}%")
        print(f"  Average improvement: {avg_dagfltrust - avg_fedavg:.1f}%")
        
        # Calculate degradation
        fedavg_degradation = self.fedavg_accuracy[0] - self.fedavg_accuracy[-1]
        dagfltrust_degradation = self.dagfltrust_accuracy[0] - self.dagfltrust_accuracy[-1]
        
        print(f"\nRobustness Analysis (0% to 50% malicious):")
        print(f"  FedAvg degradation: {fedavg_degradation:.1f}%")
        print(f"  DAGFLTrust degradation: {dagfltrust_degradation:.1f}%")
        print(f"  Robustness advantage: {fedavg_degradation - dagfltrust_degradation:.1f}%")
        
        print("="*60)
    
    def update_dagfltrust_values(self, new_values):
        """
        Update DAGFLTrust accuracy values.
        
        Args:
            new_values (list): List of 6 accuracy values for 0%, 10%, 20%, 30%, 40%, 50%
        """
        if len(new_values) != 6:
            raise ValueError("Must provide exactly 6 values for percentages 0%, 10%, 20%, 30%, 40%, 50%")
        
        self.dagfltrust_accuracy = new_values
        print("✓ DAGFLTrust accuracy values updated:")
        for perc, acc in zip(self.malicious_percentages, self.dagfltrust_accuracy):
            print(f"  {perc}%: {acc}%")

def main():
    """Main function to create the accuracy comparison plots."""
    print("🎨 Starting Model Accuracy Comparison Plotting")
    print("=" * 50)
    
    try:
        # Initialize plotter
        plotter = ModelAccuracyPlotter()
        
        # Print current values
        print("\nCurrent values:")
        print("FedAvg accuracy:", plotter.fedavg_accuracy)
        print("DAGFLTrust accuracy:", plotter.dagfltrust_accuracy)
        
        # Create plots
        plotter.create_accuracy_comparison_plot()
        plotter.create_detailed_analysis_plot()
        
        # Print summary
        plotter.print_summary_statistics()
        
        print(f"\n✅ All plots saved to: {plotter.output_dir}")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
