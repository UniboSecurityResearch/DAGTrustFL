#!/usr/bin/env python3
"""
Percentages Metrics Plotter for DAGTrustFL
==========================================

This script creates detailed plots for percentage experiments analysis.
It focuses on visualizing how different malicious and corrupted percentage 
combinations affect system performance metrics.

Features:
- Individual metric plots with trend analysis
- Comparative analysis between malicious and corrupted impact
- 3D surface plots for comprehensive visualization
- Correlation analysis between percentages and performance

Author: DAGTrustFL Analysis Suite
Date: 2025-08-31
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class PercentagesMetricsPlotter:
    """Advanced plotter for percentage experiment metrics."""
    
    def __init__(self, csv_path="csv_data/percentages_comparison_data.csv"):
        """
        Initialize the plotter.
        
        Args:
            csv_path (str): Path to the percentages comparison CSV file
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path("percentages_metrics_plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.df = self.load_data()
        
        # Define color schemes
        self.colors = {
            'malicious': '#e74c3c',    # Red
            'corrupted': '#f39c12',    # Orange  
            'good': '#27ae60',         # Green
            'total': '#3498db',        # Blue
            'accuracy': '#9b59b6'      # Purple
        }
        
    def load_data(self):
        """Load percentage comparison data."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        print(f"📊 Loaded data for {len(df)} percentage configurations")
        
        # Sort by total percentage and malicious percentage for better visualization
        df = df.sort_values(['total_perc', 'malicious_perc'])
        
        return df
    
    def create_all_plots(self):
        """Create all percentage metrics plots."""
        print("🎨 Creating percentage metrics plots...")
        
        # Individual metric plots
        self.plot_fpr_analysis()
        self.plot_tpr_analysis() 
        self.plot_fnr_analysis()
        self.plot_accuracy_analysis()
        
        # Comparative plots
        self.plot_malicious_vs_corrupted_impact()
        self.plot_total_percentage_trends()
        
        # Advanced visualizations
        self.plot_3d_surface_plots()
        self.plot_correlation_matrix()
        self.plot_configuration_comparison()
        
        print(f"✅ All plots saved to: {self.output_dir}")
    
    def plot_fpr_analysis(self):
        """Create detailed FPR analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: FPR vs Malicious Percentage
        for total_perc in sorted(self.df['total_perc'].unique()):
            subset = self.df[self.df['total_perc'] == total_perc]
            ax1.plot(subset['malicious_perc'], subset['fpr'], 'o-', 
                    label=f'Total {total_perc}%', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Malicious Percentage (%)')
        ax1.set_ylabel('False Positive Rate (%)')
        ax1.set_title('FPR vs Malicious Percentage by Total Threat Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: FPR vs Corrupted Percentage
        for mal_perc in sorted(self.df['malicious_perc'].unique()):
            subset = self.df[self.df['malicious_perc'] == mal_perc]
            if len(subset) > 1:  # Only plot if we have multiple points
                ax2.plot(subset['corrupted_perc'], subset['fpr'], 'o-', 
                        label=f'Malicious {mal_perc}%', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Corrupted Percentage (%)')
        ax2.set_ylabel('False Positive Rate (%)')
        ax2.set_title('FPR vs Corrupted Percentage by Malicious Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FPR Scatter with Total Percentage
        scatter = ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], 
                             c=self.df['fpr'], s=100, cmap='Reds', alpha=0.7)
        ax3.set_xlabel('Malicious Percentage (%)')
        ax3.set_ylabel('Corrupted Percentage (%)')
        ax3.set_title('FPR Distribution Across Configurations')
        plt.colorbar(scatter, ax=ax3, label='FPR (%)')
        
        # Plot 4: FPR Box Plot by Total Percentage
        total_percs = sorted(self.df['total_perc'].unique())
        fpr_data = [self.df[self.df['total_perc'] == tp]['fpr'].values for tp in total_percs]
        ax4.boxplot(fpr_data, labels=[f'{tp}%' for tp in total_percs])
        ax4.set_xlabel('Total Threat Percentage (%)')
        ax4.set_ylabel('False Positive Rate (%)')
        ax4.set_title('FPR Distribution by Total Threat Level')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('False Positive Rate (FPR) Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fpr_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ FPR analysis plot saved")
    
    def plot_tpr_analysis(self):
        """Create detailed TPR analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: TPR vs Malicious Percentage
        for total_perc in sorted(self.df['total_perc'].unique()):
            subset = self.df[self.df['total_perc'] == total_perc]
            ax1.plot(subset['malicious_perc'], subset['tpr_malicious'], 'o-', 
                    label=f'Total {total_perc}%', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Malicious Percentage (%)')
        ax1.set_ylabel('True Positive Rate (%)')
        ax1.set_title('TPR vs Malicious Percentage by Total Threat Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: TPR vs Corrupted Percentage
        for mal_perc in sorted(self.df['malicious_perc'].unique()):
            subset = self.df[self.df['malicious_perc'] == mal_perc]
            if len(subset) > 1:
                ax2.plot(subset['corrupted_perc'], subset['tpr_malicious'], 'o-', 
                        label=f'Malicious {mal_perc}%', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Corrupted Percentage (%)')
        ax2.set_ylabel('True Positive Rate (%)')
        ax2.set_title('TPR vs Corrupted Percentage by Malicious Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: TPR Scatter with Total Percentage
        scatter = ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], 
                             c=self.df['tpr_malicious'], s=100, cmap='Greens', alpha=0.7)
        ax3.set_xlabel('Malicious Percentage (%)')
        ax3.set_ylabel('Corrupted Percentage (%)')
        ax3.set_title('TPR Distribution Across Configurations')
        plt.colorbar(scatter, ax=ax3, label='TPR (%)')
        
        # Plot 4: TPR vs Total Percentage Trend
        ax4.scatter(self.df['total_perc'], self.df['tpr_malicious'], 
                   c=self.df['malicious_perc'], s=100, cmap='plasma', alpha=0.7)
        ax4.set_xlabel('Total Threat Percentage (%)')
        ax4.set_ylabel('True Positive Rate (%)')
        ax4.set_title('TPR vs Total Threat Level')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('True Positive Rate (TPR) Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tpr_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ TPR analysis plot saved")
    
    def plot_fnr_analysis(self):
        """Create detailed FNR analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: FNR vs Malicious Percentage
        for total_perc in sorted(self.df['total_perc'].unique()):
            subset = self.df[self.df['total_perc'] == total_perc]
            ax1.plot(subset['malicious_perc'], subset['fnr_malicious'], 'o-', 
                    label=f'Total {total_perc}%', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Malicious Percentage (%)')
        ax1.set_ylabel('False Negative Rate (%)')
        ax1.set_title('FNR vs Malicious Percentage by Total Threat Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: FNR vs Corrupted Percentage
        for mal_perc in sorted(self.df['malicious_perc'].unique()):
            subset = self.df[self.df['malicious_perc'] == mal_perc]
            if len(subset) > 1:
                ax2.plot(subset['corrupted_perc'], subset['fnr_malicious'], 'o-', 
                        label=f'Malicious {mal_perc}%', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Corrupted Percentage (%)')
        ax2.set_ylabel('False Negative Rate (%)')
        ax2.set_title('FNR vs Corrupted Percentage by Malicious Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FNR Heatmap (only if we have enough variety in data)
        if len(self.df['malicious_perc'].unique()) > 1 and len(self.df['corrupted_perc'].unique()) > 1:
            try:
                pivot_data = self.df.pivot_table(values='fnr_malicious', 
                                                index='corrupted_perc', 
                                                columns='malicious_perc', 
                                                aggfunc='mean')
                im = ax3.imshow(pivot_data.values, cmap='Oranges', aspect='auto')
                ax3.set_xticks(range(len(pivot_data.columns)))
                ax3.set_xticklabels(pivot_data.columns)
                ax3.set_yticks(range(len(pivot_data.index)))
                ax3.set_yticklabels(pivot_data.index)
                ax3.set_xlabel('Malicious Percentage (%)')
                ax3.set_ylabel('Corrupted Percentage (%)')
                ax3.set_title('FNR Heatmap')
                plt.colorbar(im, ax=ax3, label='FNR (%)')
            except Exception as e:
                print(f"   ⚠️ Could not create FNR heatmap: {e}")
                # Fallback to scatter plot
                scatter = ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], 
                                     c=self.df['fnr_malicious'], s=100, cmap='Oranges', alpha=0.7)
                ax3.set_xlabel('Malicious Percentage (%)')
                ax3.set_ylabel('Corrupted Percentage (%)')
                ax3.set_title('FNR Distribution (Scatter)')
                plt.colorbar(scatter, ax=ax3, label='FNR (%)')
        else:
            # Simple scatter plot when we don't have enough variety for heatmap
            scatter = ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], 
                                 c=self.df['fnr_malicious'], s=100, cmap='Oranges', alpha=0.7)
            ax3.set_xlabel('Malicious Percentage (%)')
            ax3.set_ylabel('Corrupted Percentage (%)')
            ax3.set_title('FNR Distribution (Scatter)')
            plt.colorbar(scatter, ax=ax3, label='FNR (%)')
        
        # Plot 4: FNR Distribution
        ax4.hist(self.df['fnr_malicious'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(self.df['fnr_malicious'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.df["fnr_malicious"].mean():.1f}%')
        ax4.set_xlabel('False Negative Rate (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('FNR Distribution Across All Configurations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('False Negative Rate (FNR) Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fnr_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ FNR analysis plot saved")
    
    def plot_accuracy_analysis(self):
        """Create detailed accuracy analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy vs Total Percentage
        for mal_perc in sorted(self.df['malicious_perc'].unique()):
            subset = self.df[self.df['malicious_perc'] == mal_perc]
            ax1.plot(subset['total_perc'], subset['accuracy'], 'o-', 
                    label=f'Malicious {mal_perc}%', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Total Threat Percentage (%)')
        ax1.set_ylabel('System Accuracy (%)')
        ax1.set_title('Accuracy vs Total Threat Level by Malicious %')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Malicious/Corrupted Ratio
        self.df['mal_cor_ratio'] = self.df['malicious_perc'] / (self.df['corrupted_perc'] + 0.1)  # Add small value to avoid division by zero
        ax2.scatter(self.df['mal_cor_ratio'], self.df['accuracy'], 
                   c=self.df['total_perc'], s=100, cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Malicious/Corrupted Ratio')
        ax2.set_ylabel('System Accuracy (%)')
        ax2.set_title('Accuracy vs Malicious/Corrupted Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy Contour Plot (only if enough data points)
        if len(self.df) >= 4:
            try:
                # Create a grid for contour plot
                mal_range = np.linspace(self.df['malicious_perc'].min(), self.df['malicious_perc'].max(), 20)
                cor_range = np.linspace(self.df['corrupted_perc'].min(), self.df['corrupted_perc'].max(), 20)
                MAL, COR = np.meshgrid(mal_range, cor_range)
                
                # Interpolate accuracy values
                from scipy.interpolate import griddata
                points = self.df[['malicious_perc', 'corrupted_perc']].values
                values = self.df['accuracy'].values
                ACC = griddata(points, values, (MAL, COR), method='cubic')
                
                contour = ax3.contourf(MAL, COR, ACC, levels=15, cmap='RdYlGn')
                ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], 
                           c='black', s=50, alpha=0.8)
                ax3.set_xlabel('Malicious Percentage (%)')
                ax3.set_ylabel('Corrupted Percentage (%)')
                ax3.set_title('Accuracy Contour Map')
                plt.colorbar(contour, ax=ax3, label='Accuracy (%)')
            except Exception as e:
                print(f"   ⚠️ Could not create contour plot: {e}")
                # Fallback to simple scatter plot
                scatter = ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], 
                                     c=self.df['accuracy'], s=100, cmap='RdYlGn', alpha=0.8)
                ax3.set_xlabel('Malicious Percentage (%)')
                ax3.set_ylabel('Corrupted Percentage (%)')
                ax3.set_title('Accuracy Distribution (Scatter)')
                plt.colorbar(scatter, ax=ax3, label='Accuracy (%)')
        else:
            # Simple scatter plot for few data points
            scatter = ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], 
                                 c=self.df['accuracy'], s=100, cmap='RdYlGn', alpha=0.8)
            ax3.set_xlabel('Malicious Percentage (%)')
            ax3.set_ylabel('Corrupted Percentage (%)')
            ax3.set_title('Accuracy Distribution (Scatter)')
            plt.colorbar(scatter, ax=ax3, label='Accuracy (%)')
        
        # Plot 4: Accuracy vs Configuration
        config_labels = [f"{row['malicious_perc']}%M,{row['corrupted_perc']}%C" 
                        for _, row in self.df.iterrows()]
        ax4.bar(range(len(config_labels)), self.df['accuracy'], 
               color=plt.cm.viridis(self.df['total_perc']/self.df['total_perc'].max()))
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('System Accuracy (%)')
        ax4.set_title('Accuracy by Configuration')
        ax4.set_xticks(range(len(config_labels)))
        ax4.set_xticklabels(config_labels, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('System Accuracy Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Accuracy analysis plot saved")
    
    def plot_malicious_vs_corrupted_impact(self):
        """Compare impact of malicious vs corrupted percentages."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate correlations
        mal_fpr_corr = self.df['malicious_perc'].corr(self.df['fpr'])
        cor_fpr_corr = self.df['corrupted_perc'].corr(self.df['fpr'])
        mal_tpr_corr = self.df['malicious_perc'].corr(self.df['tpr_malicious'])
        cor_tpr_corr = self.df['corrupted_perc'].corr(self.df['tpr_malicious'])
        
        # Plot 1: Impact on FPR
        ax1.scatter(self.df['malicious_perc'], self.df['fpr'], 
                   alpha=0.6, label=f'Malicious (r={mal_fpr_corr:.3f})', s=80)
        ax1.scatter(self.df['corrupted_perc'], self.df['fpr'], 
                   alpha=0.6, label=f'Corrupted (r={cor_fpr_corr:.3f})', s=80)
        ax1.set_xlabel('Percentage (%)')
        ax1.set_ylabel('False Positive Rate (%)')
        ax1.set_title('Malicious vs Corrupted Impact on FPR')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Impact on TPR
        ax2.scatter(self.df['malicious_perc'], self.df['tpr_malicious'], 
                   alpha=0.6, label=f'Malicious (r={mal_tpr_corr:.3f})', s=80)
        ax2.scatter(self.df['corrupted_perc'], self.df['tpr_malicious'], 
                   alpha=0.6, label=f'Corrupted (r={cor_tpr_corr:.3f})', s=80)
        ax2.set_xlabel('Percentage (%)')
        ax2.set_ylabel('True Positive Rate (%)')
        ax2.set_title('Malicious vs Corrupted Impact on TPR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Combined Impact Analysis
        self.df['threat_balance'] = self.df['malicious_perc'] - self.df['corrupted_perc']
        scatter = ax3.scatter(self.df['threat_balance'], self.df['accuracy'], 
                             c=self.df['total_perc'], s=100, cmap='coolwarm', alpha=0.7)
        ax3.set_xlabel('Threat Balance (Malicious % - Corrupted %)')
        ax3.set_ylabel('System Accuracy (%)')
        ax3.set_title('Impact of Threat Type Balance on Accuracy')
        ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax3, label='Total Threat %')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Percentage Composition Analysis
        bottom_corrupted = np.zeros(len(self.df))
        ax4.bar(range(len(self.df)), self.df['malicious_perc'], 
               label='Malicious', color=self.colors['malicious'], alpha=0.8)
        ax4.bar(range(len(self.df)), self.df['corrupted_perc'], 
               bottom=self.df['malicious_perc'], label='Corrupted', 
               color=self.colors['corrupted'], alpha=0.8)
        
        # Add accuracy line
        ax4_twin = ax4.twinx()
        ax4_twin.plot(range(len(self.df)), self.df['accuracy'], 
                     'ko-', linewidth=2, markersize=6, label='Accuracy')
        
        ax4.set_xlabel('Configuration Index')
        ax4.set_ylabel('Threat Percentage (%)')
        ax4_twin.set_ylabel('Accuracy (%)')
        ax4.set_title('Threat Composition vs System Performance')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Malicious vs Corrupted Impact Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'malicious_vs_corrupted_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Malicious vs corrupted impact plot saved")
    
    def plot_total_percentage_trends(self):
        """Analyze trends across total percentage levels."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate average metrics by total percentage (only numeric columns)
        numeric_columns = ['fpr', 'tpr_malicious', 'fnr_malicious', 'accuracy']
        total_perc_stats = self.df.groupby('total_perc')[numeric_columns].agg(['mean', 'std']).round(2)
        
        total_percs = sorted(self.df['total_perc'].unique())
        
        # Plot 1: Average metrics with error bars
        fpr_means = total_perc_stats['fpr']['mean']
        fpr_stds = total_perc_stats['fpr']['std']
        tpr_means = total_perc_stats['tpr_malicious']['mean']
        tpr_stds = total_perc_stats['tpr_malicious']['std']
        
        ax1.errorbar(total_percs, fpr_means, yerr=fpr_stds, fmt='o-', 
                    label='FPR', linewidth=2, markersize=8, capsize=5)
        ax1.errorbar(total_percs, tpr_means, yerr=tpr_stds, fmt='o-', 
                    label='TPR', linewidth=2, markersize=8, capsize=5)
        ax1.set_xlabel('Total Threat Percentage (%)')
        ax1.set_ylabel('Rate (%)')
        ax1.set_title('Average FPR and TPR by Total Threat Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy trends
        acc_means = total_perc_stats['accuracy']['mean']
        acc_stds = total_perc_stats['accuracy']['std']
        
        ax2.errorbar(total_percs, acc_means, yerr=acc_stds, fmt='o-', 
                    color=self.colors['accuracy'], linewidth=2, markersize=8, capsize=5)
        ax2.fill_between(total_percs, acc_means - acc_stds, acc_means + acc_stds, 
                        alpha=0.3, color=self.colors['accuracy'])
        ax2.set_xlabel('Total Threat Percentage (%)')
        ax2.set_ylabel('System Accuracy (%)')
        ax2.set_title('Average Accuracy by Total Threat Level')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Variance analysis (only on numeric columns)
        numeric_columns = ['fpr', 'tpr_malicious', 'fnr_malicious', 'accuracy']
        
        # Check if we have enough data for variance calculation
        group_sizes = self.df.groupby('total_perc').size()
        if any(group_sizes > 1):  # At least one group has multiple values
            try:
                variances = self.df.groupby('total_perc')[numeric_columns].var()
                ax3.plot(total_percs, variances['fpr'], 'o-', label='FPR Variance', linewidth=2)
                ax3.plot(total_percs, variances['tpr_malicious'], 'o-', label='TPR Variance', linewidth=2)
                ax3.plot(total_percs, variances['accuracy'], 'o-', label='Accuracy Variance', linewidth=2)
                ax3.set_xlabel('Total Threat Percentage (%)')
                ax3.set_ylabel('Variance')
                ax3.set_title('Metric Variance by Total Threat Level')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            except Exception as e:
                print(f"   ⚠️ Could not calculate variance: {e}")
                # Fallback: show standard deviation instead
                std_devs = self.df.groupby('total_perc')[numeric_columns].std().fillna(0)
                ax3.plot(total_percs, std_devs['fpr'], 'o-', label='FPR Std Dev', linewidth=2)
                ax3.plot(total_percs, std_devs['tpr_malicious'], 'o-', label='TPR Std Dev', linewidth=2)
                ax3.plot(total_percs, std_devs['accuracy'], 'o-', label='Accuracy Std Dev', linewidth=2)
                ax3.set_xlabel('Total Threat Percentage (%)')
                ax3.set_ylabel('Standard Deviation')
                ax3.set_title('Metric Standard Deviation by Total Threat Level')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        else:
            # All groups have only one value, show simple range plot
            means = self.df.groupby('total_perc')[numeric_columns].mean()
            ax3.plot(total_percs, means['fpr'], 'o-', label='FPR', linewidth=2)
            ax3.plot(total_percs, means['tpr_malicious'], 'o-', label='TPR', linewidth=2)
            ax3.plot(total_percs, means['accuracy'], 'o-', label='Accuracy', linewidth=2)
            ax3.set_xlabel('Total Threat Percentage (%)')
            ax3.set_ylabel('Metric Value (%)')
            ax3.set_title('Metric Values by Total Threat Level')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Configuration count by total percentage
        config_counts = self.df['total_perc'].value_counts().sort_index()
        bars = ax4.bar(config_counts.index, config_counts.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(config_counts))))
        ax4.set_xlabel('Total Threat Percentage (%)')
        ax4.set_ylabel('Number of Configurations')
        ax4.set_title('Configuration Distribution by Total Threat Level')
        
        # Add value labels on bars
        for bar, count in zip(bars, config_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Total Threat Percentage Trends Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'total_percentage_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Total percentage trends plot saved")
    
    def plot_3d_surface_plots(self):
        """Create 3D surface plots for comprehensive visualization."""
        # Check if we have enough data points for 3D interpolation
        if len(self.df) < 4:
            print("   ⚠️ Not enough data points for 3D surface plots (need at least 4)")
            # Create alternative 3D scatter plots instead
            self._create_3d_scatter_plots()
            return
            
        fig = plt.figure(figsize=(20, 15))
        
        # Prepare data for 3D plotting
        try:
            from scipy.interpolate import griddata
            
            # Create fine grid for smooth surfaces
            mal_range = np.linspace(self.df['malicious_perc'].min(), self.df['malicious_perc'].max(), 30)
            cor_range = np.linspace(self.df['corrupted_perc'].min(), self.df['corrupted_perc'].max(), 30)
            MAL, COR = np.meshgrid(mal_range, cor_range)
            
            points = self.df[['malicious_perc', 'corrupted_perc']].values
            
            # Plot 1: FPR Surface
            ax1 = fig.add_subplot(221, projection='3d')
            fpr_grid = griddata(points, self.df['fpr'].values, (MAL, COR), method='cubic')
            surf1 = ax1.plot_surface(MAL, COR, fpr_grid, cmap='Reds', alpha=0.8)
            ax1.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], self.df['fpr'], 
                       c='black', s=50)
            ax1.set_xlabel('Malicious %')
            ax1.set_ylabel('Corrupted %')
            ax1.set_zlabel('FPR %')
            ax1.set_title('FPR 3D Surface')
            
            # Plot 2: TPR Surface
            ax2 = fig.add_subplot(222, projection='3d')
            tpr_grid = griddata(points, self.df['tpr_malicious'].values, (MAL, COR), method='cubic')
            surf2 = ax2.plot_surface(MAL, COR, tpr_grid, cmap='Greens', alpha=0.8)
            ax2.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], self.df['tpr_malicious'], 
                       c='black', s=50)
            ax2.set_xlabel('Malicious %')
            ax2.set_ylabel('Corrupted %')
            ax2.set_zlabel('TPR %')
            ax2.set_title('TPR 3D Surface')
            
            # Plot 3: Accuracy Surface
            ax3 = fig.add_subplot(223, projection='3d')
            acc_grid = griddata(points, self.df['accuracy'].values, (MAL, COR), method='cubic')
            surf3 = ax3.plot_surface(MAL, COR, acc_grid, cmap='Blues', alpha=0.8)
            ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], self.df['accuracy'], 
                       c='black', s=50)
            ax3.set_xlabel('Malicious %')
            ax3.set_ylabel('Corrupted %')
            ax3.set_zlabel('Accuracy %')
            ax3.set_title('Accuracy 3D Surface')
            
            # Plot 4: Combined Performance Surface (composite score)
            ax4 = fig.add_subplot(224, projection='3d')
            composite_score = self.df['tpr_malicious'] + self.df['accuracy'] - self.df['fpr'] - self.df['fnr_malicious']
            comp_grid = griddata(points, composite_score.values, (MAL, COR), method='cubic')
            surf4 = ax4.plot_surface(MAL, COR, comp_grid, cmap='viridis', alpha=0.8)
            ax4.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], composite_score, 
                       c='black', s=50)
            ax4.set_xlabel('Malicious %')
            ax4.set_ylabel('Corrupted %')
            ax4.set_zlabel('Composite Score')
            ax4.set_title('Performance Composite Score 3D Surface')
            
            plt.suptitle('3D Performance Surfaces', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / '3d_surface_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ 3D surface plots saved")
            
        except Exception as e:
            print(f"   ⚠️ Error creating 3D surfaces: {e}")
            # Fallback to scatter plots
            self._create_3d_scatter_plots()
    
    def _create_3d_scatter_plots(self):
        """Create 3D scatter plots as fallback when surface interpolation fails."""
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: FPR 3D Scatter
        ax1 = fig.add_subplot(221, projection='3d')
        scatter1 = ax1.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], self.df['fpr'], 
                              c=self.df['fpr'], s=100, cmap='Reds', alpha=0.8)
        ax1.set_xlabel('Malicious %')
        ax1.set_ylabel('Corrupted %')
        ax1.set_zlabel('FPR %')
        ax1.set_title('FPR 3D Scatter')
        
        # Plot 2: TPR 3D Scatter
        ax2 = fig.add_subplot(222, projection='3d')
        scatter2 = ax2.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], self.df['tpr_malicious'], 
                              c=self.df['tpr_malicious'], s=100, cmap='Greens', alpha=0.8)
        ax2.set_xlabel('Malicious %')
        ax2.set_ylabel('Corrupted %')
        ax2.set_zlabel('TPR %')
        ax2.set_title('TPR 3D Scatter')
        
        # Plot 3: Accuracy 3D Scatter
        ax3 = fig.add_subplot(223, projection='3d')
        scatter3 = ax3.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], self.df['accuracy'], 
                              c=self.df['accuracy'], s=100, cmap='Blues', alpha=0.8)
        ax3.set_xlabel('Malicious %')
        ax3.set_ylabel('Corrupted %')
        ax3.set_zlabel('Accuracy %')
        ax3.set_title('Accuracy 3D Scatter')
        
        # Plot 4: Composite Score 3D Scatter
        ax4 = fig.add_subplot(224, projection='3d')
        composite_score = self.df['tpr_malicious'] + self.df['accuracy'] - self.df['fpr'] - self.df['fnr_malicious']
        scatter4 = ax4.scatter(self.df['malicious_perc'], self.df['corrupted_perc'], composite_score, 
                              c=composite_score, s=100, cmap='viridis', alpha=0.8)
        ax4.set_xlabel('Malicious %')
        ax4.set_ylabel('Corrupted %')
        ax4.set_zlabel('Composite Score')
        ax4.set_title('Performance Composite Score 3D Scatter')
        
        plt.suptitle('3D Performance Analysis (Scatter Mode)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '3d_scatter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ 3D scatter plots saved")
    
    def plot_correlation_matrix(self):
        """Create correlation analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Select relevant columns for correlation
        corr_columns = ['malicious_perc', 'corrupted_perc', 'total_perc', 
                       'fpr', 'tpr_malicious', 'fnr_malicious', 'accuracy']
        corr_data = self.df[corr_columns]
        
        # Plot 1: Full correlation matrix
        corr_matrix = corr_data.corr()
        im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(corr_columns)))
        ax1.set_xticklabels(corr_columns, rotation=45)
        ax1.set_yticks(range(len(corr_columns)))
        ax1.set_yticklabels(corr_columns)
        ax1.set_title('Full Correlation Matrix')
        
        # Add correlation values to heatmap
        for i in range(len(corr_columns)):
            for j in range(len(corr_columns)):
                ax1.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}', 
                        ha='center', va='center', fontsize=10)
        
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Percentage parameters vs performance metrics
        perf_metrics = ['fpr', 'tpr_malicious', 'fnr_malicious', 'accuracy']
        perc_params = ['malicious_perc', 'corrupted_perc', 'total_perc']
        
        perf_corr = corr_data[perc_params + perf_metrics].corr().loc[perc_params, perf_metrics]
        im2 = ax2.imshow(perf_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(perf_metrics)))
        ax2.set_xticklabels(perf_metrics, rotation=45)
        ax2.set_yticks(range(len(perc_params)))
        ax2.set_yticklabels(perc_params)
        ax2.set_title('Percentage Parameters vs Performance Metrics')
        
        for i in range(len(perc_params)):
            for j in range(len(perf_metrics)):
                ax2.text(j, i, f'{perf_corr.iloc[i,j]:.2f}', 
                        ha='center', va='center', fontsize=10)
        
        plt.colorbar(im2, ax=ax2)
        
        # Plot 3: Scatter plot matrix for key relationships
        ax3.scatter(self.df['total_perc'], self.df['accuracy'], alpha=0.6, s=80)
        ax3.set_xlabel('Total Threat Percentage (%)')
        ax3.set_ylabel('System Accuracy (%)')
        ax3.set_title(f'Total Threat vs Accuracy (r={self.df["total_perc"].corr(self.df["accuracy"]):.3f})')
        ax3.grid(True, alpha=0.3)
        
        # Add trendline
        z = np.polyfit(self.df['total_perc'], self.df['accuracy'], 1)
        p = np.poly1d(z)
        ax3.plot(self.df['total_perc'], p(self.df['total_perc']), "r--", alpha=0.8)
        
        # Plot 4: Performance trade-offs
        ax4.scatter(self.df['fpr'], self.df['tpr_malicious'], 
                   c=self.df['total_perc'], s=100, cmap='viridis', alpha=0.7)
        ax4.set_xlabel('False Positive Rate (%)')
        ax4.set_ylabel('True Positive Rate (%)')
        ax4.set_title('FPR vs TPR Trade-off')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(ax4.collections[0], ax=ax4, label='Total Threat %')
        
        plt.suptitle('Correlation and Relationship Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Correlation analysis plot saved")
    
    def plot_configuration_comparison(self):
        """Create comprehensive configuration comparison plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Calculate composite performance score
        self.df['composite_score'] = (self.df['tpr_malicious'] + self.df['accuracy'] - 
                                     self.df['fpr'] - self.df['fnr_malicious'])
        
        # Sort by composite score for ranking
        df_sorted = self.df.sort_values('composite_score', ascending=False)
        
        # Plot 1: Configuration ranking
        config_labels = [f"{row['malicious_perc']}%M,{row['corrupted_perc']}%C" 
                        for _, row in df_sorted.iterrows()]
        
        bars = ax1.barh(range(len(config_labels)), df_sorted['composite_score'], 
                       color=plt.cm.RdYlGn((df_sorted['composite_score'] - df_sorted['composite_score'].min()) / 
                                          (df_sorted['composite_score'].max() - df_sorted['composite_score'].min())))
        ax1.set_yticks(range(len(config_labels)))
        ax1.set_yticklabels(config_labels)
        ax1.set_xlabel('Composite Performance Score')
        ax1.set_title('Configuration Ranking by Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, df_sorted['composite_score'])):
            ax1.text(score + 0.5, i, f'{score:.1f}', va='center', fontsize=9)
        
        # Plot 2: Multi-metric radar chart for top configurations
        top_configs = df_sorted.head(min(5, len(df_sorted)))  # Take at most 5 or all available
        
        if len(top_configs) > 0:
            # Normalize metrics for radar chart (0-100 scale)
            metrics = ['tpr_malicious', 'accuracy']  # Higher is better
            inverse_metrics = ['fpr', 'fnr_malicious']  # Lower is better (invert for visualization)
            
            angles = np.linspace(0, 2*np.pi, len(metrics + inverse_metrics), endpoint=False)
            
            for i, (_, config) in enumerate(top_configs.iterrows()):
                values = []
                values.extend([config[m] for m in metrics])
                values.extend([100 - config[m] for m in inverse_metrics])  # Invert for "lower is better"
                values += values[:1]  # Complete the circle
                
                angles_plot = np.concatenate([angles, [angles[0]]])
                ax2.plot(angles_plot, values, 'o-', linewidth=2, 
                        label=f"{config['malicious_perc']}%M,{config['corrupted_perc']}%C")
                ax2.fill(angles_plot, values, alpha=0.1)
            
            ax2.set_xticks(angles)
            ax2.set_xticklabels(['TPR', 'Accuracy', 'FPR (inv)', 'FNR (inv)'])
            ax2.set_ylim(0, 100)
            ax2.set_title(f'Top {len(top_configs)} Configurations - Multi-Metric Comparison')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'No configurations available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Multi-Metric Comparison (No Data)')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # Plot 3: Performance vs threat level scatter
        colors = plt.cm.viridis(self.df['total_perc'] / self.df['total_perc'].max())
        scatter = ax3.scatter(self.df['total_perc'], self.df['composite_score'], 
                             c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add configuration labels for extreme points
        best_config = df_sorted.iloc[0]
        worst_config = df_sorted.iloc[-1]
        
        ax3.annotate(f"Best: {best_config['malicious_perc']}%M,{best_config['corrupted_perc']}%C", 
                    xy=(best_config['total_perc'], best_config['composite_score']),
                    xytext=(10, 10), textcoords='offset points', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
        
        ax3.annotate(f"Worst: {worst_config['malicious_perc']}%M,{worst_config['corrupted_perc']}%C", 
                    xy=(worst_config['total_perc'], worst_config['composite_score']),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        ax3.set_xlabel('Total Threat Percentage (%)')
        ax3.set_ylabel('Composite Performance Score')
        ax3.set_title('Performance vs Threat Level')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Metric distribution comparison
        metrics_to_plot = ['fpr', 'tpr_malicious', 'fnr_malicious', 'accuracy']
        box_data = [self.df[metric].values for metric in metrics_to_plot]
        
        bp = ax4.boxplot(box_data, labels=['FPR', 'TPR', 'FNR', 'Accuracy'], patch_artist=True)
        
        # Color the boxes
        colors = ['red', 'green', 'orange', 'blue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Metric Distribution Across All Configurations')
        ax4.grid(True, alpha=0.3)
        
        # Add mean lines
        for i, metric in enumerate(metrics_to_plot):
            mean_val = self.df[metric].mean()
            ax4.hlines(mean_val, i+0.8, i+1.2, colors='black', linestyles='dashed')
            ax4.text(i+1, mean_val, f'{mean_val:.1f}', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive Configuration Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'configuration_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Configuration comparison plot saved")

def main():
    """Main function to run the percentages metrics plotting."""
    print("🎨 Starting Percentages Metrics Plotting")
    print("=" * 50)
    
    try:
        # Check if comparison data exists
        csv_path = "csv_data/percentages_comparison_data.csv"
        if not Path(csv_path).exists():
            print(f"❌ Comparison data not found: {csv_path}")
            print("Please run compare_percentages_results.py first to generate the data.")
            return 1
        
        # Initialize plotter and create all plots
        plotter = PercentagesMetricsPlotter(csv_path)
        plotter.create_all_plots()
        
        print("\n" + "=" * 50)
        print("✅ PERCENTAGES METRICS PLOTTING COMPLETED!")
        print(f"📁 All plots saved to: {plotter.output_dir}")
        print("📊 Generated plots:")
        print("   • FPR detailed analysis")
        print("   • TPR detailed analysis") 
        print("   • FNR detailed analysis")
        print("   • Accuracy detailed analysis")
        print("   • Malicious vs corrupted impact analysis")
        print("   • Total percentage trends analysis")
        print("   • 3D surface analysis")
        print("   • Correlation analysis")
        print("   • Configuration comparison analysis")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
