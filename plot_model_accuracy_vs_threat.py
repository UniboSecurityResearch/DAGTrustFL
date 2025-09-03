#!/usr/bin/env python3
"""
Model Accuracy vs Threat Level Plotter for DAGTrustFL Trust Experiments
======================================================================

Questo script crea un grafico specifico che mostra la relazione tra
il livello di minacce (threat percentage) e l'accuracy finale del modello
FL negli esperimenti trust.

Asse X: Threat Percentage (malicious + corrupted)
Asse Y: Final Model Accuracy

Autore: DAGTrustFL Analysis Suite
Data: 2025-09-03
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re
from pathlib import Path
from scipy import stats

# Set matplotlib font configuration like other plots
plt.rcParams.update({
    'font.family': ['Roboto', 'DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'text.usetex': False
})

# Set seaborn style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class ModelAccuracyThreatPlotter:
    """Classe per creare il grafico accuracy vs threat level."""
    
    def __init__(self, results_base_path="FL/results_percentages_trust"):
        """
        Inizializza il plotter.
        
        Args:
            results_base_path (str): Percorso base dove sono i risultati trust
        """
        self.results_base_path = Path(results_base_path)
        self.output_dir = "percentages_trust_comparison_plots/fl_comparison_plots"
        self.experiments = {}
        
        # Crea directory per i grafici
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'baseline': '#1f77b4',
            'threat_line': '#d62728', 
            'threat_points': '#ff7f0e',
            'trend_line': '#2ca02c',
            'mean_line': '#228B22',  # Verde per la nostra retta
            'reference_line': '#1E90FF',  # Blu per la loro retta
            'grid': '#E0E0E0'
        }
        
    def scan_experiment_folders(self):
        """Scansiona le cartelle degli esperimenti trust."""
        print("🔍 Scanning for FL trust experiment folders...")
        
        if not self.results_base_path.exists():
            print(f"❌ Results path not found: {self.results_base_path}")
            return False
        
        # Pattern: mal_X_cor_Y_trust
        pattern = r'mal_(\d+)_cor_(\d+)_trust'
        
        for folder in self.results_base_path.iterdir():
            if folder.is_dir():
                match = re.match(pattern, folder.name)
                if match:
                    malicious_perc = int(match.group(1))
                    corrupted_perc = int(match.group(2))
                    
                    # Look for the FL results CSV file
                    fl_csv_file = folder / "3000_results.csv"
                    if fl_csv_file.exists():
                        self.experiments[(malicious_perc, corrupted_perc)] = {
                            'folder': folder,
                            'fl_csv_path': fl_csv_file,
                            'malicious_perc': malicious_perc,
                            'corrupted_perc': corrupted_perc,
                            'total_perc': malicious_perc + corrupted_perc,
                            'is_baseline': (malicious_perc == 0 and corrupted_perc == 0)
                        }
                        print(f"   ✓ Found: {malicious_perc}%M, {corrupted_perc}%C (Total: {malicious_perc + corrupted_perc}%)")
                    else:
                        print(f"   ⚠️ No FL CSV file in {folder.name}")
        
        if not self.experiments:
            print("❌ No valid FL experiment folders found!")
            return False
        
        print(f"\n📊 Found {len(self.experiments)} FL trust experiments")
        return True
    
    def load_final_accuracies(self):
        """Carica le accuracy finali da tutti gli esperimenti."""
        data_points = []
        
        print("📊 Loading final model accuracies...")
        
        for (mal_perc, cor_perc), exp_info in sorted(self.experiments.items()):
            try:
                # Carica il CSV FL
                raw_data = pd.read_csv(exp_info['fl_csv_path'], header=None)
                
                # La prima colonna è l'accuracy
                final_accuracy = raw_data.iloc[-1, 0]  # Ultimo valore della prima colonna
                total_threat = exp_info['total_perc']
                is_baseline = exp_info['is_baseline']
                
                data_point = {
                    'malicious_perc': mal_perc,
                    'corrupted_perc': cor_perc,
                    'total_threat_perc': total_threat,
                    'final_accuracy': final_accuracy,
                    'is_baseline': is_baseline,
                    'config_label': f"{mal_perc}%M,{cor_perc}%C" + (" (Baseline)" if is_baseline else "")
                }
                
                data_points.append(data_point)
                
                print(f"   ✓ {data_point['config_label']}: {final_accuracy:.2f}%")
                
            except Exception as e:
                print(f"   ❌ Error loading {exp_info['fl_csv_path']}: {e}")
        
        if not data_points:
            print("❌ No valid accuracy data found!")
            return None
        
        df = pd.DataFrame(data_points)
        
        # Sort by total threat percentage
        df = df.sort_values('total_threat_perc')
        
        print(f"\n📈 Loaded {len(df)} accuracy data points")
        print(f"   Threat range: {df['total_threat_perc'].min()}% - {df['total_threat_perc'].max()}%")
        print(f"   Accuracy range: {df['final_accuracy'].min():.2f}% - {df['final_accuracy'].max():.2f}%")
        
        return df
    
    def create_accuracy_threat_plot(self, df):
        """Crea il grafico principale accuracy vs threat level."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Reference baseline values (https://arxiv.org/pdf/2007.08432)
        reference_data = {
            'threat_perc': [0, 10, 20, 30, 40],
            'accuracy': [91.7, 89.7, 88.3, 88.2, 87.8]
        }
        
        # Separa baseline e threat scenarios
        baseline_df = df[df['is_baseline'] == True]
        threat_df = df[df['is_baseline'] == False]
        
        # Plot baseline as special point
        if len(baseline_df) > 0:
            baseline_acc = baseline_df['final_accuracy'].iloc[0]
            ax.scatter(0, baseline_acc, s=400, color=self.colors['baseline'], 
                      marker='*', label='Baseline (0% threats)', alpha=1.0, 
                      edgecolors='black', linewidth=3, zorder=5)
            
            # Add baseline reference line
            ax.axhline(baseline_acc, color=self.colors['baseline'], linestyle='--', 
                      linewidth=3, alpha=0.7, label=f'Baseline Level: {baseline_acc:.1f}%')
        
        # Plot reference system performance (sistema precedente)
        ax.plot(reference_data['threat_perc'], reference_data['accuracy'], 
               color=self.colors['reference_line'], linewidth=4, alpha=0.9, linestyle='-', 
               marker='s', markersize=10, markerfacecolor='white', 
               markeredgecolor=self.colors['reference_line'], markeredgewidth=3,
               label='Reference System', zorder=7)
        
        # Add reference points as scatter for visibility
        ax.scatter(reference_data['threat_perc'], reference_data['accuracy'], 
                  s=150, color=self.colors['reference_line'], alpha=0.8, marker='s',
                  edgecolors='black', linewidth=2, zorder=8)
        
        # Plot threat scenarios
        if len(threat_df) > 0:
            # Main scatter plot
            scatter = ax.scatter(threat_df['total_threat_perc'], threat_df['final_accuracy'], 
                               s=200, color=self.colors['threat_points'], alpha=0.8, 
                               edgecolors='black', linewidth=2, label='Threat scenarios', zorder=4)
            
            # Calculate mean accuracy for each unique threat percentage
            threat_means = threat_df.groupby('total_threat_perc')['final_accuracy'].agg(['mean', 'std', 'count']).reset_index()
            threat_means = threat_means.sort_values('total_threat_perc')
            
            # Add baseline point to means if it exists
            all_means = threat_means.copy()
            if len(baseline_df) > 0:
                baseline_mean = pd.DataFrame({
                    'total_threat_perc': [0],
                    'mean': [baseline_df['final_accuracy'].iloc[0]],
                    'std': [0],
                    'count': [1]
                })
                all_means = pd.concat([baseline_mean, threat_means], ignore_index=True)
                all_means = all_means.sort_values('total_threat_perc')
            
            # Plot mean accuracy line connecting the points
            if len(all_means) >= 2:
                ax.plot(all_means['total_threat_perc'], all_means['mean'], 
                       color=self.colors['mean_line'], linewidth=4, alpha=0.9, linestyle='-', 
                       marker='o', markersize=8, markerfacecolor='white', 
                       markeredgecolor=self.colors['mean_line'], markeredgewidth=3,
                       label='DAGTrustFL (Our System)', zorder=6)
                
                # Add error bars for standard deviation (only where std > 0)
                valid_std = all_means[all_means['std'] > 0]
                if len(valid_std) > 0:
                    ax.errorbar(valid_std['total_threat_perc'], valid_std['mean'], 
                               yerr=valid_std['std'], fmt='none', 
                               color=self.colors['mean_line'], alpha=0.6, capsize=5, capthick=2,
                               zorder=5)
            
            # # Annotate each point with its configuration - smart positioning to avoid overlaps
            # for i, (_, row) in enumerate(threat_df.iterrows()):
            #     # Calculate smart offset based on position and index to avoid overlaps
            #     base_offset_x = 8
            #     base_offset_y = 8
            #     
            #     # Vary the position based on index and threat level to minimize overlaps
            #     if i % 2 == 0:
            #         offset_x = base_offset_x
            #         offset_y = base_offset_y + (i // 2) * 6
            #     else:
            #         offset_x = base_offset_x + 15
            #         offset_y = base_offset_y - (i // 2) * 6
            #     
            #     # Adjust based on position in graph to avoid edges
            #     if row['total_threat_perc'] > 35:  # Right side of graph
            #         offset_x = -offset_x - 30  # Move labels to the left
            #     if row['final_accuracy'] > 90:  # Top of graph
            #         offset_y = -abs(offset_y) - 5  # Move labels down
            #     
            #     ax.annotate(f"{row['malicious_perc']}%M,{row['corrupted_perc']}%C", 
            #                (row['total_threat_perc'], row['final_accuracy']),
            #                xytext=(offset_x, offset_y), textcoords='offset points', 
            #                fontsize=14, alpha=0.9, fontweight='bold',
            #                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
            #                        alpha=0.8, edgecolor='gray', linewidth=0.5),
            #                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
            #                              color='gray', alpha=0.7, linewidth=1))
        
        # Customize the plot
        ax.set_xlabel('Total Threat (M+C)%', fontsize=20, fontweight='bold')
        ax.set_ylabel('Model Accuracy (%)', fontsize=20, fontweight='bold')
        
        # Set axis limits as specified
        x_min = -2
        x_max = df['total_threat_perc'].max() + 2
        y_min = 86  # Specificato dall'utente
        y_max = 92  # Specificato dall'utente
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Customize ticks with larger font
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend with larger font
        ax.legend(fontsize=16, loc='lower left', frameon=True, fancybox=True, shadow=True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{self.output_dir}/model_accuracy_vs_threat_level.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Accuracy vs Threat plot saved: {plot_filename}")
        
        # Also save as PDF
        pdf_filename = f"{self.output_dir}/model_accuracy_vs_threat_level.pdf"
        plt.savefig(pdf_filename, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.show()
        plt.close()
        
        return plot_filename
    
    def create_detailed_breakdown_plot(self, df):
        """Crea un grafico con breakdown malicious vs corrupted."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy vs Total Threat (main plot replica)
        baseline_df = df[df['is_baseline'] == True]
        threat_df = df[df['is_baseline'] == False]
        
        if len(baseline_df) > 0:
            baseline_acc = baseline_df['final_accuracy'].iloc[0]
            ax1.scatter(0, baseline_acc, s=300, color=self.colors['baseline'], 
                       marker='*', label='Baseline', alpha=1.0, edgecolors='black', linewidth=2)
        
        if len(threat_df) > 0:
            ax1.scatter(threat_df['total_threat_perc'], threat_df['final_accuracy'], 
                       s=150, color=self.colors['threat_points'], alpha=0.8, 
                       edgecolors='black', linewidth=1, label='Threats')
        
        ax1.set_xlabel('Total Threat %')
        ax1.set_ylabel('Final Accuracy (%)')
        ax1.set_title('Accuracy vs Total Threat Level')
        ax1.legend(fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Malicious %
        malicious_data = df[df['malicious_perc'] > 0] if len(df[df['malicious_perc'] > 0]) > 0 else pd.DataFrame()
        
        if len(malicious_data) > 0:
            ax2.scatter(malicious_data['malicious_perc'], malicious_data['final_accuracy'], 
                       s=150, color='red', alpha=0.8, edgecolors='black', linewidth=1)
            
            # Fit trend line
            if len(malicious_data) >= 2:
                x_mal = malicious_data['malicious_perc'].values
                y_acc = malicious_data['final_accuracy'].values
                z = np.polyfit(x_mal, y_acc, 1)
                p = np.poly1d(z)
                ax2.plot(x_mal, p(x_mal), "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Malicious %')
        ax2.set_ylabel('Final Accuracy (%)')
        ax2.set_title('Accuracy vs Malicious Percentage')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy vs Corrupted %
        corrupted_data = df[df['corrupted_perc'] > 0] if len(df[df['corrupted_perc'] > 0]) > 0 else pd.DataFrame()
        
        if len(corrupted_data) > 0:
            ax3.scatter(corrupted_data['corrupted_perc'], corrupted_data['final_accuracy'], 
                       s=150, color='orange', alpha=0.8, edgecolors='black', linewidth=1)
            
            # Fit trend line
            if len(corrupted_data) >= 2:
                x_cor = corrupted_data['corrupted_perc'].values
                y_acc = corrupted_data['final_accuracy'].values
                z = np.polyfit(x_cor, y_acc, 1)
                p = np.poly1d(z)
                ax3.plot(x_cor, p(x_cor), "orange", linestyle='--', alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Corrupted %')
        ax3.set_ylabel('Final Accuracy (%)')
        ax3.set_title('Accuracy vs Corrupted Percentage')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary table with all values
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create table data
        table_data = []
        for _, row in df.iterrows():
            table_data.append([
                f"{int(row['malicious_perc'])}%",
                f"{int(row['corrupted_perc'])}%",
                f"{int(row['total_threat_perc'])}%",
                f"{row['final_accuracy']:.2f}%"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Malicious %', 'Corrupted %', 'Total %', 'Final Accuracy'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Accuracy Summary Table', fontsize=16, fontweight='bold', pad=20)
        
        plt.suptitle('Model Accuracy Analysis - Detailed Breakdown', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{self.output_dir}/model_accuracy_detailed_breakdown.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Detailed breakdown plot saved: {plot_filename}")
        
        # Also save as PDF
        pdf_filename = f"{self.output_dir}/model_accuracy_detailed_breakdown.pdf"
        plt.savefig(pdf_filename, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.show()
        plt.close()
        
        return plot_filename
    
    def save_data_to_csv(self, df):
        """Salva i dati in CSV per analisi future."""
        csv_filename = f"{self.output_dir}/model_accuracy_vs_threat_data.csv"
        
        # Riordina le colonne per chiarezza
        df_export = df[['malicious_perc', 'corrupted_perc', 'total_threat_perc', 
                       'final_accuracy', 'is_baseline', 'config_label']].copy()
        
        df_export.to_csv(csv_filename, index=False)
        print(f"💾 Data saved to CSV: {csv_filename}")
        
        return csv_filename
    
    def run_analysis(self):
        """Esegui l'analisi completa."""
        print("🚀 Starting Model Accuracy vs Threat Level Analysis")
        print("="*60)
        
        # Scansiona le cartelle
        if not self.scan_experiment_folders():
            return False
        
        # Carica le accuracy finali
        df = self.load_final_accuracies()
        if df is None:
            return False
        
        # Crea i grafici
        main_plot = self.create_accuracy_threat_plot(df)
        detailed_plot = self.create_detailed_breakdown_plot(df)
        
        # Salva i dati
        csv_file = self.save_data_to_csv(df)
        
        print("\n" + "="*60)
        print("✅ MODEL ACCURACY vs THREAT ANALYSIS COMPLETED!")
        print(f"📊 Main plot: {main_plot}")
        print(f"📊 Detailed plot: {detailed_plot}")
        print(f"💾 Data CSV: {csv_file}")
        print(f"📁 Output directory: {self.output_dir}")
        print("="*60)
        
        return True


def main():
    """Funzione principale."""
    plotter = ModelAccuracyThreatPlotter()
    success = plotter.run_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")
    
    return success


if __name__ == "__main__":
    main()
