#!/usr/bin/env python3
"""
Trust Analysis Plotter for DAGTrustFL Trust Experiments
=====================================================

Questo script analizza e plotta i valori medi di trust dei peer buoni
vs peer cattivi (malicious/corrupted) al variare del threat percentage.

Legge i dati dai CSV dag_trust_analysis.             ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18, loc='upper right')ax.set_xlabel('Total Threat Percentage (%)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Mean Trust Value', fontsize=20, fontweight='bold') nelle cartelle dei risultati
e crea grafici che mostrano come evolve il trust durante gli esperimenti.

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

# Set matplotlib font configuration
plt.rcParams.update({
    'font.family': ['Roboto', 'DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.titlesize': 22,
    'text.usetex': False
})

# Set seaborn style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class TrustAnalysisPlotter:
    """Classe per analizzare e plottare i valori di trust dei peer."""
    
    def __init__(self, results_base_path="../FL/results_percentages_trust"):
        """
        Inizializza il plotter.
        
        Args:
            results_base_path (str): Percorso base dove sono i risultati trust
        """
        self.results_base_path = Path(results_base_path)
        self.output_dir = "percentages_trust_comparison_plots/trust_analysis_plots"
        self.experiments = {}
        
        # Crea directory per i grafici
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'good_peers': '#2E8B57',      # Verde scuro per peer buoni
            'bad_peers': '#DC143C',       # Rosso scuro per peer cattivi
            'malicious_peers': '#B22222', # Rosso fuoco per malicious
            'corrupted_peers': '#FF6347', # Rosso pomodoro per corrupted
            'baseline': '#4169E1',        # Blu reale per baseline
            'trend_good': '#90EE90',      # Verde chiaro per trend buoni
            'trend_bad': '#FFB6C1',       # Rosa per trend cattivi
            'grid': '#E0E0E0'
        }
        
    def scan_experiment_folders(self):
        """Scansiona le cartelle degli esperimenti trust."""
        print("🔍 Scanning for trust experiment folders...")
        
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
                    
                    # Look for the trust analysis CSV file
                    trust_csv_file = folder / "dag_trust_analysis.csv"
                    if trust_csv_file.exists():
                        self.experiments[(malicious_perc, corrupted_perc)] = {
                            'folder': folder,
                            'trust_csv_path': trust_csv_file,
                            'malicious_perc': malicious_perc,
                            'corrupted_perc': corrupted_perc,
                            'total_perc': malicious_perc + corrupted_perc,
                            'is_baseline': (malicious_perc == 0 and corrupted_perc == 0)
                        }
                        print(f"   ✓ Found: {malicious_perc}%M, {corrupted_perc}%C (Total: {malicious_perc + corrupted_perc}%)")
                    else:
                        print(f"   ⚠️ No trust CSV file in {folder.name}")
        
        if not self.experiments:
            print("❌ No valid trust experiment folders found!")
            return False
        
        print(f"\n📊 Found {len(self.experiments)} trust experiments")
        return True
    
    def load_trust_data(self):
        """Carica i dati di trust da tutti gli esperimenti."""
        all_data = []
        
        print("📊 Loading trust data from all experiments...")
        
        for (mal_perc, cor_perc), exp_info in sorted(self.experiments.items()):
            try:
                # Carica il CSV trust
                df = pd.read_csv(exp_info['trust_csv_path'])
                
                # Estrai i dati di trust per ogni peer
                peer_columns = [col for col in df.columns if col.startswith('Peer_') and col.endswith('_Trust')]
                malicious_columns = [col for col in df.columns if col.startswith('Peer_') and col.endswith('_Is_Malicious')]
                corrupted_columns = [col for col in df.columns if col.startswith('Peer_') and col.endswith('_Is_Corrupted')]
                
                # Per ogni round, calcola trust medio di peer buoni vs cattivi
                for _, row in df.iterrows():
                    round_num = row['Round']
                    
                    # Identifica peer buoni e cattivi
                    good_trust_values = []
                    malicious_trust_values = []
                    corrupted_trust_values = []
                    
                    for i in range(10):  # Assumiamo 10 peer (Peer_0 to Peer_9)
                        trust_col = f'Peer_{i}_Trust'
                        malicious_col = f'Peer_{i}_Is_Malicious'
                        corrupted_col = f'Peer_{i}_Is_Corrupted'
                        
                        if trust_col in df.columns and malicious_col in df.columns and corrupted_col in df.columns:
                            trust_val = row[trust_col]
                            is_malicious = bool(row[malicious_col])
                            is_corrupted = bool(row[corrupted_col])
                            
                            if is_malicious:
                                malicious_trust_values.append(trust_val)
                            elif is_corrupted:
                                corrupted_trust_values.append(trust_val)
                            else:
                                good_trust_values.append(trust_val)
                    
                    # Calcola medie
                    mean_good_trust = np.mean(good_trust_values) if good_trust_values else np.nan
                    mean_malicious_trust = np.mean(malicious_trust_values) if malicious_trust_values else np.nan
                    mean_corrupted_trust = np.mean(corrupted_trust_values) if corrupted_trust_values else np.nan
                    mean_bad_trust = np.mean(malicious_trust_values + corrupted_trust_values) if (malicious_trust_values + corrupted_trust_values) else np.nan
                    
                    data_point = {
                        'malicious_perc': mal_perc,
                        'corrupted_perc': cor_perc,
                        'total_threat_perc': exp_info['total_perc'],
                        'is_baseline': exp_info['is_baseline'],
                        'round': round_num,
                        'mean_good_trust': mean_good_trust,
                        'mean_malicious_trust': mean_malicious_trust,
                        'mean_corrupted_trust': mean_corrupted_trust,
                        'mean_bad_trust': mean_bad_trust,
                        'num_good_peers': len(good_trust_values),
                        'num_malicious_peers': len(malicious_trust_values),
                        'num_corrupted_peers': len(corrupted_trust_values),
                        'config_label': f"{mal_perc}%M,{cor_perc}%C"
                    }
                    
                    all_data.append(data_point)
                
                print(f"   ✓ Loaded {len(df)} rounds from {mal_perc}%M,{cor_perc}%C")
                
            except Exception as e:
                print(f"   ❌ Error loading {exp_info['trust_csv_path']}: {e}")
        
        if not all_data:
            print("❌ No valid trust data found!")
            return None
        
        trust_df = pd.DataFrame(all_data)
        
        print(f"\n📈 Loaded {len(trust_df)} trust data points")
        print(f"   Configurations: {trust_df['config_label'].nunique()}")
        print(f"   Rounds range: {trust_df['round'].min()} - {trust_df['round'].max()}")
        
        return trust_df
    
    def create_trust_evolution_plot(self, trust_df):
        """Crea grafico dell'evoluzione del trust nel tempo per ogni configurazione."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Raggruppa per threat level
        threat_levels = sorted(trust_df['total_threat_perc'].unique())
        
        for idx, threat_level in enumerate(threat_levels[:4]):  # Max 4 subplot
            ax = axes[idx]
            
            # Filtra dati per questo threat level
            threat_data = trust_df[trust_df['total_threat_perc'] == threat_level]
            
            if len(threat_data) == 0:
                ax.set_title(f'No data for {threat_level}% threat')
                continue
            
            # Plot good peers trust
            good_data = threat_data.groupby('round')['mean_good_trust'].mean()
            ax.plot(good_data.index, good_data.values, 
                   color=self.colors['good_peers'], linewidth=3, 
                   label='Good Peers', marker='o', markersize=4)
            
            # Plot bad peers trust (if exists)
            bad_data = threat_data.groupby('round')['mean_bad_trust'].mean()
            bad_data_clean = bad_data.dropna()
            if len(bad_data_clean) > 0:
                ax.plot(bad_data_clean.index, bad_data_clean.values, 
                       color=self.colors['bad_peers'], linewidth=3, 
                       label='Bad Peers', marker='s', markersize=4)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Mean Trust Value')
            ax.set_title(f'Threat Level: {threat_level}% ({threat_data["config_label"].iloc[0]})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        # Hide unused subplots
        for idx in range(len(threat_levels), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Trust Evolution Over Time by Threat Level', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{self.output_dir}/trust_evolution_by_threat.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Trust evolution plot saved: {plot_filename}")
        plt.show()
        plt.close()
        
        return plot_filename
    
    def create_final_trust_vs_threat_plot(self, trust_df):
        """Crea grafico dei valori medi di trust vs threat percentage."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calcola valori medi di trust per ogni configurazione
        mean_trust_data = []
        
        for config in trust_df['config_label'].unique():
            config_data = trust_df[trust_df['config_label'] == config]
            
            if len(config_data) > 0:
                row = config_data.iloc[0]
                
                # Calcola le medie di tutti i valori di trust durante l'esperimento
                mean_good_trust = config_data['mean_good_trust'].mean()
                mean_bad_trust = config_data['mean_bad_trust'].mean() if not config_data['mean_bad_trust'].isna().all() else np.nan
                
                mean_trust_data.append({
                    'config_label': config,
                    'total_threat_perc': row['total_threat_perc'],
                    'malicious_perc': row['malicious_perc'],
                    'corrupted_perc': row['corrupted_perc'],
                    'is_baseline': row['is_baseline'],
                    'mean_good_trust': mean_good_trust,
                    'mean_bad_trust': mean_bad_trust,
                    'std_good_trust': config_data['mean_good_trust'].std(),
                    'std_bad_trust': config_data['mean_bad_trust'].std() if not config_data['mean_bad_trust'].isna().all() else np.nan
                })
        
        mean_df = pd.DataFrame(mean_trust_data)
        mean_df = mean_df.sort_values('total_threat_perc')
        
        # Plot good peers trust with error bars
        ax.errorbar(mean_df['total_threat_perc'], mean_df['mean_good_trust'], 
                   yerr=mean_df['std_good_trust'],
                   color=self.colors['good_peers'], linewidth=4,
                   marker='o', markersize=8, markerfacecolor='white', 
                   markeredgecolor=self.colors['good_peers'], markeredgewidth=3,
                   capsize=3, capthick=1, elinewidth=1, linestyle='--',
                   ecolor=self.colors['good_peers'], alpha=0.7,
                   label='Good Peers Mean Trust')
        
        # Plot bad peers trust (excluding baseline) with error bars
        bad_data = mean_df[~mean_df['is_baseline']]
        bad_data_clean = bad_data.dropna(subset=['mean_bad_trust'])
        
        if len(bad_data_clean) > 0:
            ax.errorbar(bad_data_clean['total_threat_perc'], bad_data_clean['mean_bad_trust'], 
                       yerr=bad_data_clean['std_bad_trust'],
                       color=self.colors['bad_peers'], linewidth=4,
                       marker='s', markersize=8, markerfacecolor='white', 
                       markeredgecolor=self.colors['bad_peers'], markeredgewidth=3,
                       capsize=3, capthick=1, elinewidth=1, linestyle='--',
                       ecolor=self.colors['bad_peers'], alpha=0.7,
                       label='Bad Peers Mean Trust')
        
        # Highlight baseline
        baseline_data = mean_df[mean_df['is_baseline']]
        if len(baseline_data) > 0:
            ax.scatter(0, baseline_data['mean_good_trust'].iloc[0], 
                      s=400, color=self.colors['baseline'], marker='*', 
                      label='Baseline', alpha=1.0, edgecolors='black', linewidth=3, zorder=5)
        
        # Annotate points with smart vertical positioning to avoid overlaps
        annotations = []
        
        # Group points by x-coordinate (threat percentage) to handle vertical overlaps
        x_groups = {}
        for idx, (_, row) in enumerate(mean_df.iterrows()):
            if not row['is_baseline']:
                x_coord = row['total_threat_perc']
                if x_coord not in x_groups:
                    x_groups[x_coord] = []
                x_groups[x_coord].append((idx, row))
        
        # Debug: print x_coord values to check what we're getting
        print(f"DEBUG: x_coord values found: {list(x_groups.keys())}")
        
        # Annotate each group with smart positioning
        for x_coord, points in x_groups.items():
            print(f"DEBUG: Processing x_coord={x_coord}, type={type(x_coord)}, points={len(points)}")
            
            for point_idx, (idx, row) in enumerate(points):
                # Smart positioning: alternate above and below for multiple points at same x
                if len(points) == 1:
                    # Single point: position slightly above
                    x_offset = 5
                    y_offset = 12
                elif len(points) == 2:
                    # Special handling for problematic columns (30% and 40%)
                    # Check both int and float versions
                    if x_coord == 30 or x_coord == 30.0 or x_coord == 40 or x_coord == 40.0:
                        print(f"DEBUG: Applying special positioning for x_coord={x_coord}")
                        # More aggressive separation for 30% and 40%
                        if point_idx == 0:
                            x_offset = -15  # Left and below
                            y_offset = -25
                        else:
                            x_offset = 15   # Right and above
                            y_offset = 20
                    else:
                        # Normal logic for other threat levels
                        if point_idx == 0:
                            x_offset = 5
                            y_offset = 15  # Above
                        else:
                            x_offset = 5
                            y_offset = -20  # Below
                else:
                    if x_coord == 30 or x_coord == 30.0 or x_coord == 40 or x_coord == 40.0:
                        print(f"DEBUG: Applying special positioning for x_coord={x_coord}")
                        # More aggressive separation for 30% and 40%
                        if point_idx == 0:
                            x_offset = 10  # Left and below
                            y_offset = 25
                        else:
                            x_offset = -10   # Right and above
                            y_offset = -15
                    else:
                        # Multiple points: distribute vertically
                        x_offset = 5 + (point_idx % 2) * 8
                        y_offset = 15 + (point_idx - len(points)//2) * 18
                
                # Alternate left/right based on index for better distribution
                if idx % 2 == 1:
                    x_offset = -x_offset
                
                annotation = ax.annotate(f"{row['malicious_perc']}%M,{row['corrupted_perc']}%C", 
                           (row['total_threat_perc'], row['mean_good_trust']),
                           xytext=(x_offset, y_offset), textcoords='offset points', 
                           fontsize=10, alpha=0.95, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.95, 
                                   edgecolor='gray', linewidth=1),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.05', 
                                         color='gray', alpha=0.8, lw=1))
        
        ax.set_xlabel('Total Threat Percentage (%)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Mean Trust Value', fontsize=18, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, loc='lower left')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{self.output_dir}/mean_trust_vs_threat.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Mean trust vs threat plot saved: {plot_filename}")
        plt.show()
        plt.close()
        
        return plot_filename
    
    def create_trust_degradation_analysis(self, trust_df):
        """Analisi della degradazione del trust nel tempo."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calcola degradazione del trust nel tempo per ogni configurazione
        degradation_data = []
        
        for config in trust_df['config_label'].unique():
            config_data = trust_df[trust_df['config_label'] == config]
            
            if len(config_data) > 1:
                # Calcola trend della degradazione
                rounds = config_data['round'].values
                good_trust = config_data['mean_good_trust'].values
                bad_trust = config_data['mean_bad_trust'].values
                
                # Linear fit per good trust
                if len(good_trust) > 1 and not np.all(np.isnan(good_trust)):
                    good_slope, _, _, _, _ = stats.linregress(rounds, good_trust)
                else:
                    good_slope = 0
                
                # Linear fit per bad trust (se disponibile)
                bad_slope = 0
                if len(bad_trust) > 1 and not np.all(np.isnan(bad_trust)):
                    bad_clean_idx = ~np.isnan(bad_trust)
                    if np.sum(bad_clean_idx) > 1:
                        bad_slope, _, _, _, _ = stats.linregress(rounds[bad_clean_idx], bad_trust[bad_clean_idx])
                
                row_data = config_data.iloc[0]
                degradation_data.append({
                    'config_label': config,
                    'total_threat_perc': row_data['total_threat_perc'],
                    'good_trust_slope': good_slope,
                    'bad_trust_slope': bad_slope,
                    'is_baseline': row_data['is_baseline']
                })
        
        degrad_df = pd.DataFrame(degradation_data)
        degrad_df = degrad_df.sort_values('total_threat_perc')
        
        # Plot 1: Good trust degradation
        ax1.bar(range(len(degrad_df)), degrad_df['good_trust_slope'], 
               color=[self.colors['baseline'] if is_base else self.colors['good_peers'] 
                     for is_base in degrad_df['is_baseline']], alpha=0.8)
        
        ax1.set_xlabel('Experiment Configuration')
        ax1.set_ylabel('Trust Degradation Rate (per round)')
        ax1.set_title('Good Peers Trust Degradation Rate')
        ax1.set_xticks(range(len(degrad_df)))
        ax1.set_xticklabels(degrad_df['config_label'], rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Bad trust degradation
        bad_data = degrad_df[~degrad_df['is_baseline']]
        if len(bad_data) > 0:
            ax2.bar(range(len(bad_data)), bad_data['bad_trust_slope'], 
                   color=self.colors['bad_peers'], alpha=0.8)
            
            ax2.set_xlabel('Threat Configuration')
            ax2.set_ylabel('Trust Degradation Rate (per round)')
            ax2.set_title('Bad Peers Trust Degradation Rate')
            ax2.set_xticks(range(len(bad_data)))
            ax2.set_xticklabels(bad_data['config_label'], rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{self.output_dir}/trust_degradation_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Trust degradation analysis saved: {plot_filename}")
        plt.show()
        plt.close()
        
        return plot_filename
    
    def save_trust_summary_csv(self, trust_df):
        """Salva riassunto dei dati di trust in CSV."""
        # Calcola statistiche riassuntive per configurazione
        summary_data = []
        
        for config in trust_df['config_label'].unique():
            config_data = trust_df[trust_df['config_label'] == config]
            
            row_data = config_data.iloc[0]
            
            summary_data.append({
                'configuration': config,
                'malicious_perc': row_data['malicious_perc'],
                'corrupted_perc': row_data['corrupted_perc'], 
                'total_threat_perc': row_data['total_threat_perc'],
                'is_baseline': row_data['is_baseline'],
                'initial_good_trust': config_data['mean_good_trust'].iloc[0],
                'final_good_trust': config_data['mean_good_trust'].iloc[-1],
                'initial_bad_trust': config_data['mean_bad_trust'].iloc[0] if not pd.isna(config_data['mean_bad_trust'].iloc[0]) else None,
                'final_bad_trust': config_data['mean_bad_trust'].iloc[-1] if not pd.isna(config_data['mean_bad_trust'].iloc[-1]) else None,
                'good_trust_change': config_data['mean_good_trust'].iloc[-1] - config_data['mean_good_trust'].iloc[0],
                'rounds_analyzed': len(config_data)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('total_threat_perc')
        
        csv_filename = f"{self.output_dir}/trust_analysis_summary.csv"
        summary_df.to_csv(csv_filename, index=False)
        
        print(f"💾 Trust summary saved: {csv_filename}")
        
        return csv_filename, summary_df
    
    def run_analysis(self):
        """Esegui l'analisi completa del trust."""
        print("🚀 Starting Trust Analysis for DAGTrustFL Experiments")
        print("="*60)
        
        # Scansiona le cartelle
        if not self.scan_experiment_folders():
            return False
        
        # Carica i dati di trust
        trust_df = self.load_trust_data()
        if trust_df is None:
            return False
        
        # Crea i grafici
        evolution_plot = self.create_trust_evolution_plot(trust_df)
        mean_plot = self.create_final_trust_vs_threat_plot(trust_df)
        degradation_plot = self.create_trust_degradation_analysis(trust_df)
        
        # Salva riassunto
        csv_file, summary_df = self.save_trust_summary_csv(trust_df)
        
        print("\n" + "="*60)
        print("✅ TRUST ANALYSIS COMPLETED!")
        print(f"📊 Trust evolution plot: {evolution_plot}")
        print(f"📊 Mean trust vs threat plot: {mean_plot}")
        print(f"📊 Trust degradation analysis: {degradation_plot}")
        print(f"💾 Summary CSV: {csv_file}")
        print(f"📁 Output directory: {self.output_dir}")
        print("="*60)
        
        # Print summary statistics
        print(f"\n📈 TRUST SUMMARY STATISTICS:")
        print(summary_df.to_string(index=False))
        
        return True


def main():
    """Funzione principale."""
    plotter = TrustAnalysisPlotter()
    success = plotter.run_analysis()
    
    if success:
        print("\n🎉 Trust analysis completed successfully!")
    else:
        print("\n❌ Trust analysis failed!")
    
    return success


if __name__ == "__main__":
    main()
