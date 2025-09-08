#!/usr/bin/env python3
"""
Trust vs No-Trust Comparison Plotter
=====================================

Questo script crea grafici di confronto tra i metodi "With Trust" e "Without Trust"
mostrando l'andamento delle metriche (FPR, FNR, DACC) al variare del threat level.

Autore: DAGTrustFL Analysis Suite
Data: 2025-09-08
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path

# Set matplotlib font configuration
plt.rcParams.update({
    'font.family': ['Roboto', 'DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'text.usetex': False
})

# Set seaborn style
sns.set_style("whitegrid")

class TrustComparisonPlotter:
    """Classe per creare grafici di confronto Trust vs No-Trust."""
    
    def __init__(self, data_path="csv_data/trust_and_notrust.csv"):
        """
        Inizializza il plotter.
        
        Args:
            data_path (str): Percorso al file CSV con i dati di confronto
        """
        self.data_path = Path(data_path)
        self.output_dir = "comparison_plots"
        
        # Crea directory per i grafici
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'with_trust': '#2E8B57',      # Verde per With Trust
            'without_trust': '#DC143C',   # Rosso per Without Trust
            'grid': '#E0E0E0'
        }
        
        self.markers = {
            'with_trust': 'o',
            'without_trust': 's'
        }
        
    def load_data(self):
        """Carica e preprocessa i dati."""
        if not self.data_path.exists():
            print(f"❌ File not found: {self.data_path}")
            return None
        
        df = pd.read_csv(self.data_path)
        print(f"📊 Loaded {len(df)} rows of data")
        
        # Estrai il threat percentage totale dal scenario
        def extract_threat_percentage(scenario):
            # Scenario format: "10%M 0%C" -> 10+0=10
            parts = scenario.split()
            mal_perc = int(parts[0].replace('%M', ''))
            cor_perc = int(parts[1].replace('%C', ''))
            return mal_perc + cor_perc
        
        df['Total_Threat_Percentage'] = df['Scenario'].apply(extract_threat_percentage)
        
        # Separa i dati per metodo
        with_trust = df[df['Method'] == 'With Trust'].copy()
        without_trust = df[df['Method'] == 'Without Trust'].copy()
        
        print(f"✓ With Trust: {len(with_trust)} configurations")
        print(f"✓ Without Trust: {len(without_trust)} configurations")
        
        return df, with_trust, without_trust
    
    def create_metric_comparison_plot(self, df, with_trust, without_trust, metric='DACC'):
        """Crea grafico di confronto per una metrica specifica."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ordina i dati per threat percentage
        with_trust_sorted = with_trust.sort_values('Total_Threat_Percentage')
        without_trust_sorted = without_trust.sort_values('Total_Threat_Percentage')
        
        # Plot delle linee
        ax.plot(with_trust_sorted['Total_Threat_Percentage'], 
               with_trust_sorted[metric],
               color=self.colors['with_trust'], 
               linewidth=3, 
               marker=self.markers['with_trust'], 
               markersize=8,
               markerfacecolor='white',
               markeredgecolor=self.colors['with_trust'],
               markeredgewidth=2,
               label='With Trust (DAGTrustFL)')
        
        ax.plot(without_trust_sorted['Total_Threat_Percentage'], 
               without_trust_sorted[metric],
               color=self.colors['without_trust'], 
               linewidth=3, 
               marker=self.markers['without_trust'], 
               markersize=8,
               markerfacecolor='white',
               markeredgecolor=self.colors['without_trust'],
               markeredgewidth=2,
               label='Without Trust (Baseline)')
        
        # Configurazione assi
        ax.set_xlabel('Total Threat Percentage (%)', fontweight='bold')
        
        if metric == 'DACC':
            ax.set_ylabel('Detection Accuracy (%)', fontweight='bold')
            ax.set_title('Detection Accuracy Comparison', fontweight='bold')
        elif metric == 'FPR':
            ax.set_ylabel('False Positive Rate (%)', fontweight='bold')
            ax.set_title('False Positive Rate Comparison', fontweight='bold')
        elif metric == 'FNR':
            ax.set_ylabel('False Negative Rate (%)', fontweight='bold')
            ax.set_title('False Negative Rate Comparison', fontweight='bold')
        
        # Configurazione grafico
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Imposta limiti assi
        threat_levels = sorted(df['Total_Threat_Percentage'].unique())
        ax.set_xticks(threat_levels)
        
        if metric == 'DACC':
            ax.set_ylim(50, 105)  # Per accuracy
        elif metric == 'FPR':
            ax.set_ylim(0, max(df[metric].max() * 1.1, 10))  # Per FPR
        elif metric == 'FNR':
            ax.set_ylim(0, max(df[metric].max() * 1.1, 10))  # Per FNR
        
        plt.tight_layout()
        
        # Salva il grafico
        plot_filename = f"{self.output_dir}/{metric.lower()}_comparison.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ {metric} comparison plot saved: {plot_filename}")
        plt.show()
        plt.close()
        
        return plot_filename
    
    def create_combined_comparison_plot(self, df, with_trust, without_trust):
        """Crea grafico combinato con tutte le metriche."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['DACC', 'FPR', 'FNR']
        titles = ['Detection Accuracy (%)', 'False Positive Rate (%)', 'False Negative Rate (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            # Ordina i dati per threat percentage
            with_trust_sorted = with_trust.sort_values('Total_Threat_Percentage')
            without_trust_sorted = without_trust.sort_values('Total_Threat_Percentage')
            
            # Plot delle linee
            ax.plot(with_trust_sorted['Total_Threat_Percentage'], 
                   with_trust_sorted[metric],
                   color=self.colors['with_trust'], 
                   linewidth=3, 
                   marker=self.markers['with_trust'], 
                   markersize=6,
                   markerfacecolor='white',
                   markeredgecolor=self.colors['with_trust'],
                   markeredgewidth=2,
                   label='With Trust')
            
            ax.plot(without_trust_sorted['Total_Threat_Percentage'], 
                   without_trust_sorted[metric],
                   color=self.colors['without_trust'], 
                   linewidth=3, 
                   marker=self.markers['without_trust'], 
                   markersize=6,
                   markerfacecolor='white',
                   markeredgecolor=self.colors['without_trust'],
                   markeredgewidth=2,
                   label='Without Trust')
            
            # Configurazione
            ax.set_xlabel('Total Threat Percentage (%)', fontweight='bold')
            ax.set_ylabel(title, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Imposta limiti
            threat_levels = sorted(df['Total_Threat_Percentage'].unique())
            ax.set_xticks(threat_levels)
            
            if metric == 'DACC':
                ax.set_ylim(50, 105)
            else:
                ax.set_ylim(0, max(df[metric].max() * 1.1, 10))
            
            # Legenda solo nel primo subplot
            if idx == 0:
                ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Salva il grafico combinato
        plot_filename = f"{self.output_dir}/combined_metrics_comparison.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Combined comparison plot saved: {plot_filename}")
        plt.show()
        plt.close()
        
        return plot_filename
    
    def create_improvement_analysis(self, df, with_trust, without_trust):
        """Crea analisi del miglioramento percentuale."""
        
        # Merge dei dati per calcolare i miglioramenti
        merged = pd.merge(with_trust, without_trust, 
                         on='Total_Threat_Percentage', 
                         suffixes=('_with', '_without'))
        
        # Calcola miglioramenti
        merged['DACC_improvement'] = merged['DACC_with'] - merged['DACC_without']
        merged['FPR_improvement'] = merged['FPR_without'] - merged['FPR_with']  # Riduzione è miglioramento
        merged['FNR_improvement'] = merged['FNR_without'] - merged['FNR_with']  # Riduzione è miglioramento
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        threat_levels = merged['Total_Threat_Percentage']
        
        # Plot dei miglioramenti con tre barre per ogni threat level
        width = 1.5  # Larghezza delle barre
        x_offset = width  # Offset tra le barre
        
        ax.bar(threat_levels - x_offset, merged['DACC_improvement'], 
               width=width, label='DACC Improvement', alpha=0.7, color=self.colors['with_trust'])
        ax.bar(threat_levels, merged['FPR_improvement'], 
               width=width, label='FPR Reduction', alpha=0.7, color='orange')
        ax.bar(threat_levels + x_offset, merged['FNR_improvement'], 
               width=width, label='FNR Reduction', alpha=0.7, color='purple')
        
        ax.set_xlabel('Total Threat Percentage (%)', fontweight='bold', fontsize=20)
        ax.set_ylabel('Improvement (%)', fontweight='bold', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Imposta solo i tick desiderati (rimuove 15, 25, 35)
        desired_ticks = [10, 20, 30, 40]
        ax.set_xticks(desired_ticks)
        
        # Ingrandisci i tick
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        plt.tight_layout()
        
        # Salva il grafico di miglioramento
        plot_filename = f"{self.output_dir}/improvement_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✓ Improvement analysis plot saved: {plot_filename}")
        plt.show()
        plt.close()
        
        return plot_filename, merged
    
    def run_analysis(self):
        """Esegui l'analisi completa."""
        print("🚀 Starting Trust vs No-Trust Comparison Analysis")
        print("="*60)
        
        # Carica i dati
        data = self.load_data()
        if data is None:
            return False
        
        df, with_trust, without_trust = data
        
        # Crea grafici individuali per ogni metrica
        dacc_plot = self.create_metric_comparison_plot(df, with_trust, without_trust, 'DACC')
        fpr_plot = self.create_metric_comparison_plot(df, with_trust, without_trust, 'FPR')
        fnr_plot = self.create_metric_comparison_plot(df, with_trust, without_trust, 'FNR')
        
        # Crea grafico combinato
        combined_plot = self.create_combined_comparison_plot(df, with_trust, without_trust)
        
        # Crea analisi miglioramenti
        improvement_plot, improvement_data = self.create_improvement_analysis(df, with_trust, without_trust)
        
        print("\n" + "="*60)
        print("✅ COMPARISON ANALYSIS COMPLETED!")
        print(f"📊 DACC comparison: {dacc_plot}")
        print(f"📊 FPR comparison: {fpr_plot}")
        print(f"📊 FNR comparison: {fnr_plot}")
        print(f"📊 Combined comparison: {combined_plot}")
        print(f"📊 Improvement analysis: {improvement_plot}")
        print(f"📁 Output directory: {self.output_dir}")
        print("="*60)
        
        # Stampa statistiche di miglioramento
        print(f"\n📈 IMPROVEMENT STATISTICS:")
        for _, row in improvement_data.iterrows():
            threat = row['Total_Threat_Percentage']
            dacc_imp = row['DACC_improvement']
            fpr_imp = row['FPR_improvement']
            fnr_imp = row['FNR_improvement']
            print(f"   {threat}% threat: DACC +{dacc_imp:.1f}%, FPR -{fpr_imp:.1f}%, FNR -{fnr_imp:.1f}%")
        
        return True


def main():
    """Funzione principale."""
    plotter = TrustComparisonPlotter()
    success = plotter.run_analysis()
    
    if success:
        print("\n🎉 Trust comparison analysis completed successfully!")
    else:
        print("\n❌ Trust comparison analysis failed!")
    
    return success


if __name__ == "__main__":
    main()
