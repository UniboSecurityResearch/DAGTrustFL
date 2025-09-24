#!/usr/bin/env python3
"""
FL Results Comparator for DAGTrustFL Trust Experiments
====================================================

Questo script confronta i risultati del Federated Learning tra diversi esperimenti
trust analizzando i file 3000_results.csv dalle cartelle results_percentages_trust.

Basato su analyze_fl_csv_pdf.py ma focalizzato sul confronto tra modelli con
diversi livelli di minacce (malicious e corrupted percentages).

Autore: Giacomo Gori
Data: 2025-09-02
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
import re
from pathlib import Path
from datetime import datetime
from scipy import stats

# Set matplotlib font configuration like other plots
plt.rcParams.update({
    'font.family': ['Roboto', 'DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'text.usetex': False
})

# Set seaborn style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class FLTrustComparator:
    """Classe per confrontare i risultati FL tra esperimenti trust."""
    
    def __init__(self, results_base_path="../FL/results_percentages_trust", outlier_threshold=2.0):
        """
        Inizializza il comparatore.
        
        Args:
            results_base_path (str): Percorso base dove sono i risultati trust
            outlier_threshold (float): Z-score threshold per outlier detection (default: 2.0)
        """
        self.results_base_path = Path(results_base_path)
        self.output_dir = "../plots/percentages_trust_comparison_plots/fl_comparison_plots"
        self.experiments = {}
        self.num_classes = 10
        self.outlier_threshold = outlier_threshold
        self.outliers_detected = {}  # Track detected outliers
        
        # Crea directory per i grafici
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color scheme consistent with other plots
        self.colors = {
            'baseline': '#1f77b4',
            'low_threat': '#2ca02c',
            'medium_threat': '#ff7f0e', 
            'high_threat': '#d62728',
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D'
        }
        
    def scan_experiment_folders(self):
        """Scansiona le cartelle degli esperimenti trust."""
        print("🔍 Scanning for FL trust experiment folders...")
        
        if not self.results_base_path.exists():
            print(f"❌ Results path not found: {self.results_base_path}")
            return False
        
        # Pattern: mal_X_cor_Y_trust (e.g., mal_20_cor_10_trust for 20% malicious, 10% corrupted)
        pattern = r'mal_(\d+)_cor_(\d+)_trust'
        
        for folder in self.results_base_path.iterdir():
            if folder.is_dir():
                match = re.match(pattern, folder.name)
                if match:
                    # Extract percentages
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
                        if malicious_perc == 0 and corrupted_perc == 0:
                            print(f"   ✓ Found BASELINE FL: {malicious_perc}% mal, {corrupted_perc}% cor: {folder.name}")
                        else:
                            print(f"   ✓ Found FL results: {malicious_perc}% mal, {corrupted_perc}% cor: {folder.name}")
                    else:
                        print(f"   ⚠️ No FL CSV file in {folder.name}")
        
        if not self.experiments:
            print("❌ No valid FL experiment folders found!")
            return False
        
        print(f"\n📊 Found {len(self.experiments)} FL trust experiments")
        return True
    
    def load_fl_experiment_data(self, fl_csv_path, malicious_perc, corrupted_perc):
        """Carica e processa i dati FL da un singolo esperimento."""
        try:
            # Carica il CSV senza header (stesso formato di analyze_fl_csv_pdf.py)
            raw_data = pd.read_csv(fl_csv_path, header=None)
            
            # Struttura i dati
            df = pd.DataFrame({
                'epoch': range(1, len(raw_data) + 1),
                'accuracy': raw_data.iloc[:, 0],
                'loss': raw_data.iloc[:, 1]
            })
            
            # Aggiungi precision per ogni classe
            for i in range(self.num_classes):
                df[f'precision_class_{i}'] = raw_data.iloc[:, 2 + i]
            
            # Aggiungi recall per ogni classe  
            for i in range(self.num_classes):
                df[f'recall_class_{i}'] = raw_data.iloc[:, 12 + i]
            
            # Calcola metriche aggregate
            precision_cols = [f'precision_class_{i}' for i in range(self.num_classes)]
            recall_cols = [f'recall_class_{i}' for i in range(self.num_classes)]
            
            df['avg_precision'] = df[precision_cols].mean(axis=1)
            df['avg_recall'] = df[recall_cols].mean(axis=1)
            df['avg_f1_score'] = 2 * (df['avg_precision'] * df['avg_recall']) / (df['avg_precision'] + df['avg_recall'])
            
            # Aggiungi metadati esperimento
            df['malicious_perc'] = malicious_perc
            df['corrupted_perc'] = corrupted_perc
            df['total_perc'] = malicious_perc + corrupted_perc
            df['is_baseline'] = (malicious_perc == 0 and corrupted_perc == 0)
            df['config_label'] = f"{malicious_perc}%M,{corrupted_perc}%C" + (" (BASELINE)" if malicious_perc == 0 and corrupted_perc == 0 else "")
            
            print(f"   📊 Loaded FL data for {malicious_perc}%M,{corrupted_perc}%C: {len(df)} epochs, accuracy {df['accuracy'].iloc[-1]:.2f}%")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading FL data from {fl_csv_path}: {e}")
            return None
    
    def detect_and_remove_outliers(self, all_fl_data):
        """
        Rileva e rimuove outlier basandosi su accuracy e loss usando metodi multipli.
        
        Args:
            all_fl_data (list): Lista di DataFrame FL
            
        Returns:
            list: Lista di DataFrame FL senza outlier
        """
        print(f"\n🔍 Detecting outliers using multiple methods:")
        print(f"   • Z-score threshold: {self.outlier_threshold}")
        print(f"   • IQR method (1.5 * IQR)")
        print(f"   • Modified Z-score (MAD-based)")
        
        if len(all_fl_data) < 3:
            print("   ⚠️ Too few experiments for outlier detection")
            return all_fl_data
        
        # Collect final metrics for outlier detection
        final_accuracies = []
        final_losses = []
        final_f1_scores = []
        experiment_labels = []
        
        for fl_data in all_fl_data:
            final_accuracies.append(fl_data['accuracy'].iloc[-1])
            final_losses.append(fl_data['loss'].iloc[-1])
            final_f1_scores.append(fl_data['avg_f1_score'].iloc[-1])
            experiment_labels.append(fl_data['config_label'].iloc[0])
        
        final_accuracies = np.array(final_accuracies)
        final_losses = np.array(final_losses)
        final_f1_scores = np.array(final_f1_scores)
        
        # Method 1: Standard Z-score
        acc_z_scores = np.abs(stats.zscore(final_accuracies, nan_policy='omit'))
        loss_z_scores = np.abs(stats.zscore(final_losses, nan_policy='omit'))
        f1_z_scores = np.abs(stats.zscore(final_f1_scores, nan_policy='omit'))
        
        # Method 2: Modified Z-score (using MAD - Median Absolute Deviation)
        def modified_z_score(data):
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad == 0:
                return np.zeros_like(data)
            return 0.6745 * (data - median) / mad
        
        acc_mod_z = np.abs(modified_z_score(final_accuracies))
        loss_mod_z = np.abs(modified_z_score(final_losses))
        f1_mod_z = np.abs(modified_z_score(final_f1_scores))
        
        # Method 3: IQR method
        def iqr_outliers(data, factor=1.5):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        acc_iqr_outliers = iqr_outliers(final_accuracies)
        loss_iqr_outliers = iqr_outliers(final_losses)
        f1_iqr_outliers = iqr_outliers(final_f1_scores)
        
        # Combine methods - an experiment is an outlier if flagged by multiple methods
        outlier_votes = np.zeros(len(all_fl_data))
        
        # Z-score method (weight: 1)
        z_outliers = (acc_z_scores > self.outlier_threshold) | \
                    (loss_z_scores > self.outlier_threshold) | \
                    (f1_z_scores > self.outlier_threshold)
        outlier_votes += z_outliers.astype(int)
        
        # Modified Z-score method (weight: 1) - more robust to outliers
        mod_z_outliers = (acc_mod_z > 3.5) | (loss_mod_z > 3.5) | (f1_mod_z > 3.5)
        outlier_votes += mod_z_outliers.astype(int)
        
        # IQR method (weight: 1)
        iqr_outliers = acc_iqr_outliers | loss_iqr_outliers | f1_iqr_outliers
        outlier_votes += iqr_outliers.astype(int)
        
        # Additional check for extreme values
        acc_mean, acc_std = np.mean(final_accuracies), np.std(final_accuracies)
        loss_mean, loss_std = np.mean(final_losses), np.std(final_losses)
        
        # Flag as outlier if accuracy is more than 3 std devs away OR loss is more than 3 std devs away
        extreme_acc = np.abs(final_accuracies - acc_mean) > 3 * acc_std
        extreme_loss = np.abs(final_losses - loss_mean) > 3 * loss_std
        extreme_outliers = extreme_acc | extreme_loss
        outlier_votes += extreme_outliers.astype(int) * 2  # Higher weight for extreme outliers
        
        # An experiment is considered an outlier if it gets 2 or more votes
        outlier_threshold_votes = 2
        outlier_mask = outlier_votes >= outlier_threshold_votes
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) > 0:
            print(f"   🚨 Detected {len(outlier_indices)} outlier experiment(s):")
            
            for idx in outlier_indices:
                label = experiment_labels[idx]
                acc_val = final_accuracies[idx]
                loss_val = final_losses[idx]
                f1_val = final_f1_scores[idx]
                votes = int(outlier_votes[idx])
                
                outlier_reasons = []
                
                # Check which methods flagged this experiment
                if z_outliers[idx]:
                    outlier_reasons.append(f"Z-score: Acc={acc_z_scores[idx]:.2f}, Loss={loss_z_scores[idx]:.2f}, F1={f1_z_scores[idx]:.2f}")
                if mod_z_outliers[idx]:
                    outlier_reasons.append(f"Mod-Z: Acc={acc_mod_z[idx]:.2f}, Loss={loss_mod_z[idx]:.2f}, F1={f1_mod_z[idx]:.2f}")
                if iqr_outliers[idx]:
                    outlier_reasons.append("IQR method")
                if extreme_outliers[idx]:
                    outlier_reasons.append("Extreme values (>3σ)")
                
                print(f"      ❌ {label} (Votes: {votes}): Acc={acc_val:.2f}%, Loss={loss_val:.4f}, F1={f1_val:.3f}")
                print(f"         Reasons: {'; '.join(outlier_reasons)}")
                
                # Store outlier info
                self.outliers_detected[label] = {
                    'accuracy_z_score': acc_z_scores[idx],
                    'loss_z_score': loss_z_scores[idx],
                    'f1_z_score': f1_z_scores[idx],
                    'accuracy_mod_z': acc_mod_z[idx],
                    'loss_mod_z': loss_mod_z[idx],
                    'f1_mod_z': f1_mod_z[idx],
                    'iqr_outlier': iqr_outliers[idx],
                    'extreme_outlier': extreme_outliers[idx],
                    'votes': votes,
                    'final_accuracy': acc_val,
                    'final_loss': loss_val,
                    'final_f1': f1_val,
                    'reasons': outlier_reasons
                }
            
            # Safety check: don't remove more than 50% of experiments
            max_removable = max(1, len(all_fl_data) // 2)
            if len(outlier_indices) > max_removable:
                print(f"   ⚠️ Too many outliers detected ({len(outlier_indices)}), keeping the {max_removable} least problematic")
                # Sort by votes and keep only the worst outliers
                sorted_indices = sorted(outlier_indices, key=lambda i: outlier_votes[i], reverse=True)
                outlier_indices = sorted_indices[:max_removable]
            
            # Remove outliers
            clean_fl_data = [fl_data for i, fl_data in enumerate(all_fl_data) if i not in outlier_indices]
            
            print(f"   ✅ Removed {len(outlier_indices)} outlier(s), {len(clean_fl_data)} experiments remaining")
            
            # Show remaining experiments statistics
            remaining_acc = [final_accuracies[i] for i in range(len(final_accuracies)) if i not in outlier_indices]
            remaining_loss = [final_losses[i] for i in range(len(final_losses)) if i not in outlier_indices]
            
            if remaining_acc:
                print(f"   📊 Remaining experiments - Accuracy range: {min(remaining_acc):.2f}% - {max(remaining_acc):.2f}%")
                print(f"                           - Loss range: {min(remaining_loss):.4f} - {max(remaining_loss):.4f}")
            
            return clean_fl_data
        
        else:
            print("   ✅ No outliers detected, all experiments retained")
            return all_fl_data
    
    def create_outlier_analysis_page(self, pdf):
        """Crea una pagina di analisi degli outlier."""
        if not self.outliers_detected:
            return
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        outlier_text = "OUTLIER DETECTION ANALYSIS\n"
        outlier_text += "="*40 + "\n\n"
        
        outlier_text += f"Z-SCORE THRESHOLD: {self.outlier_threshold}\n"
        outlier_text += f"DETECTION METHODS: Multiple (Z-score, Modified Z-score, IQR, Extreme values)\n"
        outlier_text += f"OUTLIERS DETECTED: {len(self.outliers_detected)}\n\n"
        
        if self.outliers_detected:
            outlier_text += "OUTLIER DETAILS:\n"
            outlier_text += "-" * 50 + "\n"
            
            for label, info in self.outliers_detected.items():
                outlier_text += f"\n• {label} (Votes: {info['votes']}):\n"
                outlier_text += f"  Final Accuracy: {info['final_accuracy']:.2f}%\n"
                outlier_text += f"  Final Loss: {info['final_loss']:.4f}\n"
                outlier_text += f"  Final F1-Score: {info['final_f1']:.3f}\n"
                outlier_text += f"  Standard Z-Scores:\n"
                outlier_text += f"    - Accuracy: {info['accuracy_z_score']:.2f}\n"
                outlier_text += f"    - Loss: {info['loss_z_score']:.2f}\n"
                outlier_text += f"    - F1: {info['f1_z_score']:.2f}\n"
                outlier_text += f"  Modified Z-Scores:\n"
                outlier_text += f"    - Accuracy: {info['accuracy_mod_z']:.2f}\n"
                outlier_text += f"    - Loss: {info['loss_mod_z']:.2f}\n"
                outlier_text += f"    - F1: {info['f1_mod_z']:.2f}\n"
                outlier_text += f"  Other flags:\n"
                outlier_text += f"    - IQR outlier: {info['iqr_outlier']}\n"
                outlier_text += f"    - Extreme outlier: {info['extreme_outlier']}\n"
                outlier_text += f"  Detection reasons: {'; '.join(info['reasons'])}\n"
            
            outlier_text += f"\nDETECTION METHODS:\n"
            outlier_text += f"1. Standard Z-score: |Z| > {self.outlier_threshold}\n"
            outlier_text += f"2. Modified Z-score (MAD-based): |Z| > 3.5\n"
            outlier_text += f"3. IQR method: outside Q1-1.5*IQR to Q3+1.5*IQR\n"
            outlier_text += f"4. Extreme values: >3 standard deviations\n\n"
            outlier_text += f"VOTING SYSTEM:\n"
            outlier_text += f"An experiment is flagged as outlier if it receives\n"
            outlier_text += f"≥2 votes from the detection methods.\n"
            outlier_text += f"Extreme outliers get extra weight (2 votes).\n\n"
            outlier_text += f"SAFETY LIMITS:\n"
            outlier_text += f"Maximum 50% of experiments can be removed\n"
            outlier_text += f"to prevent over-aggressive filtering.\n\n"
            outlier_text += f"IMPACT ON ANALYSIS:\n"
            outlier_text += f"Outliers have been removed to ensure more\n"
            outlier_text += f"reliable comparisons and trend analysis.\n"
            outlier_text += f"This improves the quality of insights\n"
            outlier_text += f"derived from the FL performance data.\n"
        
        ax.text(0.05, 0.95, outlier_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightcoral', alpha=0.8))
        
        plt.title('Outlier Detection and Removal Analysis', fontsize=18, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def create_fl_comparison_report(self):
        """Crea il report di confronto FL completo."""
        pdf_filename = f"{self.output_dir}/fl_trust_comparison_report.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            print("📑 Creating FL trust comparison PDF...")
            
            # Carica tutti i dati FL
            all_fl_data = []
            experiment_summaries = []
            
            for (mal_perc, cor_perc), exp_info in sorted(self.experiments.items()):
                fl_data = self.load_fl_experiment_data(exp_info['fl_csv_path'], mal_perc, cor_perc)
                if fl_data is not None:
                    all_fl_data.append(fl_data)
                    
                    # Crea summary per questo esperimento
                    summary = {
                        'malicious_perc': mal_perc,
                        'corrupted_perc': cor_perc,
                        'total_perc': mal_perc + cor_perc,
                        'is_baseline': (mal_perc == 0 and cor_perc == 0),
                        'config_label': fl_data['config_label'].iloc[0],
                        'final_accuracy': fl_data['accuracy'].iloc[-1],
                        'final_loss': fl_data['loss'].iloc[-1],
                        'final_avg_precision': fl_data['avg_precision'].iloc[-1],
                        'final_avg_recall': fl_data['avg_recall'].iloc[-1],
                        'final_avg_f1': fl_data['avg_f1_score'].iloc[-1],
                        'accuracy_improvement': fl_data['accuracy'].iloc[-1] - fl_data['accuracy'].iloc[0],
                        'max_accuracy': fl_data['accuracy'].max(),
                        'min_loss': fl_data['loss'].min(),
                        'epochs': len(fl_data)
                    }
                    experiment_summaries.append(summary)
            
            if not all_fl_data:
                print("❌ No valid FL data found")
                return None
            
            # Detect and remove outliers
            all_fl_data = self.detect_and_remove_outliers(all_fl_data)
            
            if not all_fl_data:
                print("❌ No FL data remaining after outlier removal")
                return None
                
            # Recreate summaries after outlier removal
            experiment_summaries = []
            for fl_data in all_fl_data:
                summary = {
                    'malicious_perc': fl_data['malicious_perc'].iloc[0],
                    'corrupted_perc': fl_data['corrupted_perc'].iloc[0],
                    'total_perc': fl_data['total_perc'].iloc[0],
                    'is_baseline': fl_data['is_baseline'].iloc[0],
                    'config_label': fl_data['config_label'].iloc[0],
                    'final_accuracy': fl_data['accuracy'].iloc[-1],
                    'final_loss': fl_data['loss'].iloc[-1],
                    'final_avg_precision': fl_data['avg_precision'].iloc[-1],
                    'final_avg_recall': fl_data['avg_recall'].iloc[-1],
                    'final_avg_f1': fl_data['avg_f1_score'].iloc[-1],
                    'accuracy_improvement': fl_data['accuracy'].iloc[-1] - fl_data['accuracy'].iloc[0],
                    'max_accuracy': fl_data['accuracy'].max(),
                    'min_loss': fl_data['loss'].min(),
                    'epochs': len(fl_data)
                }
                experiment_summaries.append(summary)
                
            summary_df = pd.DataFrame(experiment_summaries)
            
            # Add outlier analysis page first
            if self.outliers_detected:
                self.create_outlier_analysis_page(pdf)
            
            # 1. Accuracy Evolution Comparison
            self._create_accuracy_evolution_plot(pdf, all_fl_data)
            
            # 2. Loss Evolution Comparison
            self._create_loss_evolution_plot(pdf, all_fl_data)
            
            # 3. F1-Score Evolution Comparison
            self._create_f1_evolution_plot(pdf, all_fl_data)
            
            # 4. Final Performance Comparison
            self._create_final_performance_comparison(pdf, summary_df)
            
            # 5. Baseline vs Threats Analysis
            self._create_baseline_vs_threats_analysis(pdf, summary_df)
            
            # 6. Per-Class Performance Comparison
            self._create_per_class_comparison(pdf, all_fl_data)
            
            # 7. Summary Statistics Table
            self._create_fl_summary_table(pdf, summary_df)
        
        print(f"📑 FL comparison PDF saved: {pdf_filename}")
        
        # Save summary data to CSV
        csv_filename = f"{self.output_dir}/fl_trust_comparison_summary.csv"
        summary_df.to_csv(csv_filename, index=False)
        print(f"💾 FL comparison data saved: {csv_filename}")
        
        return pdf_filename
    
    def _create_accuracy_evolution_plot(self, pdf, all_fl_data):
        """Crea il plot dell'evoluzione dell'accuracy."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: All accuracy curves
        for fl_data in all_fl_data:
            color = self.colors['baseline'] if fl_data['is_baseline'].iloc[0] else self._get_threat_color(fl_data['total_perc'].iloc[0])
            linewidth = 4 if fl_data['is_baseline'].iloc[0] else 2
            alpha = 1.0 if fl_data['is_baseline'].iloc[0] else 0.8
            
            ax1.plot(fl_data['epoch'], fl_data['accuracy'], color=color, linewidth=linewidth, 
                    alpha=alpha, label=fl_data['config_label'].iloc[0])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('🎯 Accuracy Evolution Comparison')
        ax1.legend(fontsize=12, loc='lower right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Final accuracy vs threat level
        baseline_data = [fl for fl in all_fl_data if fl['is_baseline'].iloc[0]]
        threat_data = [fl for fl in all_fl_data if not fl['is_baseline'].iloc[0]]
        
        if baseline_data:
            baseline_acc = baseline_data[0]['accuracy'].iloc[-1]
            ax2.axhline(baseline_acc, color=self.colors['baseline'], linestyle='--', 
                       linewidth=4, label=f'Baseline: {baseline_acc:.1f}%', alpha=0.9)
        
        if threat_data:
            threat_levels = [fl['total_perc'].iloc[0] for fl in threat_data]
            threat_accuracies = [fl['accuracy'].iloc[-1] for fl in threat_data]
            threat_labels = [fl['config_label'].iloc[0] for fl in threat_data]
            
            colors = [self._get_threat_color(level) for level in threat_levels]
            bars = ax2.bar(range(len(threat_levels)), threat_accuracies, color=colors, alpha=0.8)
            
            ax2.set_xlabel('Threat Configuration')
            ax2.set_ylabel('Final Accuracy (%)')
            ax2.set_title('📊 Final Accuracy vs Threat Level')
            ax2.set_xticks(range(len(threat_labels)))
            ax2.set_xticklabels(threat_labels, rotation=45)
            ax2.legend(fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy improvement
        improvements = []
        labels = []
        colors_list = []
        
        for fl_data in all_fl_data:
            improvement = fl_data['accuracy'].iloc[-1] - fl_data['accuracy'].iloc[0]
            improvements.append(improvement)
            labels.append(fl_data['config_label'].iloc[0])
            colors_list.append(self.colors['baseline'] if fl_data['is_baseline'].iloc[0] else self._get_threat_color(fl_data['total_perc'].iloc[0]))
        
        bars = ax3.bar(range(len(improvements)), improvements, color=colors_list, alpha=0.8)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Accuracy Improvement (%)')
        ax3.set_title('📈 Learning Progress Comparison')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Accuracy stability (coefficient of variation)
        stabilities = []
        for fl_data in all_fl_data:
            cv = fl_data['accuracy'].std() / fl_data['accuracy'].mean()
            stabilities.append(cv)
        
        bars = ax4.bar(range(len(stabilities)), stabilities, color=colors_list, alpha=0.8)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Coefficient of Variation')
        ax4.set_title('📊 Training Stability Comparison')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('FL Accuracy Analysis - Trust Experiments', fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_loss_evolution_plot(self, pdf, all_fl_data):
        """Crea il plot dell'evoluzione della loss."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: All loss curves
        for fl_data in all_fl_data:
            color = self.colors['baseline'] if fl_data['is_baseline'].iloc[0] else self._get_threat_color(fl_data['total_perc'].iloc[0])
            linewidth = 4 if fl_data['is_baseline'].iloc[0] else 2
            alpha = 1.0 if fl_data['is_baseline'].iloc[0] else 0.8
            
            ax1.plot(fl_data['epoch'], fl_data['loss'], color=color, linewidth=linewidth, 
                    alpha=alpha, label=fl_data['config_label'].iloc[0])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('📉 Loss Evolution Comparison')
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Final loss vs threat level
        baseline_data = [fl for fl in all_fl_data if fl['is_baseline'].iloc[0]]
        threat_data = [fl for fl in all_fl_data if not fl['is_baseline'].iloc[0]]
        
        if baseline_data:
            baseline_loss = baseline_data[0]['loss'].iloc[-1]
            ax2.axhline(baseline_loss, color=self.colors['baseline'], linestyle='--', 
                       linewidth=4, label=f'Baseline: {baseline_loss:.3f}', alpha=0.9)
        
        if threat_data:
            threat_levels = [fl['total_perc'].iloc[0] for fl in threat_data]
            threat_losses = [fl['loss'].iloc[-1] for fl in threat_data]
            threat_labels = [fl['config_label'].iloc[0] for fl in threat_data]
            
            colors = [self._get_threat_color(level) for level in threat_levels]
            bars = ax2.bar(range(len(threat_levels)), threat_losses, color=colors, alpha=0.8)
            
            ax2.set_xlabel('Threat Configuration')
            ax2.set_ylabel('Final Loss')
            ax2.set_title('📊 Final Loss vs Threat Level')
            ax2.set_xticks(range(len(threat_labels)))
            ax2.set_xticklabels(threat_labels, rotation=45)
            ax2.legend(fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Loss reduction
        reductions = []
        labels = []
        colors_list = []
        
        for fl_data in all_fl_data:
            reduction = fl_data['loss'].iloc[0] - fl_data['loss'].iloc[-1]
            reductions.append(reduction)
            labels.append(fl_data['config_label'].iloc[0])
            colors_list.append(self.colors['baseline'] if fl_data['is_baseline'].iloc[0] else self._get_threat_color(fl_data['total_perc'].iloc[0]))
        
        bars = ax3.bar(range(len(reductions)), reductions, color=colors_list, alpha=0.8)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Loss Reduction')
        ax3.set_title('📉 Loss Improvement Comparison')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Min loss achieved
        min_losses = []
        for fl_data in all_fl_data:
            min_losses.append(fl_data['loss'].min())
        
        bars = ax4.bar(range(len(min_losses)), min_losses, color=colors_list, alpha=0.8)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Minimum Loss Achieved')
        ax4.set_title('🎯 Best Loss Performance')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('FL Loss Analysis - Trust Experiments', fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_f1_evolution_plot(self, pdf, all_fl_data):
        """Crea il plot dell'evoluzione dell'F1-score."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: All F1-score curves
        for fl_data in all_fl_data:
            color = self.colors['baseline'] if fl_data['is_baseline'].iloc[0] else self._get_threat_color(fl_data['total_perc'].iloc[0])
            linewidth = 4 if fl_data['is_baseline'].iloc[0] else 2
            alpha = 1.0 if fl_data['is_baseline'].iloc[0] else 0.8
            
            ax1.plot(fl_data['epoch'], fl_data['avg_f1_score'], color=color, linewidth=linewidth, 
                    alpha=alpha, label=fl_data['config_label'].iloc[0])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average F1-Score')
        ax1.set_title('⭐ F1-Score Evolution Comparison')
        ax1.legend(fontsize=12, loc='lower right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Precision vs Recall comparison
        for fl_data in all_fl_data:
            color = self.colors['baseline'] if fl_data['is_baseline'].iloc[0] else self._get_threat_color(fl_data['total_perc'].iloc[0])
            linewidth = 4 if fl_data['is_baseline'].iloc[0] else 2
            alpha = 1.0 if fl_data['is_baseline'].iloc[0] else 0.8
            
            ax2.plot(fl_data['epoch'], fl_data['avg_precision'], color=color, linewidth=linewidth, 
                    alpha=alpha, linestyle='-', label=f"{fl_data['config_label'].iloc[0]} (P)")
            ax2.plot(fl_data['epoch'], fl_data['avg_recall'], color=color, linewidth=linewidth, 
                    alpha=alpha, linestyle='--', label=f"{fl_data['config_label'].iloc[0]} (R)")
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('🎯 Precision & Recall Evolution')
        ax2.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 1.1)
        
        # Plot 3: Final F1-scores comparison
        baseline_data = [fl for fl in all_fl_data if fl['is_baseline'].iloc[0]]
        threat_data = [fl for fl in all_fl_data if not fl['is_baseline'].iloc[0]]
        
        if baseline_data:
            baseline_f1 = baseline_data[0]['avg_f1_score'].iloc[-1]
            ax3.axhline(baseline_f1, color=self.colors['baseline'], linestyle='--', 
                       linewidth=4, label=f'Baseline: {baseline_f1:.3f}', alpha=0.9)
        
        if threat_data:
            threat_labels = [fl['config_label'].iloc[0] for fl in threat_data]
            threat_f1s = [fl['avg_f1_score'].iloc[-1] for fl in threat_data]
            colors = [self._get_threat_color(fl['total_perc'].iloc[0]) for fl in threat_data]
            
            bars = ax3.bar(range(len(threat_f1s)), threat_f1s, color=colors, alpha=0.8)
            ax3.set_xlabel('Threat Configuration')
            ax3.set_ylabel('Final F1-Score')
            ax3.set_title('📊 Final F1-Score vs Threat Level')
            ax3.set_xticks(range(len(threat_labels)))
            ax3.set_xticklabels(threat_labels, rotation=45)
            ax3.legend(fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
        
        # Plot 4: F1-score degradation from baseline
        if baseline_data:
            baseline_f1 = baseline_data[0]['avg_f1_score'].iloc[-1]
            degradations = []
            labels = []
            colors_list = []
            
            for fl_data in all_fl_data:
                if not fl_data['is_baseline'].iloc[0]:  # Skip baseline itself
                    degradation = baseline_f1 - fl_data['avg_f1_score'].iloc[-1]
                    degradations.append(degradation)
                    labels.append(fl_data['config_label'].iloc[0])
                    colors_list.append(self._get_threat_color(fl_data['total_perc'].iloc[0]))
            
            if degradations:
                bars = ax4.bar(range(len(degradations)), degradations, color=colors_list, alpha=0.8)
                ax4.set_xlabel('Threat Configuration')
                ax4.set_ylabel('F1-Score Loss from Baseline')
                ax4.set_title('📉 F1-Score Degradation')
                ax4.set_xticks(range(len(labels)))
                ax4.set_xticklabels(labels, rotation=45)
                ax4.grid(True, alpha=0.3)
                ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        plt.suptitle('FL F1-Score Analysis - Trust Experiments', fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_final_performance_comparison(self, pdf, summary_df):
        """Crea il confronto delle performance finali."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Separa baseline e threats
        baseline_df = summary_df[summary_df['is_baseline'] == True]
        threat_df = summary_df[summary_df['is_baseline'] == False]
        
        # Plot 1: Multi-metric comparison
        metrics = ['final_accuracy', 'final_avg_precision', 'final_avg_recall', 'final_avg_f1']
        metric_labels = ['Accuracy (%)', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(metrics))
        width = 0.8 / len(summary_df)
        
        for i, (_, row) in enumerate(summary_df.iterrows()):
            color = self.colors['baseline'] if row['is_baseline'] else self._get_threat_color(row['total_perc'])
            alpha = 1.0 if row['is_baseline'] else 0.8
            
            values = [row['final_accuracy'], row['final_avg_precision']*100, 
                     row['final_avg_recall']*100, row['final_avg_f1']*100]
            
            bars = ax1.bar(x + i*width, values, width, label=row['config_label'], 
                          color=color, alpha=alpha)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score (%)')
        ax1.set_title('📊 Final Performance Comparison')
        ax1.set_xticks(x + width * (len(summary_df)-1) / 2)
        ax1.set_xticklabels(metric_labels)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance vs threat level scatter
        if len(threat_df) > 0:
            ax2.scatter(threat_df['total_perc'], threat_df['final_accuracy'], 
                       c=[self._get_threat_color(x) for x in threat_df['total_perc']], 
                       s=200, alpha=0.8, edgecolors='black', linewidth=2)
            
            # Add baseline reference
            if len(baseline_df) > 0:
                baseline_acc = baseline_df['final_accuracy'].iloc[0]
                ax2.axhline(baseline_acc, color=self.colors['baseline'], linestyle='--', 
                           linewidth=4, label=f'Baseline: {baseline_acc:.1f}%', alpha=0.9)
            
            ax2.set_xlabel('Total Threat Percentage (%)')
            ax2.set_ylabel('Final Accuracy (%)')
            ax2.set_title('🎯 Accuracy vs Threat Level')
            ax2.legend(fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning effectiveness (accuracy improvement)
        configs = summary_df['config_label'].values
        improvements = summary_df['accuracy_improvement'].values
        colors = [self.colors['baseline'] if row['is_baseline'] else self._get_threat_color(row['total_perc']) 
                 for _, row in summary_df.iterrows()]
        
        bars = ax3.bar(range(len(configs)), improvements, color=colors, alpha=0.8)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Accuracy Improvement (%)')
        ax3.set_title('📈 Learning Effectiveness')
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels(configs, rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Max accuracy achieved
        max_accuracies = summary_df['max_accuracy'].values
        bars = ax4.bar(range(len(configs)), max_accuracies, color=colors, alpha=0.8)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Maximum Accuracy Achieved (%)')
        ax4.set_title('🏆 Peak Performance')
        ax4.set_xticks(range(len(configs)))
        ax4.set_xticklabels(configs, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Final Performance Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_baseline_vs_threats_analysis(self, pdf, summary_df):
        """Crea l'analisi baseline vs minacce."""
        baseline_df = summary_df[summary_df['is_baseline'] == True]
        threat_df = summary_df[summary_df['is_baseline'] == False]
        
        if len(baseline_df) == 0 or len(threat_df) == 0:
            print("⚠️ Cannot create baseline vs threats analysis - missing data")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        baseline_row = baseline_df.iloc[0]
        
        # Plot 1: Performance degradation
        metrics = ['final_accuracy', 'final_avg_precision', 'final_avg_recall', 'final_avg_f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        baseline_values = [baseline_row['final_accuracy'], baseline_row['final_avg_precision']*100,
                          baseline_row['final_avg_recall']*100, baseline_row['final_avg_f1']*100]
        
        for i, (_, threat_row) in enumerate(threat_df.iterrows()):
            threat_values = [threat_row['final_accuracy'], threat_row['final_avg_precision']*100,
                           threat_row['final_avg_recall']*100, threat_row['final_avg_f1']*100]
            degradations = [b - t for b, t in zip(baseline_values, threat_values)]
            
            color = self._get_threat_color(threat_row['total_perc'])
            ax1.bar(np.arange(len(metrics)) + i*0.15, degradations, width=0.15, 
                   label=threat_row['config_label'], color=color, alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Performance Loss from Baseline (%)')
        ax1.set_title('📉 Performance Degradation vs Baseline')
        ax1.set_xticks(np.arange(len(metrics)) + 0.15)
        ax1.set_xticklabels(metric_labels)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 2: Threat level impact
        threat_levels = threat_df['total_perc'].values
        accuracy_degradations = baseline_row['final_accuracy'] - threat_df['final_accuracy'].values
        
        ax2.scatter(threat_levels, accuracy_degradations, 
                   c=[self._get_threat_color(x) for x in threat_levels],
                   s=200, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Fit trend line
        if len(threat_levels) > 1:
            z = np.polyfit(threat_levels, accuracy_degradations, 1)
            p = np.poly1d(z)
            ax2.plot(threat_levels, p(threat_levels), "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Total Threat Percentage (%)')
        ax2.set_ylabel('Accuracy Loss from Baseline (%)')
        ax2.set_title('🎯 Threat Impact on Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning robustness (improvement comparison)
        baseline_improvement = baseline_row['accuracy_improvement']
        threat_improvements = threat_df['accuracy_improvement'].values
        threat_labels = threat_df['config_label'].values
        
        ax3.axhline(baseline_improvement, color=self.colors['baseline'], linestyle='--', 
                   linewidth=4, label=f'Baseline: {baseline_improvement:.1f}%', alpha=0.9)
        
        colors = [self._get_threat_color(row['total_perc']) for _, row in threat_df.iterrows()]
        bars = ax3.bar(range(len(threat_improvements)), threat_improvements, 
                      color=colors, alpha=0.8)
        
        ax3.set_xlabel('Threat Configuration')
        ax3.set_ylabel('Accuracy Improvement (%)')
        ax3.set_title('📈 Learning Robustness Comparison')
        ax3.set_xticks(range(len(threat_labels)))
        ax3.set_xticklabels(threat_labels, rotation=45)
        ax3.legend(fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Relative performance (% of baseline)
        relative_accuracies = (threat_df['final_accuracy'] / baseline_row['final_accuracy'] * 100).values
        
        bars = ax4.bar(range(len(relative_accuracies)), relative_accuracies, 
                      color=colors, alpha=0.8)
        ax4.axhline(100, color=self.colors['baseline'], linestyle='--', 
                   linewidth=4, label='Baseline (100%)', alpha=0.9)
        
        ax4.set_xlabel('Threat Configuration')
        ax4.set_ylabel('Relative Accuracy (% of Baseline)')
        ax4.set_title('📊 Relative Performance vs Baseline')
        ax4.set_xticks(range(len(threat_labels)))
        ax4.set_xticklabels(threat_labels, rotation=45)
        ax4.legend(fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Baseline vs Threats Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_per_class_comparison(self, pdf, all_fl_data):
        """Crea il confronto per-classe."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prendi solo l'ultimo epoch per ogni esperimento
        final_data = []
        for fl_data in all_fl_data:
            final_row = fl_data.iloc[-1]
            final_data.append(final_row)
        
        # Plot 1: Final precision per class heatmap
        precision_matrix = []
        config_labels = []
        
        for final_row in final_data:
            precision_row = [final_row[f'precision_class_{i}'] for i in range(self.num_classes)]
            precision_matrix.append(precision_row)
            config_labels.append(final_row['config_label'])
        
        precision_matrix = np.array(precision_matrix)
        im1 = ax1.imshow(precision_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Configuration')
        ax1.set_title('🎯 Final Precision per Class')
        ax1.set_xticks(range(self.num_classes))
        ax1.set_xticklabels([f'C{i}' for i in range(self.num_classes)])
        ax1.set_yticks(range(len(config_labels)))
        ax1.set_yticklabels(config_labels)
        plt.colorbar(im1, ax=ax1, label='Precision')
        
        # Plot 2: Final recall per class heatmap
        recall_matrix = []
        for final_row in final_data:
            recall_row = [final_row[f'recall_class_{i}'] for i in range(self.num_classes)]
            recall_matrix.append(recall_row)
        
        recall_matrix = np.array(recall_matrix)
        im2 = ax2.imshow(recall_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Configuration')
        ax2.set_title('🔍 Final Recall per Class')
        ax2.set_xticks(range(self.num_classes))
        ax2.set_xticklabels([f'C{i}' for i in range(self.num_classes)])
        ax2.set_yticks(range(len(config_labels)))
        ax2.set_yticklabels(config_labels)
        plt.colorbar(im2, ax=ax2, label='Recall')
        
        # Plot 3: Average class performance comparison
        baseline_data = [row for row in final_data if row['is_baseline']]
        if baseline_data:
            baseline_row = baseline_data[0]
            baseline_precision = np.mean([baseline_row[f'precision_class_{i}'] for i in range(self.num_classes)])
            baseline_recall = np.mean([baseline_row[f'recall_class_{i}'] for i in range(self.num_classes)])
            
            ax3.axhline(baseline_precision, color=self.colors['baseline'], linestyle='--', 
                       linewidth=3, label=f'Baseline Precision: {baseline_precision:.3f}', alpha=0.9)
            ax3.axhline(baseline_recall, color=self.colors['baseline'], linestyle=':', 
                       linewidth=3, label=f'Baseline Recall: {baseline_recall:.3f}', alpha=0.9)
        
        threat_data = [row for row in final_data if not row['is_baseline']]
        if threat_data:
            threat_precisions = []
            threat_recalls = []
            threat_labels = []
            colors = []
            
            for row in threat_data:
                avg_precision = np.mean([row[f'precision_class_{i}'] for i in range(self.num_classes)])
                avg_recall = np.mean([row[f'recall_class_{i}'] for i in range(self.num_classes)])
                threat_precisions.append(avg_precision)
                threat_recalls.append(avg_recall)
                threat_labels.append(row['config_label'])
                colors.append(self._get_threat_color(row['total_perc']))
            
            x_pos = range(len(threat_labels))
            width = 0.35
            
            bars1 = ax3.bar([x - width/2 for x in x_pos], threat_precisions, width, 
                           label='Precision', alpha=0.8, color=colors)
            bars2 = ax3.bar([x + width/2 for x in x_pos], threat_recalls, width, 
                           label='Recall', alpha=0.6, color=colors)
            
            ax3.set_xlabel('Threat Configuration')
            ax3.set_ylabel('Average Score')
            ax3.set_title('📊 Average Class Performance')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(threat_labels, rotation=45)
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Class performance variance
        class_variances = []
        for final_row in final_data:
            precisions = [final_row[f'precision_class_{i}'] for i in range(self.num_classes)]
            variance = np.var(precisions)
            class_variances.append(variance)
        
        colors_list = []
        for final_row in final_data:
            if final_row['is_baseline']:
                colors_list.append(self.colors['baseline'])
            else:
                colors_list.append(self._get_threat_color(final_row['total_perc']))
        
        bars = ax4.bar(range(len(class_variances)), class_variances, color=colors_list, alpha=0.8)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Precision Variance')
        ax4.set_title('📊 Class Performance Consistency')
        ax4.set_xticks(range(len(config_labels)))
        ax4.set_xticklabels(config_labels, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Per-Class Performance Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_fl_summary_table(self, pdf, summary_df):
        """Crea la tabella riassuntiva FL."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Ordina per total_perc (baseline first)
        summary_df_sorted = summary_df.sort_values(['total_perc', 'malicious_perc'])
        
        # Prepara i dati per la tabella
        table_data = []
        for _, row in summary_df_sorted.iterrows():
            status = "BASELINE" if row['is_baseline'] else "THREAT"
            table_data.append([
                f"{int(row['malicious_perc'])}%",
                f"{int(row['corrupted_perc'])}%", 
                f"{int(row['total_perc'])}%",
                f"{row['final_accuracy']:.1f}%",
                f"{row['final_loss']:.3f}",
                f"{row['final_avg_precision']:.3f}",
                f"{row['final_avg_recall']:.3f}",
                f"{row['final_avg_f1']:.3f}",
                f"{row['accuracy_improvement']:+.1f}%",
                f"{row['max_accuracy']:.1f}%",
                f"{int(row['epochs'])}",
                status
            ])
        
        # Crea la tabella
        table = ax.table(cellText=table_data,
                        colLabels=['Mal%', 'Cor%', 'Tot%', 'Final Acc', 'Final Loss', 
                                  'Precision', 'Recall', 'F1-Score', 'Acc Improve', 
                                  'Max Acc', 'Epochs', 'Type'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Stilizza la tabella
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Header styling
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Data row styling
        for i in range(1, len(table_data) + 1):
            for j in range(len(table_data[0])):
                row_data = summary_df_sorted.iloc[i-1]
                if row_data['is_baseline']:
                    table[(i, j)].set_facecolor('#E3F2FD')  # Light blue for baseline
                else:
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F5F5F5')
                    else:
                        table[(i, j)].set_facecolor('white')
        
        plt.title('FL Trust Experiments - Complete Performance Summary', 
                 fontsize=20, fontweight='bold', pad=20)
        
        # Add outlier info to table if any were detected
        if self.outliers_detected:
            outlier_note = f"\n\nNote: {len(self.outliers_detected)} outlier experiment(s) were detected and removed from analysis.\n"
            outlier_note += "See Outlier Analysis page for details."
            ax.text(0.5, -0.15, outlier_note, transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, style='italic', color='red',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _get_threat_color(self, total_perc):
        """Ottieni il colore basato sul livello di minaccia."""
        if total_perc == 0:
            return self.colors['baseline']
        elif total_perc <= 15:
            return self.colors['low_threat']
        elif total_perc <= 30:
            return self.colors['medium_threat']
        else:
            return self.colors['high_threat']
    
    def run_fl_comparison(self):
        """Esegui il confronto completo FL."""
        print("Starting DAGTrustFL FL Trust comparison...")
        print("="*70)
        
        # Scansiona le cartelle degli esperimenti
        if not self.scan_experiment_folders():
            return False
        
        # Crea il report di confronto
        pdf_filename = self.create_fl_comparison_report()
        
        if pdf_filename:
            print("\n" + "="*70)
            print("✅ FL TRUST COMPARISON COMPLETED!")
            print(f"📑 PDF Report: {pdf_filename}")
            print(f"📁 Output Directory: {self.output_dir}")
            print("="*70)
            return True
        else:
            print("❌ Failed to create FL comparison report")
            return False


def main():
    """Funzione principale."""
    print("🔬 FL Trust Results Comparator with Outlier Detection")
    print("="*60)
    
    # Optional: Allow custom outlier threshold
    outlier_threshold = 2.0  # Can be made configurable
    print(f"📊 Using Z-score threshold for outlier detection: {outlier_threshold}")
    
    comparator = FLTrustComparator(outlier_threshold=outlier_threshold)
    success = comparator.run_fl_comparison()
    
    if success:
        print("\n🎉 FL comparison analysis completed successfully!")
        if comparator.outliers_detected:
            print(f"⚠️  {len(comparator.outliers_detected)} outlier(s) were detected and removed")
    else:
        print("\n❌ FL comparison analysis failed!")
    
    return success


if __name__ == "__main__":
    main()
