#!/usr/bin/env python3
"""
Percentages Results Comparator for DAGTrustFL
=============================================

This script compares and analyzes results from different malicious and corrupted percentage experiments.
It creates comprehensive plots and analysis reports for percentage combination testing.

Percentage Combinations:
- Tests all valid combinations of malicious (10-40%) and corrupted (0-30%) percentages
- Total percentage (malicious + corrupted) ranges from 10% to 40%
- Analyzes how different threat levels affect system performance

Author: DAGTrustFL Analysis Suite
Date: 2025-08-31
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Set matplotlib font to avoid font warnings
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

class PercentagesResultsComparator:
    """Class for comparing results from different percentage experiments."""
    
    def __init__(self, results_base_path="FL/results_percentages"):
        """
        Initialize the comparator.
        
        Args:
            results_base_path (str): Base path where percentage results are stored
        """
        self.results_base_path = Path(results_base_path)
        self.output_dir = "percentages_comparison_plots"
        self.experiments = {}
        
        # Create directory for plots
        os.makedirs(self.output_dir, exist_ok=True)
        
    def scan_experiment_folders(self):
        """Scan for experiment folders and categorize them."""
        print("🔍 Scanning for percentage experiment folders...")
        
        if not self.results_base_path.exists():
            print(f"❌ Results path not found: {self.results_base_path}")
            return False
        
        # Pattern: mal_X_cor_Y (e.g., mal_20_cor_10 for 20% malicious, 10% corrupted)
        pattern = r'mal_(\d+)_cor_(\d+)'
        
        for folder in self.results_base_path.iterdir():
            if folder.is_dir():
                match = re.match(pattern, folder.name)
                if match:
                    # Extract percentages
                    malicious_perc = int(match.group(1))
                    corrupted_perc = int(match.group(2))
                    
                    # Look for the CSV file in this folder
                    csv_file = folder / "dag_trust_analysis.csv"
                    if csv_file.exists():
                        self.experiments[(malicious_perc, corrupted_perc)] = {
                            'folder': folder,
                            'csv_path': csv_file,
                            'malicious_perc': malicious_perc,
                            'corrupted_perc': corrupted_perc,
                            'total_perc': malicious_perc + corrupted_perc
                        }
                        print(f"   ✓ Found {malicious_perc}% mal, {corrupted_perc}% cor: {folder.name}")
                    else:
                        print(f"   ⚠️ No CSV file in {folder.name}")
        
        if not self.experiments:
            print("❌ No valid experiment folders found!")
            return False
        
        print(f"\n📊 Found {len(self.experiments)} percentage experiments:")
        for (mal, cor), exp in sorted(self.experiments.items()):
            total = exp['total_perc']
            print(f"   • {mal}% malicious, {cor}% corrupted (total: {total}%)")
        
        return True
    
    def load_experiment_data(self, csv_path):
        """Load and process data from a single experiment CSV."""
        try:
            df = pd.read_csv(csv_path)
            
            # Count total malicious and corrupted peers from peer columns
            peer_columns = [col for col in df.columns if col.startswith('Peer_') and col.endswith('_Is_Malicious')]
            num_peers = len(peer_columns)
            
            total_malicious_peers = sum(1 for i in range(num_peers) 
                                      if df[f'Peer_{i}_Is_Malicious'].iloc[0] == 1)
            total_corrupted_peers = sum(1 for i in range(num_peers) 
                                      if df[f'Peer_{i}_Is_Corrupted'].iloc[0] == 1)
            total_good_peers = num_peers - total_malicious_peers - total_corrupted_peers
            print("TOTAL MALICIOUS:", total_malicious_peers, "TOTAL CORRUPTED", total_corrupted_peers)

            # Sum across ALL rounds, not just final round
            # Sum all good nodes removed across all rounds
            total_good_removed = df['Good_Nodes_Removed_This_Round'].sum()
            # Sum all malicious nodes confirmed across all rounds  
            total_malicious_confirmed = df['Malicious_Nodes_Confirmed_This_Round'].sum()
            print(total_malicious_confirmed)

            # Sum all corrupted nodes confirmed across all rounds
            total_corrupted_confirmed = df['Corrupted_Nodes_Confirmed_This_Round'].sum()
            
            # Ensure non-negative values
            total_good_removed = max(0, total_good_removed)
            total_malicious_confirmed = max(0, total_malicious_confirmed)
            total_corrupted_confirmed = max(0, total_corrupted_confirmed)
            
            # Calculate total rounds
            total_rounds = len(df)
            
            # GET ACTUAL COUNTS FROM FILES
            # Get the experiment folder from the CSV path
            experiment_folder = csv_path.parent
            actual_counts = self.count_actual_nodes_sent(experiment_folder)
            
            if actual_counts is None:
                print(f"⚠️ Could not get actual counts for {experiment_folder}, using estimates")
                # Fallback estimates based on random selection probability
                total_rounds = len(df)
                estimated_good_sent = total_good_peers * 9 * total_rounds / num_peers
                estimated_malicious_sent = total_malicious_peers * 9 * total_rounds / num_peers
                estimated_corrupted_sent = total_corrupted_peers * 9 * total_rounds / num_peers
                actual_total_good_sent = estimated_good_sent
                actual_total_malicious_sent = estimated_malicious_sent
                actual_total_corrupted_sent = estimated_corrupted_sent
            else:
                actual_total_good_sent = actual_counts['total_good_sent']
                actual_total_malicious_sent = actual_counts['total_malicious_sent']
                actual_total_corrupted_sent = actual_counts['total_corrupted_sent']
            
            # FPR: Good nodes removed / Total good nodes actually sent
            fpr = (total_good_removed / actual_total_good_sent * 100) if actual_total_good_sent > 0 else 0
            
            # FNR for malicious: Malicious nodes confirmed / Total malicious sent
            # (False Negative = we failed to detect them, so they were confirmed)
            fnr_malicious = (total_malicious_confirmed / actual_total_malicious_sent * 100) if actual_total_malicious_sent > 0 else 0
            fnr_malicious = min(fnr_malicious, 100)  # Cap at 100%
            
            # TPR (Detection Rate) for malicious: Malicious nodes NOT confirmed / Total malicious sent
            # (True Positive = we correctly detected and rejected them)
            malicious_not_confirmed = actual_total_malicious_sent - total_malicious_confirmed
            tpr_malicious = (malicious_not_confirmed / actual_total_malicious_sent * 100) if actual_total_malicious_sent > 0 else 0
            tpr_malicious = max(0, tpr_malicious)  # Ensure non-negative
            
            # Same calculations for corrupted nodes
            corrupted_not_confirmed = actual_total_corrupted_sent - total_corrupted_confirmed
            fnr_corrupted = (total_corrupted_confirmed / actual_total_corrupted_sent * 100) if actual_total_corrupted_sent > 0 else 0
            fnr_corrupted = min(fnr_corrupted, 100)  # Cap at 100%
            tpr_corrupted = (corrupted_not_confirmed / actual_total_corrupted_sent * 100) if actual_total_corrupted_sent > 0 else 0
            tpr_corrupted = max(0, tpr_corrupted)  # Ensure non-negative
            
            # Accuracy: Correctly handled nodes / Total nodes
            total_nodes_sent = actual_total_good_sent + actual_total_malicious_sent + actual_total_corrupted_sent
            # Correctly handled = good nodes confirmed + malicious/corrupted nodes rejected
            correctly_handled = (actual_total_good_sent - total_good_removed) + malicious_not_confirmed + corrupted_not_confirmed
            accuracy = (correctly_handled / total_nodes_sent * 100) if total_nodes_sent > 0 else 0
            
            print(f"Debug {csv_path.name}: Good removed: {total_good_removed}, Malicious confirmed: {total_malicious_confirmed}, Rounds: {total_rounds}")
            print(f"  Actual sent - Good: {actual_total_good_sent:.1f}, Malicious: {actual_total_malicious_sent:.1f}")
            print(f"  Metrics - FPR: {fpr:.1f}%, TPR: {tpr_malicious:.1f}%, FNR: {fnr_malicious:.1f}%")
            
            return {
                'num_peers': num_peers,
                'total_malicious_peers': total_malicious_peers,
                'total_corrupted_peers': total_corrupted_peers,
                'total_good_peers': total_good_peers,
                'total_rounds': total_rounds,
                'actual_total_good_sent': actual_total_good_sent,
                'actual_total_malicious_sent': actual_total_malicious_sent,
                'actual_total_corrupted_sent': actual_total_corrupted_sent,
                'total_good_removed': total_good_removed,
                'total_malicious_confirmed': total_malicious_confirmed,
                'total_corrupted_confirmed': total_corrupted_confirmed,
                'fpr': fpr,
                'tpr_malicious': tpr_malicious,
                'fnr_malicious': fnr_malicious,
                'tpr_corrupted': tpr_corrupted,
                'fnr_corrupted': fnr_corrupted,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"❌ Error loading {csv_path}: {e}")
            return None
    
    def count_actual_nodes_sent(self, experiment_folder):
        """
        Count the actual number of malicious, corrupted, and good nodes sent 
        by reading the workers_selected.csv file.
        
        Args:
            experiment_folder (Path): Path to the experiment folder
            
        Returns:
            dict: Actual counts of nodes sent
        """
        try:
            # Read the workers selected file
            workers_file = experiment_folder / "3000_workers_selected.csv"
            dag_analysis_file = experiment_folder / "dag_trust_analysis.csv"
            
            if not workers_file.exists():
                print(f"⚠️ Workers file not found: {workers_file}")
                return None
                
            if not dag_analysis_file.exists():
                print(f"⚠️ DAG analysis file not found: {dag_analysis_file}")
                return None
            
            # Read DAG analysis to determine which peers are malicious/corrupted
            dag_df = pd.read_csv(dag_analysis_file)
            
            # Get peer status from any row (they should be the same across rounds)
            peer_status = {}
            first_row = dag_df.iloc[0]
            
            # Find all peer columns
            peer_columns = [col for col in dag_df.columns if col.startswith('Peer_') and '_Is_' in col]
            
            for col in peer_columns:
                if '_Is_Malicious' in col:
                    peer_id = int(col.replace('Peer_', '').replace('_Is_Malicious', ''))
                    if peer_id not in peer_status:
                        peer_status[peer_id] = {'malicious': False, 'corrupted': False}
                    peer_status[peer_id]['malicious'] = (first_row[col] == 1)
                elif '_Is_Corrupted' in col:
                    peer_id = int(col.replace('Peer_', '').replace('_Is_Corrupted', ''))
                    if peer_id not in peer_status:
                        peer_status[peer_id] = {'malicious': False, 'corrupted': False}
                    peer_status[peer_id]['corrupted'] = (first_row[col] == 1)
            
            # Categorize peers
            malicious_peers = {pid for pid, status in peer_status.items() if status['malicious']}
            corrupted_peers = {pid for pid, status in peer_status.items() if status['corrupted']}
            good_peers = {pid for pid, status in peer_status.items() 
                         if not status['malicious'] and not status['corrupted']}
            
            print(f"   🔍 Peer categories in {experiment_folder.name}:")
            print(f"      Malicious: {malicious_peers}")
            print(f"      Corrupted: {corrupted_peers}")
            print(f"      Good: {good_peers}")
            
            # Read workers selected file and count actual occurrences
            workers_df = pd.read_csv(workers_file)
            
            total_malicious_sent = 0
            total_corrupted_sent = 0
            total_good_sent = 0
            
            # Count occurrences in each round
            for _, row in workers_df.iterrows():
                # Get selected workers for this round (each column contains a peer ID)
                selected_workers = []
                for col_idx in range(len(row)):
                    worker_id = row.iloc[col_idx]
                    if pd.notna(worker_id):  # Check if not NaN
                        selected_workers.append(int(worker_id))
                
                # Count by category
                round_malicious = sum(1 for w in selected_workers if w in malicious_peers)
                round_corrupted = sum(1 for w in selected_workers if w in corrupted_peers)
                round_good = sum(1 for w in selected_workers if w in good_peers)
                
                total_malicious_sent += round_malicious
                total_corrupted_sent += round_corrupted
                total_good_sent += round_good
            
            print(f"   📊 Actual counts for {experiment_folder.name}:")
            print(f"      Total malicious sent: {total_malicious_sent}")
            print(f"      Total corrupted sent: {total_corrupted_sent}")
            print(f"      Total good sent: {total_good_sent}")
            
            return {
                'total_malicious_sent': total_malicious_sent,
                'total_corrupted_sent': total_corrupted_sent,
                'total_good_sent': total_good_sent,
                'malicious_peers': malicious_peers,
                'corrupted_peers': corrupted_peers,
                'good_peers': good_peers
            }
            
        except Exception as e:
            print(f"❌ Error counting actual nodes in {experiment_folder}: {e}")
            return None
    
    def create_comparison_plots(self):
        """Create comparison plots for all percentage experiments."""
        pdf_filename = f"{self.output_dir}/percentages_comparison_report.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            print("📑 Creating percentages comparison PDF...")
            
            # Load data for all percentage combinations
            percentage_data = []
            
            for (mal_perc, cor_perc), exp_info in sorted(self.experiments.items()):
                data = self.load_experiment_data(exp_info['csv_path'])
                if data:
                    data['malicious_perc'] = mal_perc
                    data['corrupted_perc'] = cor_perc
                    data['total_perc'] = mal_perc + cor_perc
                    data['config_label'] = f"{mal_perc}%M, {cor_perc}%C"
                    percentage_data.append(data)
            
            if not percentage_data:
                print("   ⚠️ No valid data found")
                return pdf_filename
            
            df_results = pd.DataFrame(percentage_data)
            
            # Create multiple comparison plots
            self._create_total_percentage_plots(pdf, df_results)
            self._create_malicious_vs_corrupted_plots(pdf, df_results)
            self._create_heatmap_plots(pdf, df_results)
            self._create_ranking_page(pdf, df_results)
            self._create_summary_page(pdf, df_results)
        
        print(f"📑 Percentages comparison PDF saved: {pdf_filename}")
        
        # Save data to CSV for further analysis
        csv_filename = self._save_data_to_csv(df_results)
        print(f"💾 Percentages comparison data saved: {csv_filename}")
        
        return pdf_filename
    
    def _create_total_percentage_plots(self, pdf, df_results):
        """Create plots organized by total percentage."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by total percentage
        total_percs = sorted(df_results['total_perc'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(total_percs)))
        
        # Plot 1: FPR vs Total Percentage
        for i, total_perc in enumerate(total_percs):
            subset = df_results[df_results['total_perc'] == total_perc]
            x_vals = [total_perc] * len(subset)
            ax1.scatter(x_vals, subset['fpr'], color=colors[i], s=100, alpha=0.7, 
                       label=f'{total_perc}%')
        
        ax1.set_xlabel('Total Threat Percentage (%)')
        ax1.set_ylabel('False Positive Rate (%)')
        ax1.set_title('FPR vs Total Threat Level')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: TPR vs Total Percentage
        for i, total_perc in enumerate(total_percs):
            subset = df_results[df_results['total_perc'] == total_perc]
            x_vals = [total_perc] * len(subset)
            ax2.scatter(x_vals, subset['tpr_malicious'], color=colors[i], s=100, alpha=0.7)
        
        ax2.set_xlabel('Total Threat Percentage (%)')
        ax2.set_ylabel('True Positive Rate (%)')
        ax2.set_title('TPR vs Total Threat Level')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FNR vs Total Percentage
        for i, total_perc in enumerate(total_percs):
            subset = df_results[df_results['total_perc'] == total_perc]
            x_vals = [total_perc] * len(subset)
            ax3.scatter(x_vals, subset['fnr_malicious'], color=colors[i], s=100, alpha=0.7)
        
        ax3.set_xlabel('Total Threat Percentage (%)')
        ax3.set_ylabel('False Negative Rate (%)')
        ax3.set_title('FNR vs Total Threat Level')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy vs Total Percentage
        for i, total_perc in enumerate(total_percs):
            subset = df_results[df_results['total_perc'] == total_perc]
            x_vals = [total_perc] * len(subset)
            ax4.scatter(x_vals, subset['accuracy'], color=colors[i], s=100, alpha=0.7)
        
        ax4.set_xlabel('Total Threat Percentage (%)')
        ax4.set_ylabel('System Accuracy (%)')
        ax4.set_title('Accuracy vs Total Threat Level')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Metrics vs Total Threat Level', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_malicious_vs_corrupted_plots(self, pdf, df_results):
        """Create plots comparing malicious vs corrupted percentages."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scatter plots with malicious on x-axis, corrupted as color
        scatter1 = ax1.scatter(df_results['malicious_perc'], df_results['fpr'], 
                              c=df_results['corrupted_perc'], s=100, alpha=0.7, 
                              cmap='plasma')
        ax1.set_xlabel('Malicious Percentage (%)')
        ax1.set_ylabel('False Positive Rate (%)')
        ax1.set_title('FPR: Malicious vs Corrupted Impact')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Corrupted %')
        
        scatter2 = ax2.scatter(df_results['malicious_perc'], df_results['tpr_malicious'], 
                              c=df_results['corrupted_perc'], s=100, alpha=0.7, 
                              cmap='plasma')
        ax2.set_xlabel('Malicious Percentage (%)')
        ax2.set_ylabel('True Positive Rate (%)')
        ax2.set_title('TPR: Malicious vs Corrupted Impact')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Corrupted %')
        
        scatter3 = ax3.scatter(df_results['malicious_perc'], df_results['fnr_malicious'], 
                              c=df_results['corrupted_perc'], s=100, alpha=0.7, 
                              cmap='plasma')
        ax3.set_xlabel('Malicious Percentage (%)')
        ax3.set_ylabel('False Negative Rate (%)')
        ax3.set_title('FNR: Malicious vs Corrupted Impact')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Corrupted %')
        
        scatter4 = ax4.scatter(df_results['malicious_perc'], df_results['accuracy'], 
                              c=df_results['corrupted_perc'], s=100, alpha=0.7, 
                              cmap='plasma')
        ax4.set_xlabel('Malicious Percentage (%)')
        ax4.set_ylabel('System Accuracy (%)')
        ax4.set_title('Accuracy: Malicious vs Corrupted Impact')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Corrupted %')
        
        plt.suptitle('Malicious vs Corrupted Percentage Impact', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_heatmap_plots(self, pdf, df_results):
        """Create heatmap plots for metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for heatmaps
        mal_percs = sorted(df_results['malicious_perc'].unique())
        cor_percs = sorted(df_results['corrupted_perc'].unique())
        
        def create_heatmap_data(metric):
            heatmap_data = np.full((len(cor_percs), len(mal_percs)), np.nan)
            for _, row in df_results.iterrows():
                mal_idx = mal_percs.index(row['malicious_perc'])
                cor_idx = cor_percs.index(row['corrupted_perc'])
                heatmap_data[cor_idx, mal_idx] = row[metric]
            return heatmap_data
        
        # FPR Heatmap
        fpr_data = create_heatmap_data('fpr')
        im1 = ax1.imshow(fpr_data, cmap='Reds', aspect='auto')
        ax1.set_xticks(range(len(mal_percs)))
        ax1.set_xticklabels(mal_percs)
        ax1.set_yticks(range(len(cor_percs)))
        ax1.set_yticklabels(cor_percs)
        ax1.set_xlabel('Malicious Percentage (%)')
        ax1.set_ylabel('Corrupted Percentage (%)')
        ax1.set_title('FPR Heatmap')
        plt.colorbar(im1, ax=ax1)
        
        # TPR Heatmap
        tpr_data = create_heatmap_data('tpr_malicious')
        im2 = ax2.imshow(tpr_data, cmap='Greens', aspect='auto')
        ax2.set_xticks(range(len(mal_percs)))
        ax2.set_xticklabels(mal_percs)
        ax2.set_yticks(range(len(cor_percs)))
        ax2.set_yticklabels(cor_percs)
        ax2.set_xlabel('Malicious Percentage (%)')
        ax2.set_ylabel('Corrupted Percentage (%)')
        ax2.set_title('TPR Heatmap')
        plt.colorbar(im2, ax=ax2)
        
        # FNR Heatmap
        fnr_data = create_heatmap_data('fnr_malicious')
        im3 = ax3.imshow(fnr_data, cmap='Oranges', aspect='auto')
        ax3.set_xticks(range(len(mal_percs)))
        ax3.set_xticklabels(mal_percs)
        ax3.set_yticks(range(len(cor_percs)))
        ax3.set_yticklabels(cor_percs)
        ax3.set_xlabel('Malicious Percentage (%)')
        ax3.set_ylabel('Corrupted Percentage (%)')
        ax3.set_title('FNR Heatmap')
        plt.colorbar(im3, ax=ax3)
        
        # Accuracy Heatmap
        acc_data = create_heatmap_data('accuracy')
        im4 = ax4.imshow(acc_data, cmap='Blues', aspect='auto')
        ax4.set_xticks(range(len(mal_percs)))
        ax4.set_xticklabels(mal_percs)
        ax4.set_yticks(range(len(cor_percs)))
        ax4.set_yticklabels(cor_percs)
        ax4.set_xlabel('Malicious Percentage (%)')
        ax4.set_ylabel('Corrupted Percentage (%)')
        ax4.set_title('Accuracy Heatmap')
        plt.colorbar(im4, ax=ax4)
        
        plt.suptitle('Performance Heatmaps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_ranking_page(self, pdf, df_results):
        """Create percentage configuration ranking page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        ranking_text = "PERCENTAGE CONFIGURATION RANKING\n"
        ranking_text += "="*40 + "\n\n"
        
        # Calculate composite score for each configuration
        config_rankings = []
        for _, row in df_results.iterrows():
            fpr = row['fpr']
            tpr = row['tpr_malicious']
            fnr = row['fnr_malicious']
            acc = row['accuracy']
            
            # Composite score: Higher TPR and Accuracy are better, lower FPR and FNR are better
            score = tpr + acc - fpr - fnr
            config_rankings.append((row['config_label'], score, fpr, tpr, fnr, acc, 
                                  row['malicious_perc'], row['corrupted_perc'], row['total_perc']))
        
        config_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (config, score, fpr, tpr, fnr, acc, mal, cor, total) in enumerate(config_rankings, 1):
            ranking_text += f"{rank}. {config} (Total: {total}%)\n"
            ranking_text += f"   FPR: {fpr:.1f}%, TPR: {tpr:.1f}%, FNR: {fnr:.1f}%, Acc: {acc:.1f}%\n"
            ranking_text += f"   Score: {score:.1f}\n\n"
        
        ranking_text += "\nPERCENTAGE TESTING:\n"
        ranking_text += "This analysis tests different combinations of malicious\n"
        ranking_text += "and corrupted peer percentages to understand how\n"
        ranking_text += "threat levels affect system performance.\n"
        ranking_text += "\nCONFIGURATIONS TESTED:\n"
        ranking_text += f"Total configurations: {len(config_rankings)}\n"
        ranking_text += "Total threat levels: 10% to 40%"
        
        ax.text(0.05, 0.95, ranking_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.title('Configuration Effectiveness Ranking', fontsize=18, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_summary_page(self, pdf, df_results):
        """Create summary statistics page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        summary_text = f"""
PERCENTAGE COMPARISON ANALYSIS - SUMMARY
========================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results Path: {self.results_base_path}

CONFIGURATIONS ANALYZED:
• Total configurations: {len(df_results)}
• Malicious percentages: {sorted(df_results['malicious_perc'].unique())}
• Corrupted percentages: {sorted(df_results['corrupted_perc'].unique())}
• Total threat levels: {sorted(df_results['total_perc'].unique())}

DETAILED RESULTS:
"""
        
        for _, row in df_results.sort_values(['malicious_perc', 'corrupted_perc']).iterrows():
            mal = int(row['malicious_perc'])
            cor = int(row['corrupted_perc'])
            total = int(row['total_perc'])
            fpr = row['fpr']
            tpr = row['tpr_malicious']
            fnr = row['fnr_malicious']
            acc = row['accuracy']
            
            summary_text += f"\\n• {mal}%M, {cor}%C (Total: {total}%):\\n"
            summary_text += f"  FPR={fpr:.1f}%, TPR={tpr:.1f}%, FNR={fnr:.1f}%, Acc={acc:.1f}%\\n"

        summary_text += f"""

METRICS EXPLANATION:
• FPR (False Positive Rate): Good nodes incorrectly removed / Total good nodes sent
• TPR (True Positive Rate): Malicious nodes correctly detected / Total malicious nodes sent
• FNR (False Negative Rate): Malicious nodes incorrectly confirmed / Total malicious nodes sent  
• Accuracy: Overall system accuracy (correctly handled nodes / total nodes)

ANALYSIS INSIGHTS:
• Total threat level ranges from 10% to 40%
• Different malicious/corrupted ratios at same total threat level
• System performance under varying adversarial conditions
• Optimal balance between security and false positives

PERCENTAGE PARAMETERS:
The malicious_percentage and corrupted_percentage parameters control
the proportion of adversarial peers in the federated learning system:
• Malicious peers: Actively try to degrade model performance
• Corrupted peers: Have degraded data quality but not malicious intent
• Total threat = malicious% + corrupted%
• Higher percentages create more challenging environments
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        
        plt.title('Summary Statistics', fontsize=18, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _save_data_to_csv(self, df_results):
        """Save all comparison data to CSV files for further analysis."""
        
        # Create CSV data directory
        csv_dir = Path(self.output_dir) / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        
        # Save percentages comparison data
        percentages_filename = csv_dir / "percentages_comparison_data.csv"
        
        # Select relevant columns for CSV export
        export_columns = [
            'malicious_perc', 'corrupted_perc', 'total_perc', 'config_label',
            'fpr', 'tpr_malicious', 'fnr_malicious', 'tpr_corrupted', 'fnr_corrupted', 'accuracy',
            'total_malicious_peers', 'total_corrupted_peers', 'total_good_peers', 'total_rounds',
            'actual_total_malicious_sent', 'actual_total_corrupted_sent', 'actual_total_good_sent',
            'total_malicious_confirmed', 'total_corrupted_confirmed', 'total_good_removed'
        ]
        
        df_export = df_results[export_columns].copy()
        df_export.to_csv(percentages_filename, index=False)
        print(f"   💾 Saved data for {len(df_export)} percentage configurations")
        
        # Save summary statistics CSV
        summary_filename = csv_dir / "percentages_summary_statistics.csv"
        self._save_summary_statistics_csv(summary_filename, df_results)
        
        return percentages_filename
    
    def _save_summary_statistics_csv(self, filename, df_results):
        """Save summary statistics to CSV."""
        
        # Calculate summary statistics and rankings
        summary_rows = []
        for _, row in df_results.iterrows():
            # Calculate composite score
            score = row['tpr_malicious'] + row['accuracy'] - row['fpr'] - row['fnr_malicious']
            
            summary_rows.append({
                'malicious_perc': row['malicious_perc'],
                'corrupted_perc': row['corrupted_perc'],
                'total_perc': row['total_perc'],
                'config_label': row['config_label'],
                'composite_score': score,
                'fpr': row['fpr'],
                'tpr_malicious': row['tpr_malicious'],
                'fnr_malicious': row['fnr_malicious'],
                'accuracy': row['accuracy'],
                'ranking': 0  # Will be filled after sorting
            })
        
        # Sort by composite score and assign rankings
        summary_rows.sort(key=lambda x: x['composite_score'], reverse=True)
        for i, row in enumerate(summary_rows):
            row['ranking'] = i + 1
        
        # Save to CSV
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(filename, index=False)
        print(f"   📊 Saved summary statistics for {len(summary_rows)} configurations")

def main():
    """Main function to run the percentages comparison analysis."""
    print("🚀 Starting Percentages Results Comparison Analysis")
    print("=" * 60)
    
    try:
        # Initialize the comparator
        comparator = PercentagesResultsComparator()
        
        # Scan for experiment folders
        if not comparator.scan_experiment_folders():
            return 1
        
        # Create comparison plots and analysis
        pdf_file = comparator.create_comparison_plots()
        
        print("\n" + "=" * 60)
        print("✅ PERCENTAGES COMPARISON ANALYSIS COMPLETED!")
        print(f"📑 PDF Report: {pdf_file}")
        print(f"📁 Output Directory: {comparator.output_dir}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
