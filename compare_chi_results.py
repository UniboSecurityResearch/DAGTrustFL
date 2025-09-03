#!/usr/bin/env python3
"""
Chi Results Comparison Script for DAGTrustFL
============================================

This script analyzes and compares results from experiments with different chi values
and different malicious/corrupted percentages, creating comparative graphs for:

1. False Positive Rate (FPR): Good nodes removed / Total good nodes sent
2. False Negative Rate (FNR): Malicious nodes confirmed / Total malicious nodes sent  
3. Accuracy: Malicious nodes NOT confirmed / Total malicious nodes sent

Author: Giacomo Gori and GitHub Copilot Assistant
Date: August 30, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
from pathlib import Path
import glob
import re
from datetime import datetime

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChiResultsComparator:
    """Class for comparing results from different chi experiments."""
    
    def __init__(self, results_base_path="FL/results"):
        """
        Initialize the comparator.
        
        Args:
            results_base_path (str): Base path where chi results are stored
        """
        self.results_base_path = Path(results_base_path)
        self.output_dir = "chi_comparison_plots"
        self.experiments = {}
        
        # Create directory for plots
        os.makedirs(self.output_dir, exist_ok=True)
        
    def scan_experiment_folders(self):
        """Scan for experiment folders and categorize them."""
        print("🔍 Scanning for experiment folders...")
        
        if not self.results_base_path.exists():
            print(f"❌ Results path {self.results_base_path} does not exist!")
            return False
        
        # Pattern: chi_X_mal_Y_cor_Z
        pattern = r'chi_(\d+)_mal_(\d+)_cor_(\d+)'
        
        for folder in self.results_base_path.iterdir():
            if folder.is_dir():
                match = re.match(pattern, folder.name)
                if match:
                    chi = int(match.group(1))
                    mal_pct = int(match.group(2))
                    cor_pct = int(match.group(3))
                    
                    # Check if dag_trust_analysis.csv exists
                    csv_path = folder / 'dag_trust_analysis.csv'
                    if csv_path.exists():
                        config_key = f"mal_{mal_pct}_cor_{cor_pct}"
                        if config_key not in self.experiments:
                            self.experiments[config_key] = {}
                        
                        self.experiments[config_key][chi] = {
                            'folder': folder,
                            'csv_path': csv_path,
                            'mal_pct': mal_pct,
                            'cor_pct': cor_pct
                        }
                        
                        print(f"✅ Found: chi={chi}, mal={mal_pct}%, cor={cor_pct}% -> {csv_path}")
                    else:
                        print(f"⚠️ Missing CSV in {folder.name}")
        
        if not self.experiments:
            print("❌ No valid experiment folders found!")
            return False
        
        print(f"\n📊 Found {len(self.experiments)} different percentage configurations:")
        for config_key, chi_data in self.experiments.items():
            chi_values = sorted(chi_data.keys())
            print(f"   • {config_key}: chi values {chi_values}")
        
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

            # CORRECTED: Sum across ALL rounds, not just final round
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
                print(f"⚠️ Could not get actual counts for {csv_path}, using estimates")
                # Fallback to estimation if files are missing
                nodes_per_round = num_peers - 1  # K-1 nodes selected per round
                expected_good_per_round = (nodes_per_round * total_good_peers) / num_peers
                expected_malicious_per_round = (nodes_per_round * total_malicious_peers) / num_peers
                expected_corrupted_per_round = (nodes_per_round * total_corrupted_peers) / num_peers
                
                actual_total_good_sent = expected_good_per_round * total_rounds
                actual_total_malicious_sent = expected_malicious_per_round * total_rounds
                actual_total_corrupted_sent = expected_corrupted_per_round * total_rounds
            else:
                # Use the actual counts from the files
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
        """Create comparison plots for all experiments."""
        pdf_filename = f"{self.output_dir}/chi_comparison_report.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            print("📑 Creating chi comparison PDF...")
            
            # Process each configuration separately
            for config_key, chi_data in self.experiments.items():
                mal_pct = list(chi_data.values())[0]['mal_pct']
                cor_pct = list(chi_data.values())[0]['cor_pct']
                
                print(f"   📊 Processing {config_key}...")
                
                # Load data for all chi values in this configuration
                chi_values = []
                metrics_data = {
                    'fpr': [],
                    'tpr_malicious': [],
                    'fnr_malicious': [],
                    'tpr_corrupted': [],
                    'fnr_corrupted': [],
                    'accuracy': []
                }
                
                for chi in sorted(chi_data.keys()):
                    data = self.load_experiment_data(chi_data[chi]['csv_path'])
                    if data:
                        chi_values.append(chi)
                        metrics_data['fpr'].append(data['fpr'])
                        metrics_data['tpr_malicious'].append(data['tpr_malicious'])
                        metrics_data['fnr_malicious'].append(data['fnr_malicious'])
                        metrics_data['tpr_corrupted'].append(data['tpr_corrupted'])
                        metrics_data['fnr_corrupted'].append(data['fnr_corrupted'])
                        metrics_data['accuracy'].append(data['accuracy'])
                
                if not chi_values:
                    print(f"   ⚠️ No valid data for {config_key}")
                    continue
                
                # Create plots for this configuration
                fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
                
                # 1. False Positive Rate (FPR)
                ax1.plot(chi_values, metrics_data['fpr'], 'o-', linewidth=3, markersize=8, 
                        color='orange', label='FPR')
                ax1.set_xlabel('Chi Value')
                ax1.set_ylabel('False Positive Rate (%)')
                ax1.set_title('False Positive Rate vs Chi\n(Good nodes incorrectly removed)')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, max(100, max(metrics_data['fpr']) * 1.1) if metrics_data['fpr'] else 100)
                
                # Add value labels on points
                for i, (chi, fpr) in enumerate(zip(chi_values, metrics_data['fpr'])):
                    ax1.annotate(f'{fpr:.1f}%', (chi, fpr), textcoords="offset points", 
                               xytext=(0,10), ha='center')
                
                # 2. True Positive Rate (TPR) for Malicious Detection
                ax2.plot(chi_values, metrics_data['tpr_malicious'], 's-', linewidth=3, markersize=8, 
                        color='red', label='Malicious TPR')
                ax2.set_xlabel('Chi Value')
                ax2.set_ylabel('True Positive Rate (%)')
                ax2.set_title('Malicious Detection Rate vs Chi\n(Malicious nodes correctly rejected)')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, max(100, max(metrics_data['tpr_malicious']) * 1.1) if metrics_data['tpr_malicious'] else 100)
                
                # Add value labels on points
                for i, (chi, tpr) in enumerate(zip(chi_values, metrics_data['tpr_malicious'])):
                    ax2.annotate(f'{tpr:.1f}%', (chi, tpr), textcoords="offset points", 
                               xytext=(0,10), ha='center')
                
                # 3. False Negative Rate (FNR) Comparison
                ax3.plot(chi_values, metrics_data['fnr_malicious'], 's-', linewidth=3, markersize=8, 
                        color='red', label='FNR Malicious')
                if any(val > 0 for val in metrics_data['fnr_corrupted']):
                    ax3.plot(chi_values, metrics_data['fnr_corrupted'], '^-', linewidth=3, markersize=8, 
                            color='purple', label='FNR Corrupted')
                ax3.set_xlabel('Chi Value')
                ax3.set_ylabel('False Negative Rate (%)')
                ax3.set_title('False Negative Rate vs Chi\n(Malicious/Corrupted nodes incorrectly confirmed)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                max_fnr = max(max(metrics_data['fnr_malicious']) if metrics_data['fnr_malicious'] else [0], 
                             max(metrics_data['fnr_corrupted']) if metrics_data['fnr_corrupted'] else [0])
                ax3.set_ylim(0, max(100, max_fnr * 1.1) if max_fnr > 0 else 100)
                
                # Add value labels on points
                for i, (chi, fnr_mal) in enumerate(zip(chi_values, metrics_data['fnr_malicious'])):
                    ax3.annotate(f'{fnr_mal:.1f}%', (chi, fnr_mal), textcoords="offset points", 
                               xytext=(0,10), ha='center', color='red')
                if any(val > 0 for val in metrics_data['fnr_corrupted']):
                    for i, (chi, fnr_cor) in enumerate(zip(chi_values, metrics_data['fnr_corrupted'])):
                        ax3.annotate(f'{fnr_cor:.1f}%', (chi, fnr_cor), textcoords="offset points", 
                                   xytext=(0,-15), ha='center', color='purple')
                
                # 4. Overall System Accuracy
                ax4.plot(chi_values, metrics_data['accuracy'], 'd-', linewidth=3, markersize=8, 
                        color='green', label='System Accuracy')
                ax4.set_xlabel('Chi Value')
                ax4.set_ylabel('Accuracy (%)')
                ax4.set_title('Overall System Accuracy vs Chi\n(Correctly handled nodes / Total nodes)')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, max(100, max(metrics_data['accuracy']) * 1.1) if metrics_data['accuracy'] else 100)
                
                # Add value labels on points
                for i, (chi, acc) in enumerate(zip(chi_values, metrics_data['accuracy'])):
                    ax4.annotate(f'{acc:.1f}%', (chi, acc), textcoords="offset points", 
                               xytext=(0,10), ha='center')
                
                # 5. TPR Comparison (Malicious vs Corrupted)
                ax5.plot(chi_values, metrics_data['tpr_malicious'], 's-', linewidth=3, markersize=8, 
                        color='red', label='TPR Malicious')
                if any(val > 0 for val in metrics_data['tpr_corrupted']):
                    ax5.plot(chi_values, metrics_data['tpr_corrupted'], '^-', linewidth=3, markersize=8, 
                            color='purple', label='TPR Corrupted')
                ax5.set_xlabel('Chi Value')
                ax5.set_ylabel('True Positive Rate (%)')
                ax5.set_title('Detection Rate Comparison vs Chi\n(Malicious vs Corrupted)')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                max_tpr = max(max(metrics_data['tpr_malicious']) if metrics_data['tpr_malicious'] else [0], 
                             max(metrics_data['tpr_corrupted']) if metrics_data['tpr_corrupted'] else [0])
                ax5.set_ylim(0, max(100, max_tpr * 1.1) if max_tpr > 0 else 100)
                
                # 6. Combined view with separate malicious and corrupted metrics
                ax6.plot(chi_values, metrics_data['fpr'], 'o-', linewidth=2, markersize=6, 
                        color='orange', label='FPR (Good removed)')
                ax6.plot(chi_values, metrics_data['tpr_malicious'], '^-', linewidth=2, markersize=6, 
                        color='darkred', label='TPR Malicious')
                ax6.plot(chi_values, metrics_data['fnr_malicious'], 's-', linewidth=2, markersize=6, 
                        color='red', label='FNR Malicious')
                ax6.plot(chi_values, metrics_data['accuracy'], 'd-', linewidth=2, markersize=6, 
                        color='green', label='Overall Accuracy')
                
                # Add corrupted metrics if there are corrupted nodes
                if any(val > 0 for val in metrics_data['tpr_corrupted']):
                    ax6.plot(chi_values, metrics_data['tpr_corrupted'], '^-', linewidth=2, markersize=6, 
                            color='purple', label='TPR Corrupted')
                if any(val > 0 for val in metrics_data['fnr_corrupted']):
                    ax6.plot(chi_values, metrics_data['fnr_corrupted'], 'v-', linewidth=2, markersize=6, 
                            color='darkviolet', label='FNR Corrupted')
                
                ax4.set_xlabel('Chi Value')
                ax4.set_ylabel('Percentage (%)')
                ax4.set_title('All Metrics vs Chi')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 100)
                
                ax6.set_xlabel('Chi Value')
                ax6.set_ylabel('Rate (%)')
                ax6.set_title('Combined Metrics Overview vs Chi')
                ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax6.grid(True, alpha=0.3)
                ax6.set_ylim(0, 100)
                
                plt.suptitle(f'Chi Comparison - Malicious {mal_pct}%, Corrupted {cor_pct}%', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Create overall comparison if we have multiple configurations
            if len(self.experiments) > 1:
                print("   📊 Creating overall comparison...")
                self._create_overall_comparison(pdf)
                
                # Create specific comparison between mal_20_cor_10 and mal_10_cor_20 if both exist
                if 'mal_20_cor_10' in self.experiments and 'mal_10_cor_20' in self.experiments:
                    print("   🔄 Creating direct comparison: mal_20_cor_10 vs mal_10_cor_20...")
                    self._create_direct_configuration_comparison(pdf)
            
            # Create summary statistics page
            print("   📋 Creating summary statistics...")
            self._create_summary_page(pdf)
        
        print(f"📑 Chi comparison PDF saved: {pdf_filename}")
        
        # Save data to CSV for further analysis
        csv_filename = self._save_data_to_csv()
        print(f"💾 Chi comparison data saved: {csv_filename}")
        
        return pdf_filename
    
    def _create_overall_comparison(self, pdf):
        """Create overall comparison across all configurations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.experiments)))
        
        for i, (config_key, chi_data) in enumerate(self.experiments.items()):
            mal_pct = list(chi_data.values())[0]['mal_pct']
            cor_pct = list(chi_data.values())[0]['cor_pct']
            
            chi_values = []
            fpr_values = []
            tpr_values = []
            fnr_values = []
            acc_values = []
            
            for chi in sorted(chi_data.keys()):
                data = self.load_experiment_data(chi_data[chi]['csv_path'])
                if data:
                    chi_values.append(chi)
                    fpr_values.append(data['fpr'])
                    tpr_values.append(data['tpr_malicious'])
                    fnr_values.append(data['fnr_malicious'])
                    acc_values.append(data['accuracy'])
            
            if chi_values:
                label = f"Mal {mal_pct}%, Cor {cor_pct}%"
                
                ax1.plot(chi_values, fpr_values, 'o-', linewidth=2, markersize=6, 
                        color=colors[i], label=label)
                ax2.plot(chi_values, tpr_values, 's-', linewidth=2, markersize=6, 
                        color=colors[i], label=label)
                ax3.plot(chi_values, acc_values, '^-', linewidth=2, markersize=6, 
                        color=colors[i], label=label)
        
        ax1.set_xlabel('Chi Value')
        ax1.set_ylabel('False Positive Rate (%)')
        ax1.set_title('FPR Comparison Across Configurations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Chi Value')
        ax2.set_ylabel('True Positive Rate (%)')
        ax2.set_title('TPR Comparison Across Configurations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Chi Value')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy Comparison Across Configurations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chi effectiveness ranking
        ax4.axis('off')
        ranking_text = "CHI EFFECTIVENESS RANKING\\n"
        ranking_text += "="*40 + "\\n\\n"
        
        # Calculate average performance for each chi across all configurations
        chi_performance = {}
        for config_key, chi_data in self.experiments.items():
            for chi in chi_data.keys():
                data = self.load_experiment_data(chi_data[chi]['csv_path'])
                if data:
                    if chi not in chi_performance:
                        chi_performance[chi] = {'fpr': [], 'tpr': [], 'fnr': [], 'accuracy': []}
                    chi_performance[chi]['fpr'].append(data['fpr'])
                    chi_performance[chi]['tpr'].append(data['tpr_malicious'])
                    chi_performance[chi]['fnr'].append(data['fnr_malicious'])
                    chi_performance[chi]['accuracy'].append(data['accuracy'])
        
        # Calculate averages and rank
        chi_rankings = []
        for chi, metrics in chi_performance.items():
            avg_fpr = np.mean(metrics['fpr'])
            avg_tpr = np.mean(metrics['tpr'])
            avg_fnr = np.mean(metrics['fnr'])
            avg_accuracy = np.mean(metrics['accuracy'])
            
            # Composite score: Higher TPR and Accuracy are better, lower FPR and FNR are better
            score = avg_tpr + avg_accuracy - avg_fpr - avg_fnr
            chi_rankings.append((chi, score, avg_fpr, avg_tpr, avg_fnr, avg_accuracy))
        
        chi_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (chi, score, fpr, tpr, fnr, acc) in enumerate(chi_rankings, 1):
            ranking_text += f"{rank}. Chi = {chi}\\n"
            ranking_text += f"   FPR: {fpr:.1f}%, TPR: {tpr:.1f}%, FNR: {fnr:.1f}%, Acc: {acc:.1f}%\\n"
            ranking_text += f"   Score: {score:.1f}\\n\\n"
        
        ax4.text(0.05, 0.95, ranking_text, transform=ax4.transAxes, fontsize=11, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Overall Chi Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_direct_configuration_comparison(self, pdf):
        """Create direct comparison between mal_20_cor_10 and mal_10_cor_20 configurations."""
        config1_key = 'mal_20_cor_10'
        config2_key = 'mal_10_cor_20'
        
        # Load data for both configurations
        config1_data = {}
        config2_data = {}
        
        # Load config1 data (mal=20%, cor=10%)
        for chi in sorted(self.experiments[config1_key].keys()):
            data = self.load_experiment_data(self.experiments[config1_key][chi]['csv_path'])
            if data:
                config1_data[chi] = data
        
        # Load config2 data (mal=10%, cor=20%)
        for chi in sorted(self.experiments[config2_key].keys()):
            data = self.load_experiment_data(self.experiments[config2_key][chi]['csv_path'])
            if data:
                config2_data[chi] = data
        
        # Get common chi values
        common_chi_values = sorted(set(config1_data.keys()) & set(config2_data.keys()))
        
        if not common_chi_values:
            print("   ⚠️ No common chi values between configurations")
            return
        
        # Create comparison plots
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        # Extract metrics for both configurations
        config1_metrics = {
            'fpr': [config1_data[chi]['fpr'] for chi in common_chi_values],
            'tpr_malicious': [config1_data[chi]['tpr_malicious'] for chi in common_chi_values],
            'fnr_malicious': [config1_data[chi]['fnr_malicious'] for chi in common_chi_values],
            'tpr_corrupted': [config1_data[chi]['tpr_corrupted'] for chi in common_chi_values],
            'fnr_corrupted': [config1_data[chi]['fnr_corrupted'] for chi in common_chi_values],
            'accuracy': [config1_data[chi]['accuracy'] for chi in common_chi_values]
        }
        
        config2_metrics = {
            'fpr': [config2_data[chi]['fpr'] for chi in common_chi_values],
            'tpr_malicious': [config2_data[chi]['tpr_malicious'] for chi in common_chi_values],
            'fnr_malicious': [config2_data[chi]['fnr_malicious'] for chi in common_chi_values],
            'tpr_corrupted': [config2_data[chi]['tpr_corrupted'] for chi in common_chi_values],
            'fnr_corrupted': [config2_data[chi]['fnr_corrupted'] for chi in common_chi_values],
            'accuracy': [config2_data[chi]['accuracy'] for chi in common_chi_values]
        }
        
        # Colors for the two configurations
        color1 = 'darkred'   # For mal=20%, cor=10%
        color2 = 'darkblue'  # For mal=10%, cor=20%
        
        # 1. False Positive Rate (FPR) Comparison
        ax1.plot(common_chi_values, config1_metrics['fpr'], 'o-', linewidth=3, markersize=8, 
                color=color1, label='Mal 20%, Cor 10%')
        ax1.plot(common_chi_values, config2_metrics['fpr'], 's-', linewidth=3, markersize=8, 
                color=color2, label='Mal 10%, Cor 20%')
        ax1.set_xlabel('Chi Value')
        ax1.set_ylabel('False Positive Rate (%)')
        ax1.set_title('FPR Comparison\n(Good nodes incorrectly removed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(100, max(max(config1_metrics['fpr']), max(config2_metrics['fpr'])) * 1.1))
        
        # 2. True Positive Rate (TPR) for Malicious Detection
        ax2.plot(common_chi_values, config1_metrics['tpr_malicious'], 'o-', linewidth=3, markersize=8, 
                color=color1, label='Mal 20%, Cor 10%')
        ax2.plot(common_chi_values, config2_metrics['tpr_malicious'], 's-', linewidth=3, markersize=8, 
                color=color2, label='Mal 10%, Cor 20%')
        ax2.set_xlabel('Chi Value')
        ax2.set_ylabel('True Positive Rate (%)')
        ax2.set_title('Malicious Detection Rate\n(Malicious nodes correctly rejected)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(100, max(max(config1_metrics['tpr_malicious']), max(config2_metrics['tpr_malicious'])) * 1.1))
        
        # 3. False Negative Rate (FNR) for Malicious
        ax3.plot(common_chi_values, config1_metrics['fnr_malicious'], 'o-', linewidth=3, markersize=8, 
                color=color1, label='Mal 20%, Cor 10%')
        ax3.plot(common_chi_values, config2_metrics['fnr_malicious'], 's-', linewidth=3, markersize=8, 
                color=color2, label='Mal 10%, Cor 20%')
        ax3.set_xlabel('Chi Value')
        ax3.set_ylabel('False Negative Rate (%)')
        ax3.set_title('Malicious FNR Comparison\n(Malicious nodes incorrectly confirmed)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(100, max(max(config1_metrics['fnr_malicious']), max(config2_metrics['fnr_malicious'])) * 1.1))
        
        # 4. True Positive Rate (TPR) for Corrupted Detection
        ax4.plot(common_chi_values, config1_metrics['tpr_corrupted'], 'o-', linewidth=3, markersize=8, 
                color=color1, label='Mal 20%, Cor 10%')
        ax4.plot(common_chi_values, config2_metrics['tpr_corrupted'], 's-', linewidth=3, markersize=8, 
                color=color2, label='Mal 10%, Cor 20%')
        ax4.set_xlabel('Chi Value')
        ax4.set_ylabel('True Positive Rate (%)')
        ax4.set_title('Corrupted Detection Rate\n(Corrupted nodes correctly rejected)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(100, max(max(config1_metrics['tpr_corrupted']), max(config2_metrics['tpr_corrupted'])) * 1.1))
        
        # 5. Overall System Accuracy
        ax5.plot(common_chi_values, config1_metrics['accuracy'], 'o-', linewidth=3, markersize=8, 
                color=color1, label='Mal 20%, Cor 10%')
        ax5.plot(common_chi_values, config2_metrics['accuracy'], 's-', linewidth=3, markersize=8, 
                color=color2, label='Mal 10%, Cor 20%')
        ax5.set_xlabel('Chi Value')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title('Overall System Accuracy\n(Correctly handled nodes / Total nodes)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, max(100, max(max(config1_metrics['accuracy']), max(config2_metrics['accuracy'])) * 1.1))
        
        # 6. Combined Key Metrics
        ax6.plot(common_chi_values, config1_metrics['fpr'], 'o-', linewidth=2, markersize=6, 
                color=color1, alpha=0.7, label='FPR - Mal 20%, Cor 10%')
        ax6.plot(common_chi_values, config2_metrics['fpr'], 's-', linewidth=2, markersize=6, 
                color=color2, alpha=0.7, label='FPR - Mal 10%, Cor 20%')
        
        ax6.plot(common_chi_values, config1_metrics['tpr_malicious'], '^-', linewidth=2, markersize=6, 
                color=color1, label='TPR Mal - Mal 20%, Cor 10%')
        ax6.plot(common_chi_values, config2_metrics['tpr_malicious'], 'v-', linewidth=2, markersize=6, 
                color=color2, label='TPR Mal - Mal 10%, Cor 20%')
        
        ax6.plot(common_chi_values, config1_metrics['accuracy'], 'd-', linewidth=2, markersize=6, 
                color=color1, alpha=0.8, label='Accuracy - Mal 20%, Cor 10%')
        ax6.plot(common_chi_values, config2_metrics['accuracy'], 'p-', linewidth=2, markersize=6, 
                color=color2, alpha=0.8, label='Accuracy - Mal 10%, Cor 20%')
        
        ax6.set_xlabel('Chi Value')
        ax6.set_ylabel('Rate (%)')
        ax6.set_title('Combined Metrics Comparison')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 100)
        
        plt.suptitle('Direct Configuration Comparison: Mal 20%/Cor 10% vs Mal 10%/Cor 20%', 
                   fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_summary_page(self, pdf):
        """Create summary statistics page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        summary_text = f"""
CHI COMPARISON ANALYSIS - SUMMARY
================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results Path: {self.results_base_path}

CONFIGURATIONS ANALYZED:
"""
        
        for config_key, chi_data in self.experiments.items():
            mal_pct = list(chi_data.values())[0]['mal_pct']
            cor_pct = list(chi_data.values())[0]['cor_pct']
            chi_values = sorted(chi_data.keys())
            
            summary_text += f"\\n• Malicious {mal_pct}%, Corrupted {cor_pct}%\\n"
            summary_text += f"  Chi values: {chi_values}\\n"
            
            # Get detailed stats for this configuration
            for chi in chi_values:
                data = self.load_experiment_data(chi_data[chi]['csv_path'])
                if data:
                    summary_text += f"    Chi {chi}: FPR={data['fpr']:.1f}%, TPR={data['tpr_malicious']:.1f}%, FNR={data['fnr_malicious']:.1f}%, Acc={data['accuracy']:.1f}%\\n"

        summary_text += f"""
\\nMETRICS EXPLANATION:
• FPR (False Positive Rate): Good nodes incorrectly removed / Total good nodes sent
• TPR (True Positive Rate): Malicious nodes correctly detected / Total malicious nodes sent
• FNR (False Negative Rate): Malicious nodes incorrectly confirmed / Total malicious nodes sent  
• Accuracy: Overall system accuracy (correctly handled nodes / total nodes)

CALCULATION METHOD:
• Each round selects K-1 = 9 nodes randomly from K = 10 peers
• Expected nodes per type = (9 × peer_count_of_type) / 10 × total_rounds
• Sums Good_Nodes_Removed_This_Round and Malicious_Nodes_Confirmed_This_Round across ALL rounds
• Uses selection probability based on peer distribution

OPTIMAL VALUES:
• FPR: 0% (no good nodes removed)
• TPR: 100% (all malicious nodes detected)
• FNR: 0% (no malicious nodes confirmed)
• Accuracy: 100% (perfect classification)

CHI PARAMETER:
The chi parameter controls the depth of past cone inspection
in the TrustInformedTipSelector_NoThreshold algorithm.
Higher chi values inspect deeper history for similarity analysis.
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        
        plt.title('Summary Statistics', fontsize=18, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _save_data_to_csv(self):
        """Save all comparison data to CSV files for further analysis."""
        import csv
        
        # Create CSV data directory
        csv_dir = Path(self.output_dir) / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        
        # Save individual configuration data
        for config_key, chi_data in self.experiments.items():
            mal_pct = list(chi_data.values())[0]['mal_pct']
            cor_pct = list(chi_data.values())[0]['cor_pct']
            
            config_filename = csv_dir / f"chi_data_{config_key}.csv"
            
            # Collect data for this configuration
            rows = []
            for chi in sorted(chi_data.keys()):
                data = self.load_experiment_data(chi_data[chi]['csv_path'])
                if data:
                    rows.append({
                        'chi': chi,
                        'malicious_pct': mal_pct,
                        'corrupted_pct': cor_pct,
                        'configuration': config_key,
                        'fpr': data['fpr'],
                        'tpr_malicious': data['tpr_malicious'],
                        'fnr_malicious': data['fnr_malicious'],
                        'tpr_corrupted': data['tpr_corrupted'],
                        'fnr_corrupted': data['fnr_corrupted'],
                        'accuracy': data['accuracy'],
                        'total_malicious_peers': data['total_malicious_peers'],
                        'total_corrupted_peers': data['total_corrupted_peers'],
                        'total_good_peers': data['total_good_peers'],
                        'total_rounds': data['total_rounds'],
                        'actual_total_malicious_sent': data['actual_total_malicious_sent'],
                        'actual_total_corrupted_sent': data['actual_total_corrupted_sent'],
                        'actual_total_good_sent': data['actual_total_good_sent'],
                        'total_malicious_confirmed': data['total_malicious_confirmed'],
                        'total_corrupted_confirmed': data['total_corrupted_confirmed'],
                        'total_good_removed': data['total_good_removed']
                    })
            
            # Save configuration-specific CSV
            if rows:
                with open(config_filename, 'w', newline='') as csvfile:
                    fieldnames = rows[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"   💾 Saved {config_filename}")
        
        # Save combined data for all configurations
        combined_filename = csv_dir / "chi_comparison_all_data.csv"
        all_rows = []
        
        for config_key, chi_data in self.experiments.items():
            mal_pct = list(chi_data.values())[0]['mal_pct']
            cor_pct = list(chi_data.values())[0]['cor_pct']
            
            for chi in sorted(chi_data.keys()):
                data = self.load_experiment_data(chi_data[chi]['csv_path'])
                if data:
                    all_rows.append({
                        'chi': chi,
                        'malicious_pct': mal_pct,
                        'corrupted_pct': cor_pct,
                        'configuration': config_key,
                        'fpr': data['fpr'],
                        'tpr_malicious': data['tpr_malicious'],
                        'fnr_malicious': data['fnr_malicious'],
                        'tpr_corrupted': data['tpr_corrupted'],
                        'fnr_corrupted': data['fnr_corrupted'],
                        'accuracy': data['accuracy'],
                        'total_malicious_peers': data['total_malicious_peers'],
                        'total_corrupted_peers': data['total_corrupted_peers'],
                        'total_good_peers': data['total_good_peers'],
                        'total_rounds': data['total_rounds'],
                        'actual_total_malicious_sent': data['actual_total_malicious_sent'],
                        'actual_total_corrupted_sent': data['actual_total_corrupted_sent'],
                        'actual_total_good_sent': data['actual_total_good_sent'],
                        'total_malicious_confirmed': data['total_malicious_confirmed'],
                        'total_corrupted_confirmed': data['total_corrupted_confirmed'],
                        'total_good_removed': data['total_good_removed']
                    })
        
        # Save combined CSV
        if all_rows:
            with open(combined_filename, 'w', newline='') as csvfile:
                fieldnames = all_rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        
        # Save summary statistics CSV
        summary_filename = csv_dir / "chi_summary_statistics.csv"
        self._save_summary_statistics_csv(summary_filename)
        
        # Save direct comparison data if both configurations exist
        if 'mal_20_cor_10' in self.experiments and 'mal_10_cor_20' in self.experiments:
            comparison_filename = csv_dir / "chi_direct_comparison_mal20cor10_vs_mal10cor20.csv"
            self._save_direct_comparison_csv(comparison_filename)
        
        return combined_filename
    
    def _save_summary_statistics_csv(self, filename):
        """Save summary statistics to CSV."""
        import csv
        
        # Calculate average performance for each chi across all configurations
        chi_performance = {}
        for config_key, chi_data in self.experiments.items():
            for chi in chi_data.keys():
                data = self.load_experiment_data(chi_data[chi]['csv_path'])
                if data:
                    if chi not in chi_performance:
                        chi_performance[chi] = {'fpr': [], 'tpr': [], 'fnr': [], 'accuracy': []}
                    chi_performance[chi]['fpr'].append(data['fpr'])
                    chi_performance[chi]['tpr'].append(data['tpr_malicious'])
                    chi_performance[chi]['fnr'].append(data['fnr_malicious'])
                    chi_performance[chi]['accuracy'].append(data['accuracy'])
        
        # Calculate averages and create summary rows
        summary_rows = []
        for chi, metrics in chi_performance.items():
            avg_fpr = np.mean(metrics['fpr'])
            avg_tpr = np.mean(metrics['tpr'])
            avg_fnr = np.mean(metrics['fnr'])
            avg_accuracy = np.mean(metrics['accuracy'])
            
            std_fpr = np.std(metrics['fpr'])
            std_tpr = np.std(metrics['tpr'])
            std_fnr = np.std(metrics['fnr'])
            std_accuracy = np.std(metrics['accuracy'])
            
            # Composite score: Higher TPR and Accuracy are better, lower FPR and FNR are better
            score = avg_tpr + avg_accuracy - avg_fpr - avg_fnr
            
            summary_rows.append({
                'chi': chi,
                'avg_fpr': avg_fpr,
                'std_fpr': std_fpr,
                'avg_tpr_malicious': avg_tpr,
                'std_tpr_malicious': std_tpr,
                'avg_fnr_malicious': avg_fnr,
                'std_fnr_malicious': std_fnr,
                'avg_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy,
                'composite_score': score,
                'num_configurations': len(metrics['fpr'])
            })
        
        # Sort by composite score (best first)
        summary_rows.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add ranking
        for rank, row in enumerate(summary_rows, 1):
            row['rank'] = rank
        
        # Save to CSV
        if summary_rows:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['rank', 'chi', 'avg_fpr', 'std_fpr', 'avg_tpr_malicious', 'std_tpr_malicious', 
                             'avg_fnr_malicious', 'std_fnr_malicious', 'avg_accuracy', 'std_accuracy', 
                             'composite_score', 'num_configurations']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"   📊 Saved summary statistics: {filename}")
    
    def _save_direct_comparison_csv(self, filename):
        """Save direct comparison data between mal_20_cor_10 and mal_10_cor_20."""
        import csv
        
        config1_key = 'mal_20_cor_10'
        config2_key = 'mal_10_cor_20'
        
        # Load data for both configurations
        config1_data = {}
        config2_data = {}
        
        for chi in sorted(self.experiments[config1_key].keys()):
            data = self.load_experiment_data(self.experiments[config1_key][chi]['csv_path'])
            if data:
                config1_data[chi] = data
        
        for chi in sorted(self.experiments[config2_key].keys()):
            data = self.load_experiment_data(self.experiments[config2_key][chi]['csv_path'])
            if data:
                config2_data[chi] = data
        
        # Get common chi values and create comparison rows
        common_chi_values = sorted(set(config1_data.keys()) & set(config2_data.keys()))
        comparison_rows = []
        
        for chi in common_chi_values:
            data1 = config1_data[chi]
            data2 = config2_data[chi]
            
            comparison_rows.append({
                'chi': chi,
                # Config 1 (mal 20%, cor 10%) metrics
                'mal20_cor10_fpr': data1['fpr'],
                'mal20_cor10_tpr_malicious': data1['tpr_malicious'],
                'mal20_cor10_fnr_malicious': data1['fnr_malicious'],
                'mal20_cor10_tpr_corrupted': data1['tpr_corrupted'],
                'mal20_cor10_fnr_corrupted': data1['fnr_corrupted'],
                'mal20_cor10_accuracy': data1['accuracy'],
                # Config 2 (mal 10%, cor 20%) metrics
                'mal10_cor20_fpr': data2['fpr'],
                'mal10_cor20_tpr_malicious': data2['tpr_malicious'],
                'mal10_cor20_fnr_malicious': data2['fnr_malicious'],
                'mal10_cor20_tpr_corrupted': data2['tpr_corrupted'],
                'mal10_cor20_fnr_corrupted': data2['fnr_corrupted'],
                'mal10_cor20_accuracy': data2['accuracy'],
                # Differences (config1 - config2)
                'diff_fpr': data1['fpr'] - data2['fpr'],
                'diff_tpr_malicious': data1['tpr_malicious'] - data2['tpr_malicious'],
                'diff_fnr_malicious': data1['fnr_malicious'] - data2['fnr_malicious'],
                'diff_tpr_corrupted': data1['tpr_corrupted'] - data2['tpr_corrupted'],
                'diff_fnr_corrupted': data1['fnr_corrupted'] - data2['fnr_corrupted'],
                'diff_accuracy': data1['accuracy'] - data2['accuracy']
            })
        
        # Save comparison CSV
        if comparison_rows:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = comparison_rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(comparison_rows)
            print(f"   🔄 Saved direct comparison: {filename}")

def main():
    """Main function to run the chi comparison analysis."""
    print("🚀 Starting Chi Results Comparison Analysis...")
    
    # Initialize comparator
    comparator = ChiResultsComparator()
    
    # Scan for experiment folders
    if not comparator.scan_experiment_folders():
        print("❌ No experiments found. Make sure to run automated_test_chi.py first!")
        return
    
    # Create comparison plots
    pdf_path = comparator.create_comparison_plots()
    
    print(f"\\n{'='*60}")
    print("🎉 CHI COMPARISON ANALYSIS COMPLETED!")
    print(f"📑 Report saved: {pdf_path}")
    print("📁 Plots saved in: chi_comparison_plots/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
