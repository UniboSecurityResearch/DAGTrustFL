#!/usr/bin/env python3
"""
Lambda Results Comparison Script for DAGTrustFL
===============================================

This script analyzes and compares results from experiments with different lambda values
and fixed malicious/corrupted percentages (20% malicious, 10% corrupted), creating comparative graphs for:

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

class LambdaResultsComparator:
    """Class for comparing results from different lambda experiments."""
    
    def __init__(self, results_base_path="../FL/results_lambda"):
        """
        Initialize the comparator.
        
        Args:
            results_base_path (str): Base path where lambda results are stored
        """
        self.results_base_path = Path(results_base_path)
        self.output_dir = "../plots/lambda_comparison_plots"
        self.experiments = {}
        
        # Create directory for plots
        os.makedirs(self.output_dir, exist_ok=True)
        
    def scan_experiment_folders(self):
        """Scan for experiment folders and categorize them."""
        print("🔍 Scanning for lambda experiment folders...")
        
        if not self.results_base_path.exists():
            print(f"❌ Results path {self.results_base_path} does not exist!")
            return False
        
        # Pattern: lambda_X_Y (e.g., lambda_0_3 for λ=0.3)
        pattern = r'lambda_(\d+)_?(\d*)'
        
        for folder in self.results_base_path.iterdir():
            if folder.is_dir():
                match = re.match(pattern, folder.name)
                if match:
                    # Reconstruct lambda value from folder name
                    integer_part = match.group(1)
                    decimal_part = match.group(2) if match.group(2) else ""
                    
                    if decimal_part:
                        lambda_value = float(f"{integer_part}.{decimal_part}")
                    else:
                        lambda_value = float(integer_part)
                    
                    # Check if dag_trust_analysis.csv exists
                    csv_path = folder / 'dag_trust_analysis.csv'
                    if csv_path.exists():
                        self.experiments[lambda_value] = {
                            'folder': folder,
                            'csv_path': csv_path,
                            'lambda_value': lambda_value
                        }
                        
                        print(f"✅ Found: lambda={lambda_value} -> {csv_path}")
                    else:
                        print(f"⚠️ Missing CSV in {folder.name}")
        
        if not self.experiments:
            print("❌ No valid experiment folders found!")
            return False
        
        print(f"\n📊 Found {len(self.experiments)} lambda experiments:")
        lambda_values = sorted(self.experiments.keys())
        print(f"   • Lambda values: {lambda_values}")
        
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
        """Create comparison plots for all lambda experiments."""
        pdf_filename = f"{self.output_dir}/lambda_comparison_report.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            print("📑 Creating lambda comparison PDF...")
            
            # Load data for all lambda values
            lambda_values = []
            metrics_data = {
                'fpr': [],
                'tpr_malicious': [],
                'fnr_malicious': [],
                'tpr_corrupted': [],
                'fnr_corrupted': [],
                'accuracy': []
            }
            
            for lambda_val in sorted(self.experiments.keys()):
                data = self.load_experiment_data(self.experiments[lambda_val]['csv_path'])
                if data:
                    lambda_values.append(lambda_val)
                    metrics_data['fpr'].append(data['fpr'])
                    metrics_data['tpr_malicious'].append(data['tpr_malicious'])
                    metrics_data['fnr_malicious'].append(data['fnr_malicious'])
                    metrics_data['tpr_corrupted'].append(data['tpr_corrupted'])
                    metrics_data['fnr_corrupted'].append(data['fnr_corrupted'])
                    metrics_data['accuracy'].append(data['accuracy'])
            
            if not lambda_values:
                print("   ⚠️ No valid data found")
                return pdf_filename
            
            # Create plots for lambda comparison
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
            
            # 1. False Positive Rate (FPR)
            ax1.plot(lambda_values, metrics_data['fpr'], 'o-', linewidth=3, markersize=8, 
                    color='orange', label='FPR')
            ax1.set_xlabel('Lambda Value')
            ax1.set_ylabel('False Positive Rate (%)')
            ax1.set_title('False Positive Rate vs Lambda\n(Good nodes incorrectly removed)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(100, max(metrics_data['fpr']) * 1.1) if metrics_data['fpr'] else 100)
            
            # Add value labels on points
            for i, (lam, fpr) in enumerate(zip(lambda_values, metrics_data['fpr'])):
                ax1.annotate(f'{fpr:.1f}%', (lam, fpr), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            # 2. True Positive Rate (TPR) for Malicious Detection
            ax2.plot(lambda_values, metrics_data['tpr_malicious'], 's-', linewidth=3, markersize=8, 
                    color='red', label='Malicious TPR')
            ax2.set_xlabel('Lambda Value')
            ax2.set_ylabel('True Positive Rate (%)')
            ax2.set_title('Malicious Detection Rate vs Lambda\n(Malicious nodes correctly rejected)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(100, max(metrics_data['tpr_malicious']) * 1.1) if metrics_data['tpr_malicious'] else 100)
            
            # Add value labels on points
            for i, (lam, tpr) in enumerate(zip(lambda_values, metrics_data['tpr_malicious'])):
                ax2.annotate(f'{tpr:.1f}%', (lam, tpr), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            # 3. False Negative Rate (FNR) Comparison
            ax3.plot(lambda_values, metrics_data['fnr_malicious'], 's-', linewidth=3, markersize=8, 
                    color='red', label='FNR Malicious')
            if any(val > 0 for val in metrics_data['fnr_corrupted']):
                ax3.plot(lambda_values, metrics_data['fnr_corrupted'], '^-', linewidth=3, markersize=8, 
                        color='purple', label='FNR Corrupted')
            ax3.set_xlabel('Lambda Value')
            ax3.set_ylabel('False Negative Rate (%)')
            ax3.set_title('False Negative Rate vs Lambda\n(Malicious/Corrupted nodes incorrectly confirmed)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            max_fnr = max(max(metrics_data['fnr_malicious']) if metrics_data['fnr_malicious'] else [0], 
                         max(metrics_data['fnr_corrupted']) if metrics_data['fnr_corrupted'] else [0])
            ax3.set_ylim(0, max(100, max_fnr * 1.1) if max_fnr > 0 else 100)
            
            # Add value labels on points
            for i, (lam, fnr_mal) in enumerate(zip(lambda_values, metrics_data['fnr_malicious'])):
                ax3.annotate(f'{fnr_mal:.1f}%', (lam, fnr_mal), textcoords="offset points", 
                           xytext=(0,10), ha='center', color='red')
            if any(val > 0 for val in metrics_data['fnr_corrupted']):
                for i, (lam, fnr_cor) in enumerate(zip(lambda_values, metrics_data['fnr_corrupted'])):
                    ax3.annotate(f'{fnr_cor:.1f}%', (lam, fnr_cor), textcoords="offset points", 
                               xytext=(0,-15), ha='center', color='purple')
            
            # 4. Overall System Accuracy
            ax4.plot(lambda_values, metrics_data['accuracy'], 'd-', linewidth=3, markersize=8, 
                    color='green', label='System Accuracy')
            ax4.set_xlabel('Lambda Value')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Overall System Accuracy vs Lambda\n(Correctly handled nodes / Total nodes)')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, max(100, max(metrics_data['accuracy']) * 1.1) if metrics_data['accuracy'] else 100)
            
            # Add value labels on points
            for i, (lam, acc) in enumerate(zip(lambda_values, metrics_data['accuracy'])):
                ax4.annotate(f'{acc:.1f}%', (lam, acc), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            # 5. TPR Comparison (Malicious vs Corrupted)
            ax5.plot(lambda_values, metrics_data['tpr_malicious'], 's-', linewidth=3, markersize=8, 
                    color='red', label='TPR Malicious')
            if any(val > 0 for val in metrics_data['tpr_corrupted']):
                ax5.plot(lambda_values, metrics_data['tpr_corrupted'], '^-', linewidth=3, markersize=8, 
                        color='purple', label='TPR Corrupted')
            ax5.set_xlabel('Lambda Value')
            ax5.set_ylabel('True Positive Rate (%)')
            ax5.set_title('Detection Rate Comparison vs Lambda\n(Malicious vs Corrupted)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            max_tpr = max(max(metrics_data['tpr_malicious']) if metrics_data['tpr_malicious'] else [0], 
                         max(metrics_data['tpr_corrupted']) if metrics_data['tpr_corrupted'] else [0])
            ax5.set_ylim(0, max(100, max_tpr * 1.1) if max_tpr > 0 else 100)
            
            # 6. Combined view with all metrics
            ax6.plot(lambda_values, metrics_data['fpr'], 'o-', linewidth=2, markersize=6, 
                    color='orange', label='FPR (Good removed)')
            ax6.plot(lambda_values, metrics_data['tpr_malicious'], '^-', linewidth=2, markersize=6, 
                    color='darkred', label='TPR Malicious')
            ax6.plot(lambda_values, metrics_data['fnr_malicious'], 's-', linewidth=2, markersize=6, 
                    color='red', label='FNR Malicious')
            ax6.plot(lambda_values, metrics_data['accuracy'], 'd-', linewidth=2, markersize=6, 
                    color='green', label='Overall Accuracy')
            
            # Add corrupted metrics if there are corrupted nodes
            if any(val > 0 for val in metrics_data['tpr_corrupted']):
                ax6.plot(lambda_values, metrics_data['tpr_corrupted'], '^-', linewidth=2, markersize=6, 
                        color='purple', label='TPR Corrupted')
            if any(val > 0 for val in metrics_data['fnr_corrupted']):
                ax6.plot(lambda_values, metrics_data['fnr_corrupted'], 'v-', linewidth=2, markersize=6, 
                        color='darkviolet', label='FNR Corrupted')
            
            ax6.set_xlabel('Lambda Value')
            ax6.set_ylabel('Rate (%)')
            ax6.set_title('Combined Metrics Overview vs Lambda')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 100)
            
            plt.suptitle('Lambda Comparison - Malicious 20%, Corrupted 10%', 
                       fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Create lambda effectiveness ranking page
            print("   📊 Creating lambda effectiveness ranking...")
            self._create_lambda_ranking(pdf, lambda_values, metrics_data)
            
            # Create summary statistics page
            print("   📋 Creating summary statistics...")
            self._create_summary_page(pdf, lambda_values, metrics_data)
        
        print(f"📑 Lambda comparison PDF saved: {pdf_filename}")
        
        # Save data to CSV for further analysis
        csv_filename = self._save_data_to_csv()
        print(f"💾 Lambda comparison data saved: {csv_filename}")
        
        return pdf_filename
    
    def _create_lambda_ranking(self, pdf, lambda_values, metrics_data):
        """Create lambda effectiveness ranking page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        ranking_text = "LAMBDA EFFECTIVENESS RANKING\n"
        ranking_text += "="*40 + "\n\n"
        
        # Calculate composite score for each lambda
        lambda_rankings = []
        for i, lambda_val in enumerate(lambda_values):
            fpr = metrics_data['fpr'][i]
            tpr = metrics_data['tpr_malicious'][i]
            fnr = metrics_data['fnr_malicious'][i]
            acc = metrics_data['accuracy'][i]
            
            # Composite score: Higher TPR and Accuracy are better, lower FPR and FNR are better
            score = tpr + acc - fpr - fnr
            lambda_rankings.append((lambda_val, score, fpr, tpr, fnr, acc))
        
        lambda_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (lambda_val, score, fpr, tpr, fnr, acc) in enumerate(lambda_rankings, 1):
            ranking_text += f"{rank}. Lambda = {lambda_val}\n"
            ranking_text += f"   FPR: {fpr:.1f}%, TPR: {tpr:.1f}%, FNR: {fnr:.1f}%, Acc: {acc:.1f}%\n"
            ranking_text += f"   Score: {score:.1f}\n\n"
        
        ranking_text += "\nLAMBDA PARAMETER:\n"
        ranking_text += "The lambda parameter controls the weight of historical trust\n"
        ranking_text += "information in the TrustInformedTipSelector algorithm.\n"
        ranking_text += "Higher lambda values give more weight to past behavior.\n"
        ranking_text += "\nLAMBDA VALUES TESTED:\n"
        ranking_text += f"Values: {sorted(lambda_values)}\n"
        ranking_text += "Configuration: Malicious 20%, Corrupted 10%"
        
        ax.text(0.05, 0.95, ranking_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.title('Lambda Effectiveness Ranking', fontsize=18, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_summary_page(self, pdf, lambda_values, metrics_data):
        """Create summary statistics page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        summary_text = f"""
LAMBDA COMPARISON ANALYSIS - SUMMARY
====================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results Path: {self.results_base_path}

CONFIGURATION ANALYZED:
• Malicious: 20%, Corrupted: 10%
• Lambda values tested: {sorted(lambda_values)}

DETAILED RESULTS:
"""
        
        for i, lambda_val in enumerate(sorted(lambda_values)):
            idx = lambda_values.index(lambda_val)
            fpr = metrics_data['fpr'][idx]
            tpr = metrics_data['tpr_malicious'][idx]
            fnr = metrics_data['fnr_malicious'][idx]
            acc = metrics_data['accuracy'][idx]
            
            summary_text += f"\\n• Lambda {lambda_val}:\\n"
            summary_text += f"  FPR={fpr:.1f}%, TPR={tpr:.1f}%, FNR={fnr:.1f}%, Acc={acc:.1f}%\\n"

        summary_text += f"""

METRICS EXPLANATION:
• FPR (False Positive Rate): Good nodes incorrectly removed / Total good nodes sent
• TPR (True Positive Rate): Malicious nodes correctly detected / Total malicious nodes sent
• FNR (False Negative Rate): Malicious nodes incorrectly confirmed / Total malicious nodes sent  
• Accuracy: Overall system accuracy (correctly handled nodes / total nodes)

CALCULATION METHOD:
• Each round selects K-1 = 9 nodes randomly from K = 10 peers
• Uses actual worker selection data from 3000_workers_selected.csv
• Sums Good_Nodes_Removed_This_Round and Malicious_Nodes_Confirmed_This_Round across ALL rounds
• Counts actual occurrences of each peer type in selected workers

OPTIMAL VALUES:
• FPR: 0% (no good nodes removed)
• TPR: 100% (all malicious nodes detected)
• FNR: 0% (no malicious nodes confirmed)
• Accuracy: 100% (perfect classification)

LAMBDA PARAMETER:
The lambda parameter controls the weight of historical trust information
in the TrustInformedTipSelector algorithm. It determines how much influence
past behavior has on current trust decisions:
• Lambda = 0.0: No historical influence (only current round)
• Lambda = 1.0: Maximum historical influence
• Higher lambda values may improve detection but can be slower to adapt
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
        
        # Save lambda comparison data
        lambda_filename = csv_dir / "lambda_comparison_data.csv"
        
        # Collect data for all lambda values
        rows = []
        for lambda_val in sorted(self.experiments.keys()):
            data = self.load_experiment_data(self.experiments[lambda_val]['csv_path'])
            if data:
                rows.append({
                    'lambda': lambda_val,
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
        
        # Save CSV
        if rows:
            with open(lambda_filename, 'w', newline='') as csvfile:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"   💾 Saved {lambda_filename}")
        
        # Save summary statistics CSV
        summary_filename = csv_dir / "lambda_summary_statistics.csv"
        self._save_summary_statistics_csv(summary_filename, rows)
        
        return lambda_filename
    
    def _save_summary_statistics_csv(self, filename, rows):
        """Save summary statistics to CSV."""
        import csv
        
        # Calculate summary statistics and rankings
        summary_rows = []
        for row in rows:
            lambda_val = row['lambda']
            fpr = row['fpr']
            tpr = row['tpr_malicious']
            fnr = row['fnr_malicious']
            acc = row['accuracy']
            
            # Composite score: Higher TPR and Accuracy are better, lower FPR and FNR are better
            score = tpr + acc - fpr - fnr
            
            summary_rows.append({
                'lambda': lambda_val,
                'fpr': fpr,
                'tpr_malicious': tpr,
                'fnr_malicious': fnr,
                'accuracy': acc,
                'composite_score': score
            })
        
        # Sort by composite score (best first)
        summary_rows.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add ranking
        for rank, row in enumerate(summary_rows, 1):
            row['rank'] = rank
        
        # Save to CSV
        if summary_rows:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['rank', 'lambda', 'fpr', 'tpr_malicious', 'fnr_malicious', 
                             'accuracy', 'composite_score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"   📊 Saved summary statistics: {filename}")

def main():
    """Main function to run the lambda comparison analysis."""
    print("🚀 Starting Lambda Results Comparison Analysis...")
    
    # Initialize comparator
    comparator = LambdaResultsComparator()
    
    # Scan for experiment folders
    if not comparator.scan_experiment_folders():
        print("❌ No experiments found. Make sure to run automated_test_lambda.py first!")
        return
    
    # Create comparison plots
    pdf_path = comparator.create_comparison_plots()
    
    print(f"\n{'='*60}")
    print("🎉 LAMBDA COMPARISON ANALYSIS COMPLETED!")
    print(f"📑 Report saved: {pdf_path}")
    print("📁 Plots saved in: lambda_comparison_plots/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
