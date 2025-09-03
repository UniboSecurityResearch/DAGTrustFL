#!/usr/bin/env python3
"""
Theta Results Comparator for DAGTrustFL
=======================================

This script compares and analyzes results from different Theta experiments.
It creates comprehensive plots and analysis reports for Theta parameter testing.

Theta Parameter:
- Controls the confirmation threshold in the DAG structure
- Higher Theta values require more cumulative weight for confirmation
- Values tested: [2, 3, 4, 5]

Author: DAGTrustFL Analysis Suite
Date: 2025-08-30
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

class ThetaResultsComparator:
    """Class for comparing results from different Theta experiments."""
    
    def __init__(self, results_base_path="FL/results_theta"):
        """
        Initialize the comparator.
        
        Args:
            results_base_path (str): Base path where Theta results are stored
        """
        self.results_base_path = Path(results_base_path)
        self.output_dir = "theta_comparison_plots"
        self.experiments = {}
        
        # Create directory for plots
        os.makedirs(self.output_dir, exist_ok=True)
        
    def scan_experiment_folders(self):
        """Scan for experiment folders and categorize them."""
        print("🔍 Scanning for Theta experiment folders...")
        
        if not self.results_base_path.exists():
            print(f"❌ Results path not found: {self.results_base_path}")
            return False
        
        # Pattern: theta_X (e.g., theta_3 for Θ=3)
        pattern = r'theta_(\d+)'
        
        for folder in self.results_base_path.iterdir():
            if folder.is_dir():
                match = re.match(pattern, folder.name)
                if match:
                    # Convert folder name back to integer value
                    theta_value = int(match.group(1))
                    
                    # Look for the CSV file in this folder
                    csv_file = folder / "dag_trust_analysis.csv"
                    if csv_file.exists():
                        self.experiments[theta_value] = {
                            'folder': folder,
                            'csv_path': csv_file
                        }
                        print(f"   ✓ Found Theta {theta_value}: {folder.name}")
                    else:
                        print(f"   ⚠️ No CSV file in {folder.name}")
        
        if not self.experiments:
            print("❌ No valid experiment folders found!")
            return False
        
        print(f"\n📊 Found {len(self.experiments)} Theta experiments:")
        theta_values = sorted(self.experiments.keys())
        print(f"   • Theta values: {theta_values}")
        
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
        """Create comparison plots for all Theta experiments."""
        pdf_filename = f"{self.output_dir}/theta_comparison_report.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            print("📑 Creating Theta comparison PDF...")
            
            # Load data for all Theta values
            theta_values = []
            metrics_data = {
                'fpr': [],
                'tpr_malicious': [],
                'fnr_malicious': [],
                'tpr_corrupted': [],
                'fnr_corrupted': [],
                'accuracy': []
            }
            
            for theta_val in sorted(self.experiments.keys()):
                data = self.load_experiment_data(self.experiments[theta_val]['csv_path'])
                if data:
                    theta_values.append(theta_val)
                    metrics_data['fpr'].append(data['fpr'])
                    metrics_data['tpr_malicious'].append(data['tpr_malicious'])
                    metrics_data['fnr_malicious'].append(data['fnr_malicious'])
                    metrics_data['tpr_corrupted'].append(data['tpr_corrupted'])
                    metrics_data['fnr_corrupted'].append(data['fnr_corrupted'])
                    metrics_data['accuracy'].append(data['accuracy'])
            
            if not theta_values:
                print("   ⚠️ No valid data found")
                return pdf_filename
            
            # Create plots for Theta comparison
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
            
            # 1. False Positive Rate (FPR)
            ax1.plot(theta_values, metrics_data['fpr'], 'o-', linewidth=3, markersize=8, 
                    color='orange', label='FPR')
            ax1.set_xlabel('Theta Value (Θ)')
            ax1.set_ylabel('False Positive Rate (%)')
            ax1.set_title('False Positive Rate vs Theta\n(Good nodes incorrectly removed)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(100, max(metrics_data['fpr']) * 1.1) if metrics_data['fpr'] else 100)
            
            # Add value labels on points
            for i, (theta, fpr) in enumerate(zip(theta_values, metrics_data['fpr'])):
                ax1.annotate(f'{fpr:.1f}%', (theta, fpr), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            # 2. True Positive Rate (TPR) for Malicious Detection
            ax2.plot(theta_values, metrics_data['tpr_malicious'], 's-', linewidth=3, markersize=8, 
                    color='red', label='Malicious TPR')
            ax2.set_xlabel('Theta Value (Θ)')
            ax2.set_ylabel('True Positive Rate (%)')
            ax2.set_title('Malicious Detection Rate vs Theta\n(Malicious nodes correctly rejected)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(100, max(metrics_data['tpr_malicious']) * 1.1) if metrics_data['tpr_malicious'] else 100)
            
            # Add value labels on points
            for i, (theta, tpr) in enumerate(zip(theta_values, metrics_data['tpr_malicious'])):
                ax2.annotate(f'{tpr:.1f}%', (theta, tpr), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            # 3. False Negative Rate (FNR) Comparison
            ax3.plot(theta_values, metrics_data['fnr_malicious'], 's-', linewidth=3, markersize=8, 
                    color='red', label='FNR Malicious')
            if any(val > 0 for val in metrics_data['fnr_corrupted']):
                ax3.plot(theta_values, metrics_data['fnr_corrupted'], '^-', linewidth=3, markersize=8, 
                        color='purple', label='FNR Corrupted')
            ax3.set_xlabel('Theta Value (Θ)')
            ax3.set_ylabel('False Negative Rate (%)')
            ax3.set_title('False Negative Rate vs Theta\n(Malicious/Corrupted nodes incorrectly confirmed)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            max_fnr = max(max(metrics_data['fnr_malicious']) if metrics_data['fnr_malicious'] else [0], 
                         max(metrics_data['fnr_corrupted']) if metrics_data['fnr_corrupted'] else [0])
            ax3.set_ylim(0, max(100, max_fnr * 1.1) if max_fnr > 0 else 100)
            
            # Add value labels on points
            for i, (theta, fnr_mal) in enumerate(zip(theta_values, metrics_data['fnr_malicious'])):
                ax3.annotate(f'{fnr_mal:.1f}%', (theta, fnr_mal), textcoords="offset points", 
                           xytext=(0,10), ha='center', color='red')
            if any(val > 0 for val in metrics_data['fnr_corrupted']):
                for i, (theta, fnr_cor) in enumerate(zip(theta_values, metrics_data['fnr_corrupted'])):
                    ax3.annotate(f'{fnr_cor:.1f}%', (theta, fnr_cor), textcoords="offset points", 
                               xytext=(0,-15), ha='center', color='purple')
            
            # 4. Overall System Accuracy
            ax4.plot(theta_values, metrics_data['accuracy'], 'd-', linewidth=3, markersize=8, 
                    color='green', label='System Accuracy')
            ax4.set_xlabel('Theta Value (Θ)')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Overall System Accuracy vs Theta\n(Correctly handled nodes / Total nodes)')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, max(100, max(metrics_data['accuracy']) * 1.1) if metrics_data['accuracy'] else 100)
            
            # Add value labels on points
            for i, (theta, acc) in enumerate(zip(theta_values, metrics_data['accuracy'])):
                ax4.annotate(f'{acc:.1f}%', (theta, acc), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            # 5. TPR Comparison (Malicious vs Corrupted)
            ax5.plot(theta_values, metrics_data['tpr_malicious'], 's-', linewidth=3, markersize=8, 
                    color='red', label='TPR Malicious')
            if any(val > 0 for val in metrics_data['tpr_corrupted']):
                ax5.plot(theta_values, metrics_data['tpr_corrupted'], '^-', linewidth=3, markersize=8, 
                        color='purple', label='TPR Corrupted')
            ax5.set_xlabel('Theta Value (Θ)')
            ax5.set_ylabel('True Positive Rate (%)')
            ax5.set_title('Detection Rate Comparison vs Theta\n(Malicious vs Corrupted)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            max_tpr = max(max(metrics_data['tpr_malicious']) if metrics_data['tpr_malicious'] else [0], 
                         max(metrics_data['tpr_corrupted']) if metrics_data['tpr_corrupted'] else [0])
            ax5.set_ylim(0, max(100, max_tpr * 1.1) if max_tpr > 0 else 100)
            
            # 6. Combined view with all metrics
            ax6.plot(theta_values, metrics_data['fpr'], 'o-', linewidth=2, markersize=6, 
                    color='orange', label='FPR (Good removed)')
            ax6.plot(theta_values, metrics_data['tpr_malicious'], '^-', linewidth=2, markersize=6, 
                    color='darkred', label='TPR Malicious')
            ax6.plot(theta_values, metrics_data['fnr_malicious'], 's-', linewidth=2, markersize=6, 
                    color='red', label='FNR Malicious')
            ax6.plot(theta_values, metrics_data['accuracy'], 'd-', linewidth=2, markersize=6, 
                    color='green', label='Overall Accuracy')
            
            # Add corrupted metrics if there are corrupted nodes
            if any(val > 0 for val in metrics_data['tpr_corrupted']):
                ax6.plot(theta_values, metrics_data['tpr_corrupted'], '^-', linewidth=2, markersize=6, 
                        color='purple', label='TPR Corrupted')
            if any(val > 0 for val in metrics_data['fnr_corrupted']):
                ax6.plot(theta_values, metrics_data['fnr_corrupted'], 'v-', linewidth=2, markersize=6, 
                        color='darkviolet', label='FNR Corrupted')
            
            ax6.set_xlabel('Theta Value (Θ)')
            ax6.set_ylabel('Rate (%)')
            ax6.set_title('Combined Metrics Overview vs Theta')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 100)
            
            plt.suptitle('Theta Comparison - Confirmation Threshold Analysis', 
                       fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Create Theta effectiveness ranking page
            print("   📊 Creating Theta effectiveness ranking...")
            self._create_theta_ranking(pdf, theta_values, metrics_data)
            
            # Create summary statistics page
            print("   📋 Creating summary statistics...")
            self._create_summary_page(pdf, theta_values, metrics_data)
        
        print(f"📑 Theta comparison PDF saved: {pdf_filename}")
        
        # Save data to CSV for further analysis
        csv_filename = self._save_data_to_csv()
        print(f"💾 Theta comparison data saved: {csv_filename}")
        
        return pdf_filename
    
    def _create_theta_ranking(self, pdf, theta_values, metrics_data):
        """Create Theta effectiveness ranking page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        ranking_text = "THETA EFFECTIVENESS RANKING\n"
        ranking_text += "="*40 + "\n\n"
        
        # Calculate composite score for each Theta
        theta_rankings = []
        for i, theta_val in enumerate(theta_values):
            fpr = metrics_data['fpr'][i]
            tpr = metrics_data['tpr_malicious'][i]
            fnr = metrics_data['fnr_malicious'][i]
            acc = metrics_data['accuracy'][i]
            
            # Composite score: Higher TPR and Accuracy are better, lower FPR and FNR are better
            score = tpr + acc - fpr - fnr
            theta_rankings.append((theta_val, score, fpr, tpr, fnr, acc))
        
        theta_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (theta_val, score, fpr, tpr, fnr, acc) in enumerate(theta_rankings, 1):
            ranking_text += f"{rank}. Theta = {theta_val}\n"
            ranking_text += f"   FPR: {fpr:.1f}%, TPR: {tpr:.1f}%, FNR: {fnr:.1f}%, Acc: {acc:.1f}%\n"
            ranking_text += f"   Score: {score:.1f}\n\n"
        
        ranking_text += "\nTHETA PARAMETER:\n"
        ranking_text += "The Theta parameter controls the confirmation threshold\n"
        ranking_text += "in the DAG structure. A node is confirmed when its\n"
        ranking_text += "cumulative weight (CW) exceeds Theta.\n"
        ranking_text += "Higher Theta values require more confirmations.\n"
        ranking_text += "\nTHETA VALUES TESTED:\n"
        ranking_text += f"Values: {sorted(theta_values)}\n"
        ranking_text += "Configuration: Testing confirmation threshold effects"
        
        ax.text(0.05, 0.95, ranking_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.title('Theta Effectiveness Ranking', fontsize=18, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_summary_page(self, pdf, theta_values, metrics_data):
        """Create summary statistics page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        summary_text = f"""
THETA COMPARISON ANALYSIS - SUMMARY
===================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results Path: {self.results_base_path}

CONFIGURATION ANALYZED:
• Confirmation Threshold Testing
• Theta values tested: {sorted(theta_values)}

DETAILED RESULTS:
"""
        
        for i, theta_val in enumerate(sorted(theta_values)):
            idx = theta_values.index(theta_val)
            fpr = metrics_data['fpr'][idx]
            tpr = metrics_data['tpr_malicious'][idx]
            fnr = metrics_data['fnr_malicious'][idx]
            acc = metrics_data['accuracy'][idx]
            
            summary_text += f"\\n• Theta {theta_val}:\\n"
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

THETA PARAMETER:
The Theta parameter controls the confirmation threshold in the DAG.
A node is confirmed when its cumulative weight (CW) > Theta:
• Theta = 2: Low threshold (easier confirmation)
• Theta = 3: Default threshold
• Theta = 4: High threshold (stricter confirmation)
• Theta = 5: Very high threshold (very strict)
• Higher Theta values may improve security but reduce throughput
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        
        plt.title('Summary Statistics', fontsize=18, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _save_data_to_csv(self):
        """Save all comparison data to CSV files for further analysis."""
        
        # Create CSV data directory
        csv_dir = Path(self.output_dir) / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        
        # Save Theta comparison data
        theta_filename = csv_dir / "theta_comparison_data.csv"
        
        # Collect data for all Theta values
        rows = []
        for theta_val in sorted(self.experiments.keys()):
            data = self.load_experiment_data(self.experiments[theta_val]['csv_path'])
            if data:
                rows.append({
                    'theta': theta_val,
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
            with open(theta_filename, 'w', newline='') as csvfile:
                fieldnames = rows[0].keys()
                writer = pd.DataFrame(rows)
                writer.to_csv(theta_filename, index=False)
                print(f"   💾 Saved data for {len(rows)} Theta values")

        
        # Save summary statistics CSV
        summary_filename = csv_dir / "theta_summary_statistics.csv"
        self._save_summary_statistics_csv(summary_filename, rows)
        
        return theta_filename
    
    def _save_summary_statistics_csv(self, filename, rows):
        """Save summary statistics to CSV."""
        
        # Calculate summary statistics and rankings
        summary_rows = []
        for row in rows:
            # Calculate composite score
            score = row['tpr_malicious'] + row['accuracy'] - row['fpr'] - row['fnr_malicious']
            
            summary_rows.append({
                'theta': row['theta'],
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
        print(f"   📊 Saved summary statistics for {len(summary_rows)} Theta values")

def main():
    """Main function to run the Theta comparison analysis."""
    print("🚀 Starting Theta Results Comparison Analysis")
    print("=" * 60)
    
    try:
        # Initialize the comparator
        comparator = ThetaResultsComparator()
        
        # Scan for experiment folders
        if not comparator.scan_experiment_folders():
            return 1
        
        # Create comparison plots and analysis
        pdf_file = comparator.create_comparison_plots()
        
        print("\n" + "=" * 60)
        print("✅ THETA COMPARISON ANALYSIS COMPLETED!")
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
