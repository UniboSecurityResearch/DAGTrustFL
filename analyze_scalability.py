#!/usr/bin/env python3
"""
Scalability Analysis Script for DAGTrustFL
This script analyzes the performance data collected from scalability experiments
and generates comprehensive reports comparing timing and performance across different K values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Set the font to Roboto
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.size'] = 12

def load_timing_data(results_dir):
    """Load timing data from all scalability experiments"""
    results_path = Path(results_dir)
    timing_data = {}
    
    # Find all K_* folders
    for folder in results_path.glob("K_*_scalability"):
        # Extract K value from folder name
        k_str = folder.name.split('_')[1]
        try:
            k_value = int(k_str)
        except ValueError:
            continue
        
        # Look for timing results
        timing_folder = folder / 'timing_results'
        if timing_folder.exists():
            # Find timing summary CSV
            summary_files = list(timing_folder.glob("*_timing_summary.csv"))
            if summary_files:
                try:
                    # Read the metric,value format CSV
                    df = pd.read_csv(summary_files[0])
                    
                    # Convert from metric,value format to a proper DataFrame
                    timing_dict = {}
                    for _, row in df.iterrows():
                        timing_dict[row['metric']] = row['value']
                    
                    # Create a single-row DataFrame with the timing data
                    timing_df = pd.DataFrame([timing_dict])
                    timing_data[k_value] = timing_df
                    print(f"Loaded timing data for K={k_value}")
                except Exception as e:
                    print(f"Error loading timing data for K={k_value}: {e}")
    
    return timing_data

def load_dag_analysis_data(results_dir):
    """Load DAG analysis data from all scalability experiments"""
    results_path = Path(results_dir)
    dag_data = {}
    
    for folder in results_path.glob("K_*_scalability"):
        k_str = folder.name.split('_')[1]
        try:
            k_value = int(k_str)
        except ValueError:
            continue
        
        dag_file = folder / 'dag_trust_analysis.csv'
        if dag_file.exists():
            try:
                df = pd.read_csv(dag_file)
                dag_data[k_value] = df
                print(f"Loaded DAG analysis data for K={k_value}")
            except Exception as e:
                print(f"Error loading DAG data for K={k_value}: {e}")
    
    return dag_data

def load_fl_results_data(results_dir):
    """Load FL results data from all scalability experiments"""
    results_path = Path(results_dir)
    fl_data = {}
    
    for folder in results_path.glob("K_*_scalability"):
        k_str = folder.name.split('_')[1]
        try:
            k_value = int(k_str)
        except ValueError:
            continue
        
        fl_file = folder / '3000_results.csv'
        if fl_file.exists():
            try:
                df = pd.read_csv(fl_file)
                fl_data[k_value] = df
                print(f"Loaded FL results data for K={k_value}")
            except Exception as e:
                print(f"Error loading FL data for K={k_value}: {e}")
    
    return fl_data

def plot_timing_scalability(timing_data, output_dir):
    """Generate timing scalability plots"""
    if not timing_data:
        print("No timing data available for plotting")
        return
    
    k_values = sorted(timing_data.keys())
    ml_times = []
    dag_times = []
    total_times = []
    
    for k in k_values:
        df = timing_data[k]
        if not df.empty:
            # Use average times PER ROUND for scalability analysis
            ml_time = df['avg_ml_training_time_per_round'].iloc[0] if 'avg_ml_training_time_per_round' in df.columns else 0
            dag_time = df['avg_dag_operations_time_per_round'].iloc[0] if 'avg_dag_operations_time_per_round' in df.columns else 0
            total_time = df['avg_total_round_time'].iloc[0] if 'avg_total_round_time' in df.columns else ml_time + dag_time
            
            ml_times.append(ml_time)
            dag_times.append(dag_time)
            total_times.append(total_time)
        else:
            ml_times.append(0)
            dag_times.append(0)
            total_times.append(0)
    
    # Create timing scalability plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute timing
    ax1.plot(k_values, ml_times, 'o-', label='ML Training Time per Round', linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(k_values, dag_times, 's-', label='DAG Operation Time per Round', linewidth=2, markersize=8, color='#A23B72')
    ax1.plot(k_values, total_times, '^-', label='Total Time per Round', linewidth=2, markersize=8, color='#F18F01')
    
    ax1.set_xlabel('Number of Participants (K)')
    ax1.set_ylabel('Time per Round (seconds)')
    ax1.set_title('Average Time per Round vs Number of Participants')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Plot 2: Time ratio and percentages
    if ml_times and dag_times:
        ratios = [dag_times[i] / ml_times[i] if ml_times[i] > 0 else 0 for i in range(len(k_values))]
        ax2.plot(k_values, ratios, 'o-', linewidth=2, markersize=8, color='#C73E1D')
        ax2.set_xlabel('Number of Participants (K)')
        ax2.set_ylabel('DAG Time / ML Time Ratio per Round')
        ax2.set_title('DAG vs ML Time Ratio per Round')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
    
    plt.tight_layout()
    output_path = output_dir / 'timing_scalability_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Timing scalability plot saved to: {output_path}")
    
    # Create additional plot with time percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Time per round
    avg_ml_per_round = []
    avg_dag_per_round = []
    
    for k in k_values:
        df = timing_data[k]
        if not df.empty:
            ml_avg = df['avg_ml_training_time_per_round'].iloc[0] if 'avg_ml_training_time_per_round' in df.columns else 0
            dag_avg = df['avg_dag_operations_time_per_round'].iloc[0] if 'avg_dag_operations_time_per_round' in df.columns else 0
            avg_ml_per_round.append(ml_avg)
            avg_dag_per_round.append(dag_avg)
        else:
            avg_ml_per_round.append(0)
            avg_dag_per_round.append(0)
    
    ax1.plot(k_values, avg_ml_per_round, 'o-', label='Avg ML Time per Round', linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(k_values, avg_dag_per_round, 's-', label='Avg DAG Time per Round', linewidth=2, markersize=8, color='#A23B72')
    ax1.set_xlabel('Number of Participants (K)')
    ax1.set_ylabel('Time per Round (seconds)')
    ax1.set_title('Average Time per Round')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Plot 2: Time percentages stacked bar
    ml_percentages = []
    dag_percentages = []
    
    for k in k_values:
        df = timing_data[k]
        if not df.empty:
            ml_pct = df['ml_training_percentage'].iloc[0] if 'ml_training_percentage' in df.columns else 0
            dag_pct = df['dag_operations_percentage'].iloc[0] if 'dag_operations_percentage' in df.columns else 0
            ml_percentages.append(ml_pct)
            dag_percentages.append(dag_pct)
        else:
            ml_percentages.append(0)
            dag_percentages.append(0)
    
    width = 0.6
    ax2.bar(k_values, ml_percentages, width, label='ML Training %', color='#2E86AB', alpha=0.8)
    ax2.bar(k_values, dag_percentages, width, bottom=ml_percentages, label='DAG Operations %', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Number of Participants (K)')
    ax2.set_ylabel('Percentage of Total Time')
    ax2.set_title('Time Distribution by Component')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(k_values)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path2 = output_dir / 'timing_efficiency_analysis.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Timing efficiency plot saved to: {output_path2}")
    
    # Print timing data for debugging
    print("\nTiming Data Summary (per Round):")
    print("K\tML Time (s)\tDAG Time (s)\tTotal Time (s)\tDAG/ML Ratio\tML %\tDAG %")
    print("-" * 85)
    for i, k in enumerate(k_values):
        ratio = dag_times[i] / ml_times[i] if ml_times[i] > 0 else 0
        ml_pct = ml_percentages[i] if i < len(ml_percentages) else 0
        dag_pct = dag_percentages[i] if i < len(dag_percentages) else 0
        print(f"{k}\t{ml_times[i]:.1f}\t\t{dag_times[i]:.1f}\t\t{total_times[i]:.1f}\t\t{ratio:.3f}\t\t{ml_pct:.1f}%\t{dag_pct:.1f}%")

def plot_performance_scalability(dag_data, fl_data, output_dir):
    """Generate performance scalability plots"""
    if not dag_data and not fl_data:
        print("No performance data available for plotting")
        return
    
    k_values = sorted(set(list(dag_data.keys()) + list(fl_data.keys())))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Trust detection metrics from DAG data
    if dag_data:
        total_malicious_detected = []
        total_corrupted_detected = []
        false_positives = []
        
        for k in k_values:
            if k in dag_data:
                df = dag_data[k]
                if not df.empty:
                    mal_detected = df['Total_Malicious_Confirmed'].iloc[-1] if 'Total_Malicious_Confirmed' in df.columns else 0
                    cor_detected = df['Total_Corrupted_Confirmed'].iloc[-1] if 'Total_Corrupted_Confirmed' in df.columns else 0
                    good_removed = df['Total_Good_Removed'].iloc[-1] if 'Total_Good_Removed' in df.columns else 0
                    
                    total_malicious_detected.append(mal_detected)
                    total_corrupted_detected.append(cor_detected)
                    false_positives.append(good_removed)
                else:
                    total_malicious_detected.append(0)
                    total_corrupted_detected.append(0)
                    false_positives.append(0)
            else:
                total_malicious_detected.append(0)
                total_corrupted_detected.append(0)
                false_positives.append(0)
        
        ax1.plot(k_values, total_malicious_detected, 'o-', label='Malicious Detected', linewidth=2, markersize=8)
        ax1.plot(k_values, total_corrupted_detected, 's-', label='Corrupted Detected', linewidth=2, markersize=8)
        ax1.plot(k_values, false_positives, '^-', label='False Positives', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Participants (K)')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title('Trust Detection Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
    
    # Plot 2: FL accuracy from FL data
    if fl_data:
        final_accuracies = []
        
        for k in k_values:
            if k in fl_data:
                df = fl_data[k]
                if not df.empty and 'accuracy' in df.columns:
                    final_acc = df['accuracy'].iloc[-1]
                    final_accuracies.append(final_acc)
                else:
                    final_accuracies.append(0)
            else:
                final_accuracies.append(0)
        
        ax2.plot(k_values, final_accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax2.set_xlabel('Number of Participants (K)')
        ax2.set_ylabel('Final Accuracy')
        ax2.set_title('Final Model Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
    
    # Plot 3: Communication overhead (total nodes in DAG)
    if dag_data:
        total_nodes = []
        
        for k in k_values:
            if k in dag_data:
                df = dag_data[k]
                if not df.empty and 'Total_Nodes' in df.columns:
                    max_nodes = df['Total_Nodes'].max()
                    total_nodes.append(max_nodes)
                else:
                    total_nodes.append(0)
            else:
                total_nodes.append(0)
        
        ax3.plot(k_values, total_nodes, 'o-', linewidth=2, markersize=8, color='#A23B72')
        ax3.set_xlabel('Number of Participants (K)')
        ax3.set_ylabel('Maximum DAG Nodes')
        ax3.set_title('Communication Overhead (DAG Size)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(k_values)
    
    # Plot 4: Scalability efficiency (time per participant)
    if dag_data:
        time_per_participant = []
        
        for k in k_values:
            if k in dag_data:
                df = dag_data[k]
                if not df.empty:
                    num_rounds = len(df)
                    # Estimate time per participant (this is a rough estimate)
                    time_estimate = num_rounds * k * 0.1  # Rough scaling
                    time_per_participant.append(time_estimate / k)
                else:
                    time_per_participant.append(0)
            else:
                time_per_participant.append(0)
        
        ax4.plot(k_values, time_per_participant, 'o-', linewidth=2, markersize=8, color='#F18F01')
        ax4.set_xlabel('Number of Participants (K)')
        ax4.set_ylabel('Estimated Time per Participant')
        ax4.set_title('Time Efficiency per Participant')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(k_values)
    
    plt.tight_layout()
    output_path = output_dir / 'performance_scalability_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance scalability plot saved to: {output_path}")

def plot_dag_scaling_linearity(timing_data, output_dir):
    """Generate DAG scaling linearity analysis plot with blockchain comparison"""
    if not timing_data:
        print("No timing data available for DAG scaling analysis")
        return
    
    k_values = sorted(timing_data.keys())
    dag_times = []
    ml_times = []
    
    for k in k_values:
        df = timing_data[k]
        if not df.empty:
            # Use average times PER ROUND for proper scaling comparison
            dag_time = df['avg_dag_operations_time_per_round'].iloc[0] if 'avg_dag_operations_time_per_round' in df.columns else 0
            ml_time = df['avg_ml_training_time_per_round'].iloc[0] if 'avg_ml_training_time_per_round' in df.columns else 0
            dag_times.append(dag_time)
            ml_times.append(ml_time)
        else:
            dag_times.append(0)
            ml_times.append(0)
    
    if len(k_values) < 2:
        print("Need at least 2 K values for scaling analysis")
        return
    
    # Calculate blockchain theoretical times
    def estimate_blockchain_sync_time(k, base_time, base_k):
        """
        Estimate blockchain synchronous consensus time based on K participants
        
        Blockchain characteristics:
        - Sequential block creation (no parallelism)
        - Consensus overhead grows with participants
        - Network communication: O(K²) for full connectivity
        - Validation time: O(K) for signature verification
        """
        if base_time <= 0 or base_k <= 0:
            return 0
            
        # Base components for blockchain
        block_creation_time = base_time * 0.3  # Constant per block
        consensus_overhead = base_time * 0.4 * (k / base_k)  # Linear with participants
        network_comm = base_time * 0.3 * ((k * k) / (base_k * base_k))  # Quadratic for full mesh
        
        return block_creation_time + consensus_overhead + network_comm
    
    # Create the scaling analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: DAG Time vs K with linear reference and blockchain comparison
    ax1.plot(k_values, dag_times, 'o-', label='DAG Time per Round (Actual)', linewidth=3, markersize=10, color='#A23B72')
    
    # Calculate linear scaling reference based on first data point
    if dag_times[0] > 0:
        base_k = k_values[0]
        base_time = dag_times[0]
        linear_reference = [base_time * (k / base_k) for k in k_values]
        ax1.plot(k_values, linear_reference, '--', label='Linear Scaling Reference', linewidth=2, color='#E63946', alpha=0.8)
        
        # Calculate quadratic scaling reference
        quadratic_reference = [base_time * ((k / base_k) ** 2) for k in k_values]
        ax1.plot(k_values, quadratic_reference, ':', label='Quadratic Scaling Reference', linewidth=2, color='#F77F00', alpha=0.8)
        
        # Calculate blockchain synchronous estimation
        blockchain_times = [estimate_blockchain_sync_time(k, base_time, base_k) for k in k_values]
        ax1.plot(k_values, blockchain_times, 's-', label='Blockchain Sync per Round (Theoretical)', linewidth=2, markersize=8, color='#2A9D8F', alpha=0.9)
    
    ax1.set_xlabel('Number of Participants (K)', fontsize=12)
    ax1.set_ylabel('Time per Round (seconds)', fontsize=12)
    ax1.set_title('DAG vs Blockchain Scaling per Round', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Plot 2: Relative performance (speedup factor)
    if dag_times[0] > 0:
        base_k = k_values[0]
        base_time = dag_times[0]
        blockchain_times = [estimate_blockchain_sync_time(k, base_time, base_k) for k in k_values]
        
        # Calculate speedup factors (blockchain_time / dag_time)
        speedup_factors = []
        blockchain_dag_ratios = []
        
        for i, k in enumerate(k_values):
            if dag_times[i] > 0:
                speedup = blockchain_times[i] / dag_times[i]
                speedup_factors.append(speedup)
                blockchain_dag_ratios.append(blockchain_times[i] / dag_times[i])
            else:
                speedup_factors.append(1)
                blockchain_dag_ratios.append(1)
        
        ax2.plot(k_values, speedup_factors, 'o-', linewidth=3, markersize=10, color='#2A9D8F')
        ax2.axhline(y=1, color='#E63946', linestyle='--', alpha=0.7, label='No difference')
        
        ax2.set_xlabel('Number of Participants (K)', fontsize=12)
        ax2.set_ylabel('Blockchain Time / DAG Time Ratio', fontsize=12)
        ax2.set_title('DAG Performance Advantage vs Blockchain', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        ax2.legend(fontsize=10)
        
        # Add text annotations for key insights
        if len(speedup_factors) > 1:
            max_speedup = max(speedup_factors)
            max_speedup_k = k_values[speedup_factors.index(max_speedup)]
            ax2.annotate(f'Max advantage: {max_speedup:.1f}x\nat K={max_speedup_k}', 
                        xy=(max_speedup_k, max_speedup), xytext=(max_speedup_k + 5, max_speedup + 0.5),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / 'dag_vs_blockchain_scaling_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"DAG vs Blockchain scaling comparison plot saved to: {output_path}")
    
    # Calculate and print scaling factors with blockchain comparison
    print("\nDAG vs Blockchain Scaling Analysis (per Round):")
    print("="*60)
    
    if len(k_values) >= 2 and dag_times[0] > 0 and dag_times[-1] > 0:
        base_k = k_values[0]
        base_time = dag_times[0]
        
        k_factor = k_values[-1] / k_values[0]
        dag_factor = dag_times[-1] / dag_times[0]
        
        # Calculate final blockchain time
        final_blockchain_time = estimate_blockchain_sync_time(k_values[-1], base_time, base_k)
        blockchain_factor = final_blockchain_time / base_time if base_time > 0 else 0
        
        print(f"K scaling factor: {k_factor:.2f}x (from {k_values[0]} to {k_values[-1]})")
        print(f"DAG time per round scaling: {dag_factor:.2f}x")
        print(f"Blockchain time per round scaling (theoretical): {blockchain_factor:.2f}x")
        print(f"DAG scaling efficiency: {k_factor/dag_factor:.3f} (1.0 = perfect linear)")
        print(f"Blockchain scaling efficiency: {k_factor/blockchain_factor:.3f} (1.0 = perfect linear)")
        
        # Performance advantage
        initial_advantage = blockchain_times[0] / dag_times[0] if dag_times[0] > 0 else 1
        final_advantage = final_blockchain_time / dag_times[-1] if dag_times[-1] > 0 else 1
        
        print(f"\nPerformance Advantage per Round (Blockchain time / DAG time):")
        print(f"At K={k_values[0]}: {initial_advantage:.2f}x")
        print(f"At K={k_values[-1]}: {final_advantage:.2f}x")
        print(f"Advantage growth: {final_advantage/initial_advantage:.2f}x")
        
        # Show actual times for clarity
        print(f"\nActual Times per Round:")
        print(f"At K={k_values[0]}: DAG={dag_times[0]:.1f}s, Blockchain≈{blockchain_times[0]:.1f}s")
        print(f"At K={k_values[-1]}: DAG={dag_times[-1]:.1f}s, Blockchain≈{final_blockchain_time:.1f}s")
        
        # Determine scaling behavior
        dag_linearity_ratio = dag_factor / k_factor
        blockchain_linearity_ratio = blockchain_factor / k_factor
        
        print(f"\nScaling Behavior Comparison:")
        print(f"DAG Operations:")
        if dag_linearity_ratio < 0.8:
            print("  ✅ Better than linear scaling (sublinear)")
        elif dag_linearity_ratio <= 1.2:
            print("  ✅ Approximately linear scaling")
        elif dag_linearity_ratio <= 2.0:
            print("  ⚠️  Superlinear scaling (concerning)")
        else:
            print("  ❌ Poor scaling (possibly quadratic or worse)")
        
        print(f"Blockchain Sync (Theoretical):")
        if blockchain_linearity_ratio < 0.8:
            print("  ✅ Better than linear scaling (sublinear)")
        elif blockchain_linearity_ratio <= 1.2:
            print("  ✅ Approximately linear scaling")
        elif blockchain_linearity_ratio <= 2.0:
            print("  ⚠️  Superlinear scaling")
        else:
            print("  ❌ Poor scaling (quadratic or worse)")
        
        print(f"\n🎯 Key Insight:")
        if final_advantage > initial_advantage:
            print(f"   DAG advantage INCREASES with scale ({final_advantage:.1f}x faster at K={k_values[-1]})")
        else:
            print(f"   DAG advantage DECREASES with scale ({final_advantage:.1f}x faster at K={k_values[-1]})")
        
        print("="*60)

def generate_scalability_report(timing_data, dag_data, fl_data, output_dir):
    """Generate a comprehensive scalability report"""
    report_path = output_dir / 'scalability_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("DAGTrustFL Scalability Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. TIMING ANALYSIS\n")
        f.write("-" * 20 + "\n")
        if timing_data:
            k_values = sorted(timing_data.keys())
            f.write(f"Tested K values: {k_values}\n\n")
            
            f.write("Timing Results (Average per Round):\n")
            f.write("K\tML Time (s)\tDAG Time (s)\tTotal Time (s)\tDAG/ML Ratio\tML %\tDAG %\n")
            f.write("-" * 80 + "\n")
            for k in k_values:
                df = timing_data[k]
                if not df.empty:
                    ml_time = df['avg_ml_training_time_per_round'].iloc[0] if 'avg_ml_training_time_per_round' in df.columns else 0
                    dag_time = df['avg_dag_operations_time_per_round'].iloc[0] if 'avg_dag_operations_time_per_round' in df.columns else 0
                    total_time = df['avg_total_round_time'].iloc[0] if 'avg_total_round_time' in df.columns else ml_time + dag_time
                    ml_pct = df['ml_training_percentage'].iloc[0] if 'ml_training_percentage' in df.columns else 0
                    dag_pct = df['dag_operations_percentage'].iloc[0] if 'dag_operations_percentage' in df.columns else 0
                    ratio = dag_time / ml_time if ml_time > 0 else 0
                    f.write(f"{k}\t{ml_time:.1f}\t\t{dag_time:.1f}\t\t{total_time:.1f}\t\t{ratio:.3f}\t\t{ml_pct:.1f}%\t{dag_pct:.1f}%\n")
        else:
            f.write("No timing data available\n")
        
        f.write("\n2. PERFORMANCE ANALYSIS\n")
        f.write("-" * 25 + "\n")
        if dag_data:
            f.write("Trust Detection Performance:\n")
            f.write("K\tMalicious Detected\tCorrupted Detected\tFalse Positives\n")
            for k in sorted(dag_data.keys()):
                df = dag_data[k]
                if not df.empty:
                    mal_detected = df['Total_Malicious_Confirmed'].iloc[-1] if 'Total_Malicious_Confirmed' in df.columns else 0
                    cor_detected = df['Total_Corrupted_Confirmed'].iloc[-1] if 'Total_Corrupted_Confirmed' in df.columns else 0
                    good_removed = df['Total_Good_Removed'].iloc[-1] if 'Total_Good_Removed' in df.columns else 0
                    f.write(f"{k}\t{mal_detected}\t\t{cor_detected}\t\t{good_removed}\n")
        
        if fl_data:
            f.write("\nFL Accuracy Performance:\n")
            f.write("K\tFinal Accuracy\n")
            for k in sorted(fl_data.keys()):
                df = fl_data[k]
                if not df.empty and 'accuracy' in df.columns:
                    final_acc = df['accuracy'].iloc[-1]
                    f.write(f"{k}\t{final_acc:.4f}\n")
        
        f.write("\n3. SCALABILITY INSIGHTS\n")
        f.write("-" * 25 + "\n")
        
        if timing_data and len(timing_data) > 1:
            k_values = sorted(timing_data.keys())
            ml_times = []
            dag_times = []
            
            for k in k_values:
                df = timing_data[k]
                ml_time = df['avg_ml_training_time_per_round'].iloc[0] if 'avg_ml_training_time_per_round' in df.columns else 0
                dag_time = df['avg_dag_operations_time_per_round'].iloc[0] if 'avg_dag_operations_time_per_round' in df.columns else 0
                ml_times.append(ml_time)
                dag_times.append(dag_time)
            
            # Calculate scaling factors
            if ml_times[0] > 0 and dag_times[0] > 0:
                ml_scaling = ml_times[-1] / ml_times[0]
                dag_scaling = dag_times[-1] / dag_times[0]
                k_scaling = k_values[-1] / k_values[0]
                
                f.write(f"K scaling factor: {k_scaling:.1f}x (from {k_values[0]} to {k_values[-1]})\n")
                f.write(f"ML time scaling: {ml_scaling:.1f}x\n")
                f.write(f"DAG time scaling: {dag_scaling:.1f}x\n")
                f.write(f"ML efficiency: {k_scaling/ml_scaling:.2f} (closer to 1.0 = better linear scaling)\n")
                f.write(f"DAG efficiency: {k_scaling/dag_scaling:.2f} (closer to 1.0 = better linear scaling)\n")
                
                # Additional insights
                if ml_scaling < k_scaling:
                    f.write("ML training shows better than linear scaling (good parallelization)\n")
                elif ml_scaling > k_scaling * 1.5:
                    f.write("ML training shows concerning non-linear scaling\n")
                
                if dag_scaling < k_scaling:
                    f.write("DAG operations show better than linear scaling\n")
                elif dag_scaling > k_scaling * 1.5:
                    f.write("DAG operations show concerning non-linear scaling\n")
                
                # Additional detailed scaling analysis
                f.write(f"\n4. DETAILED SCALING ANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                dag_linearity_ratio = dag_scaling / k_scaling
                ml_linearity_ratio = ml_scaling / k_scaling
                
                f.write(f"DAG linearity ratio: {dag_linearity_ratio:.3f} (1.0 = perfect linear)\n")
                f.write(f"ML linearity ratio: {ml_linearity_ratio:.3f} (1.0 = perfect linear)\n")
                
                # Blockchain theoretical comparison
                f.write(f"\n5. BLOCKCHAIN COMPARISON (THEORETICAL)\n")
                f.write("-" * 40 + "\n")
                
                # Calculate blockchain theoretical scaling
                base_k = k_values[0]
                base_time = ml_times[0] if ml_times[0] > 0 else dag_times[0]
                
                def estimate_blockchain_sync_time(k, base_time, base_k):
                    if base_time <= 0 or base_k <= 0:
                        return 0
                    block_creation_time = base_time * 0.3
                    consensus_overhead = base_time * 0.4 * (k / base_k)
                    network_comm = base_time * 0.3 * ((k * k) / (base_k * base_k))
                    return block_creation_time + consensus_overhead + network_comm
                
                blockchain_final_time = estimate_blockchain_sync_time(k_values[-1], base_time, base_k)
                blockchain_scaling = blockchain_final_time / base_time if base_time > 0 else 0
                blockchain_linearity = blockchain_scaling / k_scaling
                
                f.write(f"Blockchain theoretical scaling: {blockchain_scaling:.2f}x\n")
                f.write(f"Blockchain linearity ratio: {blockchain_linearity:.3f}\n")
                
                # Performance advantage
                dag_final_time = dag_times[-1] if dag_times[-1] > 0 else ml_times[-1]
                if dag_final_time > 0:
                    advantage = blockchain_final_time / dag_final_time
                    f.write(f"DAG performance advantage at K={k_values[-1]}: {advantage:.2f}x faster than blockchain\n")
                
                f.write("\nScaling Classification:\n")
                if dag_linearity_ratio < 0.8:
                    f.write("DAG: Sublinear scaling (excellent)\n")
                elif dag_linearity_ratio <= 1.2:
                    f.write("DAG: Linear scaling (good)\n")
                elif dag_linearity_ratio <= 2.0:
                    f.write("DAG: Superlinear scaling (concerning)\n")
                else:
                    f.write("DAG: Poor scaling (quadratic or worse)\n")
                
                if blockchain_linearity < 0.8:
                    f.write("Blockchain: Sublinear scaling (excellent)\n")
                elif blockchain_linearity <= 1.2:
                    f.write("Blockchain: Linear scaling (good)\n")
                elif blockchain_linearity <= 2.0:
                    f.write("Blockchain: Superlinear scaling (concerning)\n")
                else:
                    f.write("Blockchain: Poor scaling (quadratic or worse)\n")
                
                if ml_linearity_ratio < 0.8:
                    f.write("ML: Sublinear scaling (excellent)\n")
                elif ml_linearity_ratio <= 1.2:
                    f.write("ML: Linear scaling (good)\n")
                elif ml_linearity_ratio <= 2.0:
                    f.write("ML: Superlinear scaling (concerning)\n")
                else:
                    f.write("ML: Poor scaling (quadratic or worse)\n")
        
        f.write(f"\nReport generated on: {pd.Timestamp.now()}\n")
    
    print(f"Scalability report saved to: {report_path}")

def main():
    """Main analysis function"""
    results_dir = '/home/giac/DAGTrustFL/DAGTrustFL/FL/results_scalability'
    output_dir = Path(results_dir) / 'analysis_plots'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading scalability experiment data...")
    
    # Load all data
    timing_data = load_timing_data(results_dir)
    dag_data = load_dag_analysis_data(results_dir)
    fl_data = load_fl_results_data(results_dir)
    
    if not timing_data and not dag_data and not fl_data:
        print("No data found in results_scalability folder. Please run experiments first.")
        return
    
    print(f"\nGenerating scalability analysis plots...")
    
    # Generate plots
    plot_timing_scalability(timing_data, output_dir)
    plot_performance_scalability(dag_data, fl_data, output_dir)
    plot_dag_scaling_linearity(timing_data, output_dir)
    
    # Generate comprehensive report
    generate_scalability_report(timing_data, dag_data, fl_data, output_dir)
    
    print(f"\n✓ Scalability analysis complete!")
    print(f"Analysis results saved in: {output_dir}")

if __name__ == '__main__':
    main()
