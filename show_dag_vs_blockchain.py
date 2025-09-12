#!/usr/bin/env python3
"""
DAG vs Blockchain Scaling Visualization
Interactive script that shows th    # Create the scaling analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))DAG vs Blockchain scaling comparison in real-time
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set the font to Roboto
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

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
                    print(f"✓ Loaded timing data for K={k_value}")
                except Exception as e:
                    print(f"❌ Error loading timing data for K={k_value}: {e}")
    
    return timing_data

def plot_dag_scaling_analysis(timing_data):
    """Generate and display DAG scaling complexity analysis interactively"""
    if not timing_data:
        print("❌ No timing data available for DAG scaling analysis")
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
        print("❌ Need at least 2 K values for scaling analysis")
        return
    
    # Create the first plot - DAG vs Complexity Classes
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot 1: DAG Time vs K with complexity references
    ax1.plot(k_values, dag_times, 'o-', label='Mean DAG-time/round', linewidth=3, markersize=10, color='#A23B72')
    
    # Calculate linear scaling reference based on first data point
    if dag_times[0] > 0:
        base_k = k_values[0]
        base_time = dag_times[0]
        linear_reference = [base_time * (k / base_k) for k in k_values]
        ax1.plot(k_values, linear_reference, '--', label='Linear Ref O(N)', linewidth=2, color='#70E639', alpha=0.8)
        
        # Calculate quadratic scaling reference
        quadratic_reference = [base_time * ((k / base_k) ** 2) for k in k_values]
        ax1.plot(k_values, quadratic_reference, ':', label='Quadratic Ref O(N$^2$)', linewidth=2, color='#F77F00', alpha=0.8)

        # Calculate exponential scaling reference (base^(k/base_k))
        exponential_reference = [base_time * (np.exp((k / base_k) - 1)) for k in k_values]
        ax1.plot(k_values, exponential_reference, '-.', label='Exponential Ref O(e$^N$)', linewidth=2, color='#FF006E', alpha=0.8)

    ax1.set_xlabel('Number of Participants (K)', fontsize=16)
    ax1.set_ylabel('Time per Round (seconds)', fontsize=16)
    ax1.set_ylim(0, 200)
    ax1.legend(fontsize=13, loc='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    plt.tight_layout()
    plt.show()
    
    # Create the second plot - Scaling Complexity Analysis (Log Scale)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
        # Plot 2: Scaling Complexity Comparison (O(N²) analysis)
    if dag_times[0] > 0:
        base_k = k_values[0]
        base_time = dag_times[0]
        
        # DAG actual scaling
        ax2.plot(k_values, dag_times, 'o-', label='DAG (Actual)', linewidth=3, markersize=10, color='#A23B72')
        
        # Linear O(N) reference
        linear_reference = [base_time * (k / base_k) for k in k_values]
        ax2.plot(k_values, linear_reference, '--', label='O(N) Linear', linewidth=2, color='#70E639', alpha=0.8)
        
        # Quadratic O(N²) reference  
        quadratic_reference = [base_time * ((k / base_k) ** 2) for k in k_values]
        ax2.plot(k_values, quadratic_reference, '-.', label='O(N$^2$) Quadratic', linewidth=3, color='#F77F00', alpha=0.9)
        
        # Exponential O(e^N) reference
        exponential_reference = [base_time * (np.exp((k / base_k) - 1)) for k in k_values]
        ax2.plot(k_values, exponential_reference, ':', label='O(e$^N$) Exponential', linewidth=3, color='#FF006E', alpha=0.9)
        
        # Log scale for better visualization of different complexity classes
        ax2.set_yscale('log')
        ax2.set_ylim(0.1, 200)
        
        ax2.set_xlabel('Number of Participants (K)', fontsize=16)
        ax2.set_ylabel('Time per Round (seconds) - Log Scale', fontsize=16)
        ax2.legend(fontsize=13, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        
        # Add complexity annotations closer to the lines
        ax2.text(k_values[-1] * 0.95, quadratic_reference[-1] * 1.01, 'O(N$^2$)', 
                fontsize=14, fontweight='bold', color='#F77F00', 
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.95))
        
        ax2.text(k_values[-1] * 0.95, linear_reference[-1] * 1.01, 'O(N)', 
                fontsize=14, fontweight='bold', color="#70E639",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.95))
        
        ax2.text(k_values[-1] * 0.95, exponential_reference[-1] * 0.99, 'O(e$^N$)', 
                fontsize=14, fontweight='bold', color='#FF006E',
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.95))
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis to console
    print("\n" + "="*70)
    print("🔗 DAG SCALING COMPLEXITY ANALYSIS (per Round)")
    print("="*70)
    
    if len(k_values) >= 2 and dag_times[0] > 0 and dag_times[-1] > 0:
        base_k = k_values[0]
        base_time = dag_times[0]
        
        k_factor = k_values[-1] / k_values[0]
        dag_factor = dag_times[-1] / dag_times[0]
        
        print(f"📈 Scaling Factors:")
        print(f"   K scaling: {k_factor:.2f}x (from {k_values[0]} to {k_values[-1]} participants)")
        print(f"   DAG time per round: {dag_factor:.2f}x")
        print(f"   DAG efficiency: {k_factor/dag_factor:.3f} (1.0 = perfect linear)")
        
        # Show actual times for clarity
        print(f"\n🕐 Actual Times per Round:")
        print(f"   At K={k_values[0]}: DAG={dag_times[0]:.1f}s")
        print(f"   At K={k_values[-1]}: DAG={dag_times[-1]:.1f}s")
        
        # Determine scaling behavior
        dag_linearity_ratio = dag_factor / k_factor
        
        # Calculate O(N²) comparison
        quadratic_factor = (k_values[-1] / k_values[0]) ** 2
        dag_vs_quadratic_ratio = dag_factor / quadratic_factor
        
        # Calculate exponential comparison
        exponential_factor = np.exp((k_values[-1] / k_values[0]) - 1)
        dag_vs_exponential_ratio = dag_factor / exponential_factor
        
        print(f"\n🎯 Scaling Behavior Analysis:")
        print(f"   Complexity Comparison (vs theoretical):")
        print(f"     Linear O(N):         {k_factor:.2f}x growth expected")
        print(f"     Quadratic O(N²):     {quadratic_factor:.2f}x growth expected")
        print(f"     Exponential O(e^N):  {exponential_factor:.1f}x growth expected")
        print(f"     DAG actual:          {dag_factor:.2f}x growth")
        
        print(f"\n   Efficiency vs Complexity Classes:")
        print(f"     DAG vs O(N²):     {dag_vs_quadratic_ratio:.4f} (lower is better)")
        print(f"     DAG vs O(e^N):    {dag_vs_exponential_ratio:.6f} (lower is better)")
        
        print(f"\n   DAG Operations:")
        if dag_linearity_ratio < 0.8:
            print("     ✅ Better than linear scaling (sublinear)")
        elif dag_linearity_ratio <= 1.2:
            print("     ✅ Approximately linear scaling")
        elif dag_linearity_ratio <= 2.0:
            print("     ⚠️  Superlinear scaling (concerning)")
        else:
            print("     ❌ Poor scaling (possibly quadratic or worse)")
        
        print(f"\n🏆 Key Insight:")
        if dag_linearity_ratio <= 1.2:
            print(f"   DAG maintains excellent linear scaling efficiency!")
            print(f"   Time grows only {dag_factor:.1f}x while participants grow {k_factor:.1f}x")
        else:
            print(f"   DAG scaling could be improved")
            print(f"   Time grows {dag_factor:.1f}x while participants grow {k_factor:.1f}x")
        
        print(f"\n📊 O(N²) Complexity Analysis:")
        print(f"   If DAG had O(N²) scaling like naive algorithms:")
        print(f"     Time at K={k_values[-1]} would be: {dag_times[0] * quadratic_factor:.1f}s")
        print(f"     Actual DAG time: {dag_times[-1]:.1f}s")
        print(f"     DAG efficiency vs O(N²): {(dag_times[0] * quadratic_factor / dag_times[-1]):.1f}x better!")
        
        print(f"\n🚀 O(e^N) Exponential Analysis:")
        print(f"   If DAG had O(e^N) scaling like worst-case algorithms:")
        print(f"     Time at K={k_values[-1]} would be: {dag_times[0] * exponential_factor:.1f}s")
        print(f"     Actual DAG time: {dag_times[-1]:.1f}s")
        print(f"     DAG efficiency vs O(e^N): {(dag_times[0] * exponential_factor / dag_times[-1]):.0f}x better!")
        
        if dag_vs_quadratic_ratio < 0.5:
            print(f"   🎉 DAG scales MUCH better than O(N²) - excellent efficiency!")
        elif dag_vs_quadratic_ratio < 1.0:
            print(f"   ✅ DAG scales better than O(N²) - good efficiency")
        else:
            print(f"   ⚠️  DAG scaling approaches O(N²) - room for improvement")
            
        if dag_vs_exponential_ratio < 0.01:
            print(f"   🌟 DAG scales INCREDIBLY better than O(e^N) - outstanding!")
        elif dag_vs_exponential_ratio < 0.1:
            print(f"   🎉 DAG scales much better than O(e^N) - excellent!")
        else:
            print(f"   ✅ DAG scales better than O(e^N) - good efficiency")
        
    print("="*70)

def main():
    """Main function"""
    results_dir = '/home/giac/DAGTrustFL/DAGTrustFL/FL/results_scalability'
    
    print("🔍 Loading scalability experiment data...")
    
    # Load timing data
    timing_data = load_timing_data(results_dir)
    
    if not timing_data:
        print("❌ No timing data found in results_scalability folder.")
        print("   Please run scalability experiments first with:")
        print("   python3 automated_test_scalability.py")
        return
    
    print(f"✅ Found data for K values: {sorted(timing_data.keys())}")
    print("\n🚀 Generating interactive DAG scaling complexity analysis...")
    
    # Generate and show interactive plot
    plot_dag_scaling_analysis(timing_data)

if __name__ == '__main__':
    main()
