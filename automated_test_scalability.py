#!/usr/bin/env python3
"""
Automated scalability test runner script for DAGTrustFL experiments.
This script runs experiments with different numbers of participants (K) to test system scalability.

The script tests the following K values:
- K = 20, 40, 60, 80, 100 participants

For each K value, it collects:
- Standard DAG analysis data (dag_trust_analysis.csv)
- FL results (3000_results.csv, 3000_workers_selected.csv)
- Timing data (ML training times vs DAG operation times)
- Performance analysis plots
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import re
import time

def modify_k_in_dag_functions(k_value):
    """Modify the K parameter in DAG_functions.py"""
    file_path = '/home/giac/DAGTrustFL/DAGTrustFL/DAG/DAG_functions.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace K value
    k_pattern = r'(K=\s*)[0-9]+(\s*# Number of peers)'
    content = re.sub(k_pattern, rf'\g<1>{k_value}\g<2>', content)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Modified K value to {k_value} participants in DAG_functions.py")

def run_python_script(script_path, description, timeout=7200):
    """Run a Python script and wait for completion"""
    print(f"\n=== Running {description} ===")
    print(f"Command: python3 {script_path}")
    
    try:
        result = subprocess.run(
            ['python3', script_path], 
            cwd='/home/giac/DAGTrustFL/DAGTrustFL/FL',
            capture_output=True, 
            text=True,
            timeout=timeout  # 2 hours timeout for larger experiments
        )
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Show last 500 chars
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} timed out after {timeout/3600:.1f} hours")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def create_results_folder(k_value):
    """Create and return the results folder name for given K value"""
    folder_name = f"K_{k_value}_scalability"
    results_path = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL/results_scalability')
    
    # Create main results directory if it doesn't exist
    results_path.mkdir(exist_ok=True)
    
    # Create specific folder for this K value
    folder_path = results_path / folder_name
    folder_path.mkdir(exist_ok=True)
    
    print(f"Created results folder: {folder_path}")
    return folder_name

def move_results_to_folder(folder_name):
    """Move all result files to the specified folder"""
    source_dir = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL')
    dest_dir = Path(f'/home/giac/DAGTrustFL/DAGTrustFL/FL/results_scalability/{folder_name}')
    
    # Files to move
    files_to_move = [
        '3000_results.csv',
        '3000_workers_selected.csv', 
        'dag_trust_analysis.csv',
        'cosine_similarity_report.csv',
        'cosine_similarity_temporal_analysis.pdf'
    ]
    
    # Move files
    for file_name in files_to_move:
        src = source_dir / file_name
        dst = dest_dir / file_name
        
        if src.exists():
            if dst.exists():
                dst.unlink()  # Remove existing file
            shutil.move(str(src), str(dst))
            print(f"Moved {file_name} to {folder_name}")
        else:
            print(f"Warning: {file_name} not found")
    
    # Move plots folder if it exists
    plots_src = source_dir / 'dag_analysis_plots'
    plots_dst = dest_dir / 'dag_analysis_plots'
    
    if plots_src.exists():
        if plots_dst.exists():
            shutil.rmtree(plots_dst)
        shutil.move(str(plots_src), str(plots_dst))
        print(f"Moved dag_analysis_plots folder to {folder_name}")
    else:
        print("Warning: dag_analysis_plots folder not found")
    
    # Move timing results if they exist
    timing_src = source_dir / 'timing_results'
    timing_dst = dest_dir / 'timing_results'
    
    if timing_src.exists():
        if timing_dst.exists():
            shutil.rmtree(timing_dst)
        shutil.move(str(timing_src), str(timing_dst))
        print(f"Moved timing_results folder to {folder_name}")
    else:
        print("Warning: timing_results folder not found")

def run_scalability_experiment(k_value):
    """Run a complete scalability experiment for given K value"""
    print(f"\n{'='*70}")
    print(f"RUNNING SCALABILITY EXPERIMENT:")
    print(f"  Number of participants (K): {k_value}")
    print(f"  Estimated time: {estimate_experiment_time(k_value)} minutes")
    print(f"{'='*70}")
    
    experiment_start_time = time.time()
    
    # 1. Modify K value
    modify_k_in_dag_functions(k_value)
    
    # 2. Run label flipping attack
    if not run_python_script('label_flipping_attack.py', 'Label Flipping Attack', timeout=7200):
        print("Label flipping attack failed, continuing anyway...")
    
    # 3. Run DAG analysis
    if not run_python_script('analyze_dag_csv_pdf.py', 'DAG Analysis PDF', timeout=3600):
        print("DAG analysis failed, continuing anyway...")
    
    # 4. Run FL analysis
    if not run_python_script('analyze_fl_csv_pdf.py', 'FL Analysis PDF', timeout=1800):
        print("FL analysis failed, continuing anyway...")
    
    # 5. Move results to folder
    folder_name = create_results_folder(k_value)
    move_results_to_folder(folder_name)
    
    experiment_duration = (time.time() - experiment_start_time) / 60  # in minutes
    print(f"✓ Scalability experiment for K={k_value} completed in {experiment_duration:.1f} minutes")
    print(f"  Results stored in {folder_name}")
    
    return experiment_duration

def estimate_experiment_time(k_value):
    """Estimate experiment time based on K value"""
    # Rough estimation: time scales approximately linearly with K
    # Base time for K=10 is around 30 minutes
    base_time = 30  # minutes for K=10
    base_k = 10
    estimated_time = (k_value / base_k) * base_time
    return int(estimated_time)

def generate_scalability_summary(results, output_dir):
    """Generate a summary of scalability results"""
    summary_path = output_dir / "scalability_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("DAGTrustFL Scalability Test Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(results)}\n\n")
        
        f.write("Results by K value:\n")
        f.write("-" * 30 + "\n")
        for k_value, duration in results.items():
            f.write(f"K = {k_value:3d}: {duration:6.1f} minutes\n")
        
        f.write("\nData collected for each experiment:\n")
        f.write("- DAG trust analysis (dag_trust_analysis.csv)\n")
        f.write("- FL results (3000_results.csv, 3000_workers_selected.csv)\n")
        f.write("- Performance timing data (timing_results/)\n")
        f.write("- Analysis plots (dag_analysis_plots/)\n")
        f.write("- Cosine similarity analysis\n")
        
        total_time = sum(results.values())
        f.write(f"\nTotal experimental time: {total_time:.1f} minutes ({total_time/60:.1f} hours)\n")
        
        if len(results) > 1:
            k_values = list(results.keys())
            durations = list(results.values())
            avg_scaling = sum(durations[i]/durations[0] * k_values[0]/k_values[i] 
                             for i in range(1, len(k_values))) / (len(k_values) - 1)
            f.write(f"Average scaling factor: {avg_scaling:.2f}\n")
    
    print(f"Scalability summary saved to: {summary_path}")

def main():
    """Main function to run all scalability experiments"""
    print("Starting automated scalability test runner for DAGTrustFL")
    print("This will test system performance with different numbers of participants")
    print("Results will be stored in results_scalability/ folder")
    
    # K values to test
    k_values = [200]
    
    print(f"\n{'#'*80}")
    print(f"SCALABILITY TESTING - {len(k_values)} experiments")
    print("K values to test:")
    for k in k_values:
        estimated_time = estimate_experiment_time(k)
        print(f"  • K = {k:3d} participants (estimated time: ~{estimated_time} minutes)")
    
    total_estimated_time = sum(estimate_experiment_time(k) for k in k_values)
    print(f"\nTotal estimated time: ~{total_estimated_time} minutes ({total_estimated_time/60:.1f} hours)")
    print(f"{'#'*80}")
    
    # Ask for confirmation
    try:
        confirm = input("\nProceed with scalability testing? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Scalability testing cancelled.")
            return
    except KeyboardInterrupt:
        print("\nScalability testing cancelled.")
        return
    
    results = {}
    failed_experiments = []
    
    try:
        for i, k_value in enumerate(k_values, 1):
            print(f"\nProgress: {i}/{len(k_values)} experiments")
            print(f"Testing K = {k_value} participants...")
            
            try:
                duration = run_scalability_experiment(k_value)
                results[k_value] = duration
            except Exception as e:
                print(f"❌ Experiment for K={k_value} failed: {e}")
                failed_experiments.append(k_value)
                continue
        
        # Generate summary
        results_path = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL/results_scalability')
        generate_scalability_summary(results, results_path)
        
        print(f"\n{'='*80}")
        print("🎉 SCALABILITY TESTING COMPLETED!")
        print(f"Successful experiments: {len(results)}/{len(k_values)}")
        if results:
            print("Results summary:")
            for k_value, duration in results.items():
                print(f"  • K = {k_value:3d}: {duration:6.1f} minutes")
        
        if failed_experiments:
            print(f"\nFailed experiments (K values): {failed_experiments}")
        
        print("All results are organized in the FL/results_scalability/ folder")
        print("Each folder contains:")
        print("  - Standard analysis files (CSV, PDF)")
        print("  - Timing performance data (timing_results/)")
        print("  - Analysis plots and summaries")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scalability testing interrupted by user")
        if results:
            print("Partial results collected:")
            for k_value, duration in results.items():
                print(f"  • K = {k_value}: {duration:.1f} minutes")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during scalability testing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
