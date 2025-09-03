#!/usr/bin/env python3
"""
Automated test runner script for DAGTrustFL experiments with different varsigma values.
This script runs experiments with different varsigma values (cumulative weight penalty factor),
organizing results in separate folders.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import re

def modify_varsigma_in_dag_functions(varsigma_value):
    """Modify the varsigma parameter in DAG_functions.py"""
    file_path = '/home/giac/DAGTrustFL/DAGTrustFL/DAG/DAG_functions.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the selector initialization and replace varsigma value
    # Look for the TrustInformedTipSelector_NoThreshold initialization
    pattern = r'(selector = TipSelectionAlgorithm\.TrustInformedTipSelector_NoThreshold\s*\(\s*chi=\d+,\s*similarity_method=\'[^\']*\'(?:,\s*varsigma=[0-9.]+)?)\s*\)'
    
    # Check if varsigma parameter already exists
    if 'varsigma=' in content:
        # Replace existing varsigma value
        content = re.sub(r'varsigma=[0-9.]+', f'varsigma={varsigma_value}', content)
    else:
        # Add varsigma parameter to the existing initialization
        content = re.sub(
            r'(selector = TipSelectionAlgorithm\.TrustInformedTipSelector_NoThreshold\s*\(\s*chi=\d+,\s*similarity_method=\'[^\']*\')\s*\)',
            r'\1,\n        varsigma=' + str(varsigma_value) + '\n    )',
            content
        )
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Modified varsigma parameter to {varsigma_value} in DAG_functions.py")

def run_python_script(script_path, description):
    """Run a Python script and wait for completion"""
    print(f"\n=== Running {description} ===")
    print(f"Command: python3 {script_path}")
    
    try:
        result = subprocess.run(
            ['python3', script_path], 
            cwd='/home/giac/DAGTrustFL/DAGTrustFL/FL',
            capture_output=True, 
            text=True,
            check=True
        )
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False

def create_results_folder(varsigma_value):
    """Create a folder name based on varsigma value"""
    varsigma_str = str(varsigma_value).replace('.', '_')
    folder_name = f"varsigma_{varsigma_str}"
    return folder_name

def move_results_to_folder(folder_name):
    """Move all results to the specified folder"""
    base_path = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL')
    results_folder = base_path / 'results_varsigma' / folder_name
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Files to move from FL directory
    csv_files = [
        '3000_workers_selected.csv',
        '3000_results.csv', 
        'dag_trust_analysis.csv'
    ]
    
    # Move CSV files
    for csv_file in csv_files:
        src_path = base_path / csv_file
        if src_path.exists():
            dst_path = results_folder / csv_file
            shutil.move(str(src_path), str(dst_path))
            print(f"Moved {csv_file} to {folder_name}")
        else:
            print(f"Warning: {csv_file} not found")
    
    # Move dag_analysis_plots folder
    plots_src = base_path / 'dag_analysis_plots'
    if plots_src.exists():
        plots_dst = results_folder / 'dag_analysis_plots'
        if plots_dst.exists():
            shutil.rmtree(plots_dst)
        shutil.move(str(plots_src), str(plots_dst))
        print(f"Moved dag_analysis_plots folder to {folder_name}")
    else:
        print("Warning: dag_analysis_plots folder not found")

def run_experiment_set(varsigma_value):
    """Run a complete experiment set for given varsigma value"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT SET:")
    print(f"  varsigma = {varsigma_value}")
    print(f"{'='*60}")
    
    # 1. Modify varsigma parameter
    modify_varsigma_in_dag_functions(varsigma_value)
    
    # 2. Run label flipping attack
    if not run_python_script('label_flipping_attack.py', 'Label Flipping Attack'):
        print("Label flipping attack failed, continuing anyway...")
    
    # 3. Run DAG analysis
    if not run_python_script('analyze_dag_csv_pdf.py', 'DAG Analysis PDF'):
        print("DAG analysis failed, continuing anyway...")
    
    # 4. Run FL analysis
    if not run_python_script('analyze_fl_csv_pdf.py', 'FL Analysis PDF'):
        print("FL analysis failed, continuing anyway...")
    
    # 5. Move results to folder
    folder_name = create_results_folder(varsigma_value)
    move_results_to_folder(folder_name)
    
    print(f"✓ Experiment set completed and results stored in {folder_name}")

def main():
    """Main function to run all experiments"""
    print("Starting automated test runner for DAGTrustFL - Varsigma Value Testing")
    print("This will run experiments with different varsigma values")
    
    # Varsigma values to test (cumulative weight penalty factor)
    varsigma_values = [0.00, 0.02, 0.03, 0.05]
    
    total_experiments = len(varsigma_values)
    
    try:
        print(f"\n{'#'*70}")
        print(f"VARSIGMA VALUE TESTING - {total_experiments} experiments")
        print(f"Testing varsigma values: {varsigma_values}")
        print(f"{'#'*70}")
        
        for i, varsigma_value in enumerate(varsigma_values, 1):
            print(f"\nProgress: {i}/{total_experiments}")
            
            run_experiment_set(varsigma_value)
        
        print(f"\n{'='*70}")
        print("🎉 ALL VARSIGMA EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"Total experiments run: {total_experiments}")
        print(f"Varsigma values tested: {varsigma_values}")
        print("Results are organized in the FL/results_varsigma/ folder")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
