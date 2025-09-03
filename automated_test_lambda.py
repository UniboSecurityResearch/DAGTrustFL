#!/usr/bin/env python3
"""
Automated test runner script for DAGTrustFL experiments with different lambda values.
This script runs experiments with different lambda values (historical trust influence),
organizing results in separate folders.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import re

def modify_lambda_in_dag_functions(lambda_value):
    """Modify the lambda_ parameter in DAG_functions.py"""
    file_path = '/home/giac/DAGTrustFL/DAGTrustFL/DAG/DAG_functions.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace lambda_ value using regex to handle different existing values
    content = re.sub(r'lambda_ = [0-9.]+', f'lambda_ = {lambda_value}', content)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Modified lambda_ parameter to {lambda_value} in DAG_functions.py")

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

def create_results_folder(lambda_value):
    """Create a folder name based on lambda value"""
    lambda_str = str(lambda_value).replace('.', '_')
    folder_name = f"lambda_{lambda_str}"
    return folder_name

def move_results_to_folder(folder_name):
    """Move all results to the specified folder"""
    base_path = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL')
    results_folder = base_path / 'results_lambda' / folder_name
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

def run_experiment_set(lambda_value):
    """Run a complete experiment set for given lambda value"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT SET:")
    print(f"  lambda = {lambda_value}")
    print(f"{'='*60}")
    
    # 1. Modify lambda parameter
    modify_lambda_in_dag_functions(lambda_value)
    
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
    folder_name = create_results_folder(lambda_value)
    move_results_to_folder(folder_name)
    
    print(f"✓ Experiment set completed and results stored in {folder_name}")

def main():
    """Main function to run all experiments"""
    print("Starting automated test runner for DAGTrustFL - Lambda Value Testing")
    print("This will run experiments with different lambda values")
    
    # Lambda values to test
    lambda_values = [0.1]
    
    total_experiments = len(lambda_values)
    
    try:
        print(f"\n{'#'*70}")
        print(f"LAMBDA VALUE TESTING - {total_experiments} experiments")
        print(f"Testing lambda values: {lambda_values}")
        print(f"{'#'*70}")
        
        for i, lambda_value in enumerate(lambda_values, 1):
            print(f"\nProgress: {i}/{total_experiments}")
            
            run_experiment_set(lambda_value)
        
        print(f"\n{'='*70}")
        print("🎉 ALL LAMBDA EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"Total experiments run: {total_experiments}")
        print(f"Lambda values tested: {lambda_values}")
        print("Results are organized in the FL/results_lambda/ folder")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
