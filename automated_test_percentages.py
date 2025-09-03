#!/usr/bin/env python3
"""
Automated test runner script for DAGTrustFL experiments with different malicious and corrupted percentages.
This script runs experiments with different combinations of malicious and corrupted percentages,
organizing results in separate folders.

The script tests all combinations where:
- Malicious percentage: 10%, 20%, 30%, 40%
- Corrupted percentage: 0%, 10%, 20%, 30%
- Total (malicious + corrupted) between 10% and 40%
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import re

def modify_percentages_in_dag_functions(malicious_perc, corrupted_perc):
    """Modify the malicious_percentage and corrupted_percentage parameters in DAG_functions.py"""
    file_path = '/home/giac/DAGTrustFL/DAGTrustFL/DAG/DAG_functions.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace malicious_percentage
    malicious_pattern = r'(malicious_percentage = )[0-9.]+(\s*# Percentage of malicious peers)'
    content = re.sub(malicious_pattern, rf'\g<1>{malicious_perc/100:.1f}\g<2>', content)
    
    # Replace corrupted_percentage  
    corrupted_pattern = r'(corrupted_percentage = )[0-9.]+(\s*# Percentage of corrupted peers)'
    content = re.sub(corrupted_pattern, rf'\g<1>{corrupted_perc/100:.1f}\g<2>', content)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Modified percentages: {malicious_perc}% malicious, {corrupted_perc}% corrupted in DAG_functions.py")

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

def create_results_folder(malicious_perc, corrupted_perc):
    """Create a folder name based on percentages"""
    folder_name = f"mal_{malicious_perc}_cor_{corrupted_perc}"
    return folder_name

def move_results_to_folder(folder_name):
    """Move all results to the specified folder"""
    base_path = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL')
    results_folder = base_path / 'results_percentages' / folder_name
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

def run_experiment_set(malicious_perc, corrupted_perc):
    """Run a complete experiment set for given percentages"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT SET:")
    print(f"  Malicious: {malicious_perc}%, Corrupted: {corrupted_perc}%")
    print(f"  Total: {malicious_perc + corrupted_perc}%")
    print(f"{'='*60}")
    
    # 1. Modify percentages
    modify_percentages_in_dag_functions(malicious_perc, corrupted_perc)
    
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
    folder_name = create_results_folder(malicious_perc, corrupted_perc)
    move_results_to_folder(folder_name)
    
    print(f"✓ Experiment set completed and results stored in {folder_name}")

def generate_combinations():
    """Generate all valid combinations of malicious and corrupted percentages"""
    combinations = []
    
    # Possible percentages
    malicious_options = [0,10,20,30,40]
    corrupted_options = [0,10,20,30]
    
    for mal in malicious_options:
        for cor in corrupted_options:
            total = mal + cor
            # Only include combinations where total is between 10% and 40%
            if 10 <= total <= 40:
                combinations.append((mal, cor))
    
    return combinations

def main():
    """Main function to run all experiments"""
    print("Starting automated test runner for DAGTrustFL - Percentage Testing")
    print("This will run experiments with different malicious and corrupted percentages")
    
    # Generate all valid combinations
    combinations = generate_combinations()
    
    print(f"\n{'#'*70}")
    print(f"PERCENTAGE TESTING - {len(combinations)} experiments")
    print("Valid combinations (malicious%, corrupted%):")
    for mal, cor in combinations:
        print(f"  • {mal}% malicious, {cor}% corrupted (total: {mal+cor}%)")
    print(f"{'#'*70}")
    
    try:
        for i, (malicious_perc, corrupted_perc) in enumerate(combinations, 1):
            print(f"\nProgress: {i}/{len(combinations)}")
            
            run_experiment_set(malicious_perc, corrupted_perc)
        
        print(f"\n{'='*70}")
        print("🎉 ALL PERCENTAGE EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"Total experiments run: {len(combinations)}")
        print("Combinations tested:")
        for mal, cor in combinations:
            print(f"  • {mal}% malicious, {cor}% corrupted")
        print("Results are organized in the FL/results_percentages/ folder")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
