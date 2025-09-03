#!/usr/bin/env python3
"""
Automated test runner script for DAGTrustFL experiments.
This script runs experiments with different chi values and malicious/corrupted percentages,
organizing results in separate folders.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def modify_chi_in_dag_functions(chi_value):
    """Modify the chi parameter in DAG_functions.py"""
    file_path = '/home/giac/DAGTrustFL/DAGTrustFL/DAG/DAG_functions.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the TrustInformedTipSelector_NoThreshold initialization
    old_pattern = """    selector = TipSelectionAlgorithm.TrustInformedTipSelector_NoThreshold(
        similarity_method='cosine',
    )"""
    
    new_pattern = f"""    selector = TipSelectionAlgorithm.TrustInformedTipSelector_NoThreshold(
        chi={chi_value},
        similarity_method='cosine',
    )"""
    
    # If the pattern with chi already exists, replace it
    import re
    chi_pattern = r'selector = TipSelectionAlgorithm\.TrustInformedTipSelector_NoThreshold\(\s*chi=\d+,\s*similarity_method=\'cosine\',\s*\)'
    if re.search(chi_pattern, content):
        content = re.sub(chi_pattern, f"selector = TipSelectionAlgorithm.TrustInformedTipSelector_NoThreshold(\n        chi={chi_value},\n        similarity_method='cosine',\n    )", content)
    else:
        # Replace the version without chi
        content = content.replace(old_pattern, new_pattern)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Modified chi parameter to {chi_value} in DAG_functions.py")

def modify_malicious_corrupted_percentages(malicious_pct, corrupted_pct):
    """Modify malicious_percentage and corrupted_percentage in DAG_functions.py"""
    file_path = '/home/giac/DAGTrustFL/DAGTrustFL/DAG/DAG_functions.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the percentages
    content = content.replace(
        f'malicious_percentage = 0.1', 
        f'malicious_percentage = {malicious_pct}'
    ).replace(
        f'malicious_percentage = 0.2', 
        f'malicious_percentage = {malicious_pct}'
    ).replace(
        f'corrupted_percentage = 0.2', 
        f'corrupted_percentage = {corrupted_pct}'
    ).replace(
        f'corrupted_percentage = 0.1', 
        f'corrupted_percentage = {corrupted_pct}'
    )
    
    # Use regex to replace any existing values
    import re
    content = re.sub(r'malicious_percentage = [0-9.]+', f'malicious_percentage = {malicious_pct}', content)
    content = re.sub(r'corrupted_percentage = [0-9.]+', f'corrupted_percentage = {corrupted_pct}', content)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Modified malicious_percentage to {malicious_pct} and corrupted_percentage to {corrupted_pct}")

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

def create_results_folder(chi_value, malicious_pct, corrupted_pct):
    """Create a folder name based on parameters"""
    mal_pct_str = str(int(malicious_pct * 100))
    cor_pct_str = str(int(corrupted_pct * 100))
    folder_name = f"chi_{chi_value}_mal_{mal_pct_str}_cor_{cor_pct_str}"
    return folder_name

def move_results_to_folder(folder_name):
    """Move all results to the specified folder"""
    base_path = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL')
    results_folder = base_path / 'results' / folder_name
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

def run_experiment_set(chi_value, malicious_pct, corrupted_pct):
    """Run a complete experiment set for given parameters"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT SET:")
    print(f"  chi = {chi_value}")
    print(f"  malicious_percentage = {malicious_pct}")
    print(f"  corrupted_percentage = {corrupted_pct}")
    print(f"{'='*60}")
    
    # 1. Modify parameters
    modify_chi_in_dag_functions(chi_value)
    modify_malicious_corrupted_percentages(malicious_pct, corrupted_pct)
    
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
    folder_name = create_results_folder(chi_value, malicious_pct, corrupted_pct)
    move_results_to_folder(folder_name)
    
    print(f"✓ Experiment set completed and results stored in {folder_name}")

def main():
    """Main function to run all experiments"""
    print("Starting automated test runner for DAGTrustFL")
    print("This will run experiments with different chi values and percentage settings")
    
    # Experiment configurations
    configurations = [
        # First set: malicious=10%, corrupted=20%, chi=0-4
        (0.1, 0.2, list(range(5))),  # chi 0,1,2,3,4
        # Second set: malicious=20%, corrupted=10%, chi=0-4  
        (0.2, 0.1, list(range(5)))   # chi 0,1,2,3,4
    ]
    
    total_experiments = sum(len(chi_values) for _, _, chi_values in configurations)
    current_experiment = 0
    
    try:
        for malicious_pct, corrupted_pct, chi_values in configurations:
            print(f"\n{'#'*70}")
            print(f"CONFIGURATION SET: malicious={int(malicious_pct*100)}%, corrupted={int(corrupted_pct*100)}%")
            print(f"{'#'*70}")
            
            for chi_value in chi_values:
                current_experiment += 1
                print(f"\nProgress: {current_experiment}/{total_experiments}")
                
                run_experiment_set(chi_value, malicious_pct, corrupted_pct)
        
        print(f"\n{'='*70}")
        print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"Total experiments run: {total_experiments}")
        print("Results are organized in the FL/results/ folder")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
