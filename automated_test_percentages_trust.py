#!/usr/bin/env python3
"""
Automated test runner script for DAGTrustFL experiments with different malicious and corrupted percentages.
This script runs experiments with different combinations of malicious and corrupted percentages,
organizing results in separate folders with "trust" suffix.

The script tests all combinations where:
- Malicious percentage: 0%, 10%, 20%, 30%, 40%
- Corrupted percentage: 0%, 10%, 20%, 30%
- Total (malicious + corrupted) between 10% and 40%
- Special case: 0% malicious only tested with 0% corrupted
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
            timeout=3800  # 1 hour timeout
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
        print(f"❌ {description} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def create_results_folder(malicious_perc, corrupted_perc):
    """Create and return the results folder name for given percentages"""
    folder_name = f"mal_{malicious_perc}_cor_{corrupted_perc}_trust"
    results_path = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL/results_percentages_trust')
    
    # Create main results directory if it doesn't exist
    results_path.mkdir(exist_ok=True)
    
    # Create specific folder for this combination
    folder_path = results_path / folder_name
    folder_path.mkdir(exist_ok=True)
    
    print(f"Created results folder: {folder_path}")
    return folder_name

def move_results_to_folder(folder_name):
    """Move all result files to the specified folder"""
    source_dir = Path('/home/giac/DAGTrustFL/DAGTrustFL/FL')
    dest_dir = Path(f'/home/giac/DAGTrustFL/DAGTrustFL/FL/results_percentages_trust/{folder_name}')
    
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

def run_experiment_set(malicious_perc, corrupted_perc):
    """Run a complete experiment set for given percentages"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT SET (TRUST VERSION):")
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
    malicious_options = [30]
    corrupted_options = [0]
    
    for mal in malicious_options:
        for cor in corrupted_options:
            # Special case: 0% malicious only with 0% corrupted
            if mal == 0:
                if cor == 0:
                    combinations.append((mal, cor))
                continue
            
            total = mal + cor
            # Only include combinations where total is between 10% and 40%
            if 10 <= total <= 40:
                combinations.append((mal, cor))
    
    return combinations

def main():
    """Main function to run all experiments"""
    print("Starting automated test runner for DAGTrustFL - Percentage Testing (TRUST VERSION)")
    print("This will run experiments with different malicious and corrupted percentages")
    print("Results will be stored in results_percentages_trust/ folder")
    
    # Generate all valid combinations
    combinations = generate_combinations()
    
    print(f"\n{'#'*70}")
    print(f"PERCENTAGE TESTING (TRUST) - {len(combinations)} experiments")
    print("Valid combinations (malicious%, corrupted%):")
    for mal, cor in combinations:
        if mal == 0:
            print(f"  • {mal}% malicious, {cor}% corrupted (baseline case)")
        else:
            print(f"  • {mal}% malicious, {cor}% corrupted (total: {mal+cor}%)")
    print(f"{'#'*70}")
    
    try:
        for i, (malicious_perc, corrupted_perc) in enumerate(combinations, 1):
            print(f"\nProgress: {i}/{len(combinations)}")
            
            run_experiment_set(malicious_perc, corrupted_perc)
        
        print(f"\n{'='*70}")
        print("🎉 ALL PERCENTAGE EXPERIMENTS (TRUST) COMPLETED SUCCESSFULLY!")
        print(f"Total experiments run: {len(combinations)}")
        print("Combinations tested:")
        for mal, cor in combinations:
            if mal == 0:
                print(f"  • {mal}% malicious, {cor}% corrupted (baseline)")
            else:
                print(f"  • {mal}% malicious, {cor}% corrupted")
        print("Results are organized in the FL/results_percentages_trust/ folder")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during experiments: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
