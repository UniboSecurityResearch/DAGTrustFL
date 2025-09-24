#!/usr/bin/env python3
"""
Extract Accuracy Data to CSV - Quick Data Export
===============================================

Script veloce per estrarre i dati di accuracy finale
in formato CSV per analisi esterne.
"""

import pandas as pd
import os
import re
from pathlib import Path

def extract_accuracy_data():
    """Estrae i dati di accuracy e li salva in CSV."""
    
    results_base_path = Path("../FL/results_percentages_trust")
    experiments = {}
    
    print("🔍 Scanning FL trust experiment folders...")
    
    if not results_base_path.exists():
        print(f"❌ Results path not found: {results_base_path}")
        return None
    
    # Pattern: mal_X_cor_Y_trust
    pattern = r'mal_(\d+)_cor_(\d+)_trust'
    
    for folder in results_base_path.iterdir():
        if folder.is_dir():
            match = re.match(pattern, folder.name)
            if match:
                malicious_perc = int(match.group(1))
                corrupted_perc = int(match.group(2))
                
                # Look for FL results CSV file
                fl_csv_file = folder / "3000_results.csv"
                if fl_csv_file.exists():
                    experiments[(malicious_perc, corrupted_perc)] = {
                        'folder': folder,
                        'fl_csv_path': fl_csv_file,
                        'malicious_perc': malicious_perc,
                        'corrupted_perc': corrupted_perc,
                        'total_perc': malicious_perc + corrupted_perc,
                        'is_baseline': (malicious_perc == 0 and corrupted_perc == 0)
                    }
                    print(f"   ✓ Found: {malicious_perc}%M, {corrupted_perc}%C")
    
    # Load final accuracies
    data_points = []
    
    print("\n📊 Loading final model accuracies...")
    
    for (mal_perc, cor_perc), exp_info in sorted(experiments.items()):
        try:
            # Load FL CSV
            raw_data = pd.read_csv(exp_info['fl_csv_path'], header=None)
            
            # First column is accuracy
            final_accuracy = raw_data.iloc[-1, 0]  # Last value of first column
            total_threat = exp_info['total_perc']
            is_baseline = exp_info['is_baseline']
            
            data_point = {
                'malicious_percentage': mal_perc,
                'corrupted_percentage': cor_perc,
                'total_threat_percentage': total_threat,
                'final_model_accuracy': round(final_accuracy, 2),
                'is_baseline': is_baseline,
                'experiment_type': 'Baseline' if is_baseline else 'Threat Scenario'
            }
            
            data_points.append(data_point)
            
            print(f"   ✓ {mal_perc}%M,{cor_perc}%C: {final_accuracy:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error loading {exp_info['fl_csv_path']}: {e}")
    
    if not data_points:
        print("❌ No valid accuracy data found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data_points)
    
    # Sort by total threat percentage
    df = df.sort_values('total_threat_percentage')
    
    # Save to CSV
    csv_filename = "accuracy_data_table.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\n💾 Data saved to: {csv_filename}")
    print(f"📊 Total experiments: {len(df)}")
    print(f"   Threat range: {df['total_threat_percentage'].min()}% - {df['total_threat_percentage'].max()}%")
    print(f"   Accuracy range: {df['final_model_accuracy'].min()}% - {df['final_model_accuracy'].max()}%")
    
    # Print the data for immediate viewing
    print(f"\n📋 ACCURACY DATA TABLE:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Also create a compact version
    compact_df = df[['malicious_percentage', 'corrupted_percentage', 'total_threat_percentage', 'final_model_accuracy']].copy()
    compact_csv = "accuracy_data_compact.csv"
    compact_df.to_csv(compact_csv, index=False)
    
    print(f"\n💾 Compact version saved to: {compact_csv}")
    
    return df

if __name__ == "__main__":
    print("🚀 Extracting Accuracy Data to CSV")
    print("=" * 50)
    
    df = extract_accuracy_data()
    
    if df is not None:
        print("\n✅ Data extraction completed successfully!")
        print(f"📁 Files created:")
        print(f"   - accuracy_data_table.csv (full data)")
        print(f"   - accuracy_data_compact.csv (essential columns)")
    else:
        print("\n❌ Data extraction failed!")
