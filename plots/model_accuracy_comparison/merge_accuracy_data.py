#!/usr/bin/env python3
"""
Merge Accuracy Data with Trust Comparison Data
=============================================

Questo script unisce i dati di accuracy_data_compact.csv con 
percentages_trust_comparison_data.csv aggiungendo le colonne:
- fpr: False Positive Rate
- fnr_malicious: False Negative Rate per peer malicious 
- DACC: Detection Accuracy (diversa da final_model_accuracy)

Autore: DAGTrustFL Analysis Suite
Data: 2025-09-03
"""

import pandas as pd
import numpy as np
from pathlib import Path

def merge_accuracy_data():
    """Merge accuracy data with trust comparison data."""
    
    print("🔄 Merging accuracy data with trust comparison data...")
    
    # Carica i file CSV
    accuracy_file = "accuracy_data_compact.csv"
    trust_file = "percentages_trust_comparison_plots/csv_data/percentages_trust_comparison_data.csv"
    
    if not Path(accuracy_file).exists():
        print(f"❌ File not found: {accuracy_file}")
        return False
    
    if not Path(trust_file).exists():
        print(f"❌ File not found: {trust_file}")
        return False
    
    # Leggi i CSV
    accuracy_df = pd.read_csv(accuracy_file)
    trust_df = pd.read_csv(trust_file)
    
    print(f"📊 Loaded accuracy data: {len(accuracy_df)} rows")
    print(f"📊 Loaded trust data: {len(trust_df)} rows")
    
    # Crea una chiave di merge basata su malicious_perc, corrupted_perc, total_threat_perc
    accuracy_df['merge_key'] = accuracy_df.apply(
        lambda row: f"{row['malicious_percentage']}_{row['corrupted_percentage']}_{row['total_threat_percentage']}", 
        axis=1
    )
    
    trust_df['merge_key'] = trust_df.apply(
        lambda row: f"{row['malicious_perc']}_{row['corrupted_perc']}_{row['total_perc']}", 
        axis=1
    )
    
    print("\n🔍 Merge keys comparison:")
    print("Accuracy keys:", sorted(accuracy_df['merge_key'].tolist()))
    print("Trust keys:", sorted(trust_df['merge_key'].tolist()))
    
    # Seleziona solo le colonne necessarie dal trust_df
    trust_cols = trust_df[['merge_key', 'fpr', 'fnr_malicious', 'accuracy']].copy()
    trust_cols = trust_cols.rename(columns={'accuracy': 'DACC'})
    
    # Merge dei dataframe
    merged_df = accuracy_df.merge(trust_cols, on='merge_key', how='left')
    
    # Rimuovi la colonna merge_key
    merged_df = merged_df.drop('merge_key', axis=1)
    
    # Riordina le colonne per avere una struttura logica
    column_order = [
        'malicious_percentage', 
        'corrupted_percentage', 
        'total_threat_percentage',
        'final_model_accuracy',
        'fpr',
        'fnr_malicious', 
        'DACC'
    ]
    
    merged_df = merged_df[column_order]
    
    # Arrotonda i valori numerici per leggibilità
    numeric_cols = ['final_model_accuracy', 'fpr', 'fnr_malicious', 'DACC']
    for col in numeric_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].round(2)
    
    print(f"\n✅ Merged data successfully!")
    print(f"📊 Final dataset: {len(merged_df)} rows x {len(merged_df.columns)} columns")
    
    # Mostra il risultato
    print("\n📋 Preview of merged data:")
    print(merged_df.to_string(index=False))
    
    # Salva il file aggiornato
    output_file = "accuracy_data_compact_enhanced.csv"
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Enhanced data saved to: {output_file}")
    
    # Controlla se ci sono valori mancanti
    missing_data = merged_df.isnull().sum()
    if missing_data.any():
        print("\n⚠️  Missing data found:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"   {col}: {count} missing values")
    else:
        print("\n✅ No missing data found!")
    
    # Sovrascrivi anche il file originale se desiderato
    overwrite_original = True
    if overwrite_original:
        merged_df.to_csv(accuracy_file, index=False)
        print(f"💾 Original file updated: {accuracy_file}")
    
    return True

def main():
    """Funzione principale."""
    print("🚀 Starting accuracy data merge process...")
    print("="*60)
    
    success = merge_accuracy_data()
    
    if success:
        print("\n🎉 Data merge completed successfully!")
    else:
        print("\n❌ Data merge failed!")
    
    return success

if __name__ == "__main__":
    main()
