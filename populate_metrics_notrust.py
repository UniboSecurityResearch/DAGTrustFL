#!/usr/bin/env python3
"""
Populate metrics_notrust.csv with data from percentages_comparison_data.csv
=========================================================================

Questo script estrae i valori fpr, fnr_malicious e DACC (accuracy) dal file
percentages_comparison_data.csv e li inserisce nel file metrics_notrust.csv
che ha le stesse configurazioni ma valori mancanti.

Autore: DAGTrustFL Analysis Suite
Data: 2025-09-04
"""

import pandas as pd
import numpy as np
from pathlib import Path

def populate_metrics_notrust():
    """Popola metrics_notrust.csv con i dati da percentages_comparison_data.csv."""
    
    print("🔄 Populating metrics_notrust.csv with data...")
    
    # Percorsi dei file
    source_file = "percentages_comparison_plots/csv_data/percentages_comparison_data.csv"
    target_file = "metrics_notrust.csv"
    
    # Verifica esistenza file
    if not Path(source_file).exists():
        print(f"❌ Source file not found: {source_file}")
        return False
    
    if not Path(target_file).exists():
        print(f"❌ Target file not found: {target_file}")
        return False
    
    # Carica i file
    source_df = pd.read_csv(source_file)
    target_df = pd.read_csv(target_file)
    
    print(f"📊 Loaded source data: {len(source_df)} rows")
    print(f"📊 Loaded target data: {len(target_df)} rows")
    
    # Mostra le colonne disponibili
    print(f"\n📋 Source columns: {list(source_df.columns)}")
    print(f"📋 Target columns: {list(target_df.columns)}")
    
    # Crea chiavi di merge
    source_df['merge_key'] = source_df.apply(
        lambda row: f"{row['malicious_perc']}_{row['corrupted_perc']}_{row['total_perc']}", 
        axis=1
    )
    
    target_df['merge_key'] = target_df.apply(
        lambda row: f"{row['malicious_percentage']}_{row['corrupted_percentage']}_{row['total_threat_percentage']}", 
        axis=1
    )
    
    print(f"\n🔍 Source merge keys: {sorted(source_df['merge_key'].tolist())}")
    print(f"🔍 Target merge keys: {sorted(target_df['merge_key'].tolist())}")
    
    # Crea un dizionario per il mapping
    data_mapping = {}
    for _, row in source_df.iterrows():
        key = row['merge_key']
        data_mapping[key] = {
            'fpr': row['fpr'],
            'fnr_malicious': row['fnr_malicious'],
            'DACC': row['accuracy']  # accuracy nel source diventa DACC nel target
        }
    
    print(f"\n📊 Created mapping for {len(data_mapping)} configurations")
    
    # Popola il target dataframe
    matches_found = 0
    for idx, row in target_df.iterrows():
        key = row['merge_key']
        if key in data_mapping:
            target_df.at[idx, 'fpr'] = round(data_mapping[key]['fpr'], 2)
            target_df.at[idx, 'fnr_malicious'] = round(data_mapping[key]['fnr_malicious'], 2)
            target_df.at[idx, 'DACC'] = round(data_mapping[key]['DACC'], 2)
            matches_found += 1
            print(f"✓ Populated row {idx}: {key}")
        else:
            print(f"⚠️ No match found for: {key}")
    
    # Rimuovi la colonna merge_key
    target_df = target_df.drop('merge_key', axis=1)
    
    print(f"\n✅ Successfully populated {matches_found} out of {len(target_df)} rows")
    
    # Mostra il risultato
    print("\n📋 Updated metrics_notrust.csv:")
    print(target_df.to_string(index=False))
    
    # Salva il file aggiornato
    target_df.to_csv(target_file, index=False)
    print(f"\n💾 Updated file saved: {target_file}")
    
    # Verifica che non ci siano valori mancanti
    missing_count = target_df[['fpr', 'fnr_malicious', 'DACC']].isnull().sum().sum()
    if missing_count == 0:
        print("✅ All values successfully populated!")
    else:
        print(f"⚠️ {missing_count} values still missing")
    
    return True

def main():
    """Funzione principale."""
    print("🚀 Starting metrics population process...")
    print("="*60)
    
    success = populate_metrics_notrust()
    
    if success:
        print("\n🎉 Metrics population completed successfully!")
    else:
        print("\n❌ Metrics population failed!")
    
    return success

if __name__ == "__main__":
    main()
