#!/usr/bin/env python3
"""
Remove final_model_accuracy column from accuracy_data_compact.csv
================================================================

Rimuove la colonna final_model_accuracy dal file CSV.

Autore: DAGTrustFL Analysis Suite
Data: 2025-09-04
"""

import pandas as pd
from pathlib import Path

def remove_final_model_accuracy():
    """Rimuove la colonna final_model_accuracy dal CSV."""
    
    csv_file = "metrics_notrust.csv"
    
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        return False
    
    # Leggi il CSV
    df = pd.read_csv(csv_file)
    
    print(f"📊 Original data shape: {df.shape}")
    print("Original columns:", list(df.columns))
    
    # Rimuovi la colonna final_model_accuracy se esiste
    if 'final_model_accuracy' in df.columns:
        df = df.drop('final_model_accuracy', axis=1)
        print("✅ Removed 'final_model_accuracy' column")
    else:
        print("⚠️  Column 'final_model_accuracy' not found")
    
    print(f"📊 Updated data shape: {df.shape}")
    print("Updated columns:", list(df.columns))
    
    # Salva il file aggiornato
    df.to_csv(csv_file, index=False)
    print(f"💾 Updated file saved: {csv_file}")
    
    # Mostra il contenuto aggiornato
    print("\n📋 Updated CSV content:")
    print(df.to_string(index=False))
    
    return True

def main():
    """Funzione principale."""
    print("🚀 Removing final_model_accuracy column...")
    print("="*50)
    
    success = remove_final_model_accuracy()
    
    if success:
        print("\n🎉 Column removal completed successfully!")
    else:
        print("\n❌ Column removal failed!")
    
    return success

if __name__ == "__main__":
    main()
