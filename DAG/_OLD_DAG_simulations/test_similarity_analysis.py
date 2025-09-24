#!/usr/bin/env python3
"""
Test script per verificare l'analisi delle similitudini nel CSV DAG Trust.

Questo script crea dati di test con le nuove colonne di similarità e 
testa i script di analisi modificati.
"""

import pandas as pd
import numpy as np
import os

def create_test_csv():
    """Crea un CSV di test con le nuove colonne di similarità."""
    
    # Parametri di test
    rounds = 10
    num_peers = 5
    
    # Crea dati di test
    data = []
    
    for round_num in range(1, rounds + 1):
        # Dati base del round
        row = {
            'Round': round_num,
            'Total_Nodes': 10 + round_num * 2,
            'Total_Tips': 3 + round_num,
            'Good_Nodes_Removed_This_Round': np.random.randint(0, 2),
            'Malicious_Nodes_Confirmed_This_Round': np.random.randint(0, 2),
            'Corrupted_Nodes_Confirmed_This_Round': 0,
            'Total_Good_Removed': round_num * 0.5,
            'Total_Malicious_Confirmed': round_num * 0.3,
            'Total_Corrupted_Confirmed': 0,
        }
        
        # Similarity threshold che cresce logaritmicamente
        min_threshold = 0.35
        max_threshold = 0.68
        threshold_progress = np.log(round_num) / np.log(rounds)
        threshold = min_threshold + (max_threshold - min_threshold) * threshold_progress
        row['Similarity_Threshold'] = threshold
        
        # Similitudini medie - i buoni hanno similitudine più alta dei cattivi
        base_good_sim = 0.7 + 0.2 * np.random.random()  # 0.7-0.9
        base_bad_sim = 0.3 + 0.3 * np.random.random()   # 0.3-0.6
        
        # Aggiungi un po' di rumore temporale
        noise = 0.1 * np.sin(round_num / 2)
        row['Mean_Similarity_Good'] = base_good_sim + noise
        row['Mean_Similarity_Bad'] = base_bad_sim + noise * 0.5
        
        # Dati per ogni peer
        for i in range(num_peers):
            # Peer 0 e 2 sono maliciosi
            is_malicious = i in [0, 2]
            is_corrupted = False  # Nessun peer corrotto in questo test
            
            # Trust che diminuisce per i maliciosi
            if is_malicious:
                trust = max(0.1, 1.0 - (round_num * 0.08))
                struct_consistency = 0.0 if round_num > 3 else 1.0
                updates_contrib = 0.0 if round_num > 5 else 0.5
            else:
                trust = min(1.0, 0.8 + (round_num * 0.02))
                struct_consistency = 1.0
                updates_contrib = 1.0
            
            row[f'Peer_{i}_Trust'] = trust
            row[f'Peer_{i}_Structural_Consistency'] = struct_consistency
            row[f'Peer_{i}_Updates_Contribution'] = updates_contrib
            row[f'Peer_{i}_Is_Malicious'] = 1 if is_malicious else 0
            row[f'Peer_{i}_Is_Corrupted'] = 1 if is_corrupted else 0
        
        data.append(row)
    
    # Crea DataFrame e salva
    df = pd.DataFrame(data)
    csv_file = 'test_dag_trust_analysis.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"✅ CSV di test creato: {csv_file}")
    print(f"📊 {len(df)} rounds, {num_peers} peers")
    print(f"🔴 Peers maliciosi: [0, 2]")
    print(f"📈 Similarity threshold: {df['Similarity_Threshold'].iloc[0]:.4f} → {df['Similarity_Threshold'].iloc[-1]:.4f}")
    
    return csv_file

def test_analysis_scripts(csv_file):
    """Testa i script di analisi con il CSV creato."""
    
    print(f"\n🧪 Testing analysis scripts con {csv_file}...")
    
    # Test 1: Caricamento dati
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Caricamento CSV: OK ({len(df)} righe, {len(df.columns)} colonne)")
        
        # Verifica le nuove colonne
        required_cols = ['Similarity_Threshold', 'Mean_Similarity_Good', 'Mean_Similarity_Bad']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Colonne mancanti: {missing_cols}")
            return False
        else:
            print(f"✅ Nuove colonne di similarità: presenti")
            
    except Exception as e:
        print(f"❌ Errore caricamento CSV: {e}")
        return False
    
    # Test 2: Script analyze_dag_csv_pdf.py
    try:
        print(f"\n📊 Testing analyze_dag_csv_pdf.py...")
        from FL.analyze_dag_csv_pdf import DAGAnalyzer
        
        analyzer = DAGAnalyzer(csv_file)
        if analyzer.load_data():
            print(f"✅ DAGAnalyzer caricamento: OK")
            
            # Test statistiche
            analyzer.generate_summary_stats()
            print(f"✅ Summary stats: OK")
            
        else:
            print(f"❌ DAGAnalyzer caricamento: FALLITO")
            return False
            
    except Exception as e:
        print(f"❌ Errore analyze_dag_csv_pdf: {e}")
        return False
    
    # Test 3: Script quick_dag_analysis_pdf.py  
    try:
        print(f"\n⚡ Testing quick_dag_analysis_pdf.py...")
        from quick_dag_analysis_pdf import quick_analysis_pdf
        
        # Esegui analisi rapida
        success = quick_analysis_pdf(csv_file)
        if success:
            print(f"✅ Quick analysis: OK")
        else:
            print(f"❌ Quick analysis: FALLITO")
            return False
            
    except Exception as e:
        print(f"❌ Errore quick_dag_analysis_pdf: {e}")
        return False
    
    return True

def main():
    """Funzione principale di test."""
    print("🧪 TEST SIMILARITY ANALYSIS")
    print("=" * 40)
    
    # Crea CSV di test
    csv_file = create_test_csv()
    
    # Testa gli script
    success = test_analysis_scripts(csv_file)
    
    if success:
        print(f"\n✅ TUTTI I TEST PASSATI!")
        print(f"📁 File di test: {csv_file}")
        print(f"📑 PDFs generati: analyze_dag_csv_pdf output + quick_dag_analysis.pdf")
    else:
        print(f"\n❌ ALCUNI TEST FALLITI!")
    
    # Pulizia opzionale
    cleanup = input(f"\n🗑️  Rimuovere file di test? (y/N): ").strip().lower()
    if cleanup == 'y':
        try:
            os.remove(csv_file)
            if os.path.exists('quick_dag_analysis.pdf'):
                os.remove('quick_dag_analysis.pdf')
            print(f"✅ File di test rimossi")
        except Exception as e:
            print(f"⚠️ Errore rimozione: {e}")

if __name__ == "__main__":
    main()
