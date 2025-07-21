#!/usr/bin/env python3
"""
Test semplice per il FL analyzer
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo
import matplotlib.pyplot as plt
import numpy as np

def test_basic_loading():
    """Test base del caricamento dati."""
    try:
        print("🔍 Test caricamento CSV...")
        data = pd.read_csv('3000_results.csv', header=None)
        print(f"✅ Dati caricati: {data.shape}")
        print(f"📊 Prime righe:")
        print(data.head())
        
        # Test creazione grafico semplice
        print("\n🔍 Test creazione grafico...")
        plt.figure(figsize=(10, 6))
        plt.plot(data.iloc[:, 0], 'b-', linewidth=2)
        plt.title('Accuracy Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Grafico di test salvato: test_plot.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Test FL Results Analyzer")
    print("=" * 40)
    
    if test_basic_loading():
        print("\n✅ Test completato con successo!")
        
        # Ora prova a importare la classe principale
        try:
            print("\n🔍 Test import classe FLResultsAnalyzer...")
            from fl_results_analyzer import FLResultsAnalyzer
            print("✅ Import riuscito!")
            
            # Test istanza
            analyzer = FLResultsAnalyzer()
            print("✅ Istanza creata!")
            
            # Test caricamento dati
            if analyzer.load_data():
                print("✅ Caricamento dati riuscito!")
                analyzer.generate_summary_stats()
            else:
                print("❌ Errore nel caricamento dati")
                
        except Exception as e:
            print(f"❌ Errore nell'import: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ Test fallito!")
