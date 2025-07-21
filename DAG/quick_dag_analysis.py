#!/usr/bin/env python3
"""
Quick DAG Analysis Script
========================

Script rapido per analizzare velocemente i dati DAG Trust.
Versione semplificata di analyze_dag_csv.py per analisi veloci.

Uso: python quick_dag_analysis.py [file.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def quick_analysis(csv_file="dag_trust_analysis.csv"):
    """Analisi rapida del file CSV."""
    
    try:
        # Carica dati
        df = pd.read_csv(csv_file)
        print(f"✅ Caricati {len(df)} rounds da {csv_file}")
        
        # Trova numero di peer
        peer_cols = [col for col in df.columns if col.startswith('Peer_') and col.endswith('_Trust')]
        num_peers = len(peer_cols)
        print(f"👥 Peers: {num_peers}")
        
        # Identifica peer maliciosi
        malicious_peers = []
        for i in range(num_peers):
            if df[f'Peer_{i}_Is_Malicious'].iloc[0] == 1:
                malicious_peers.append(i)
        print(f"🔴 Peers maliciosi: {malicious_peers}")
        
        # Crea grafico rapido
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        rounds = df['Round']
        
        # 1. Evoluzione nodi
        ax1.plot(rounds, df['Total_Nodes'], 'b-', linewidth=2, label='Nodi Totali')
        ax1.plot(rounds, df['Total_Tips'], 'r--', linewidth=2, label='Tips')
        ax1.set_title('Evoluzione DAG')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trust evolution (solo alcuni peer)
        colors = ['green', 'blue', 'orange', 'purple', 'brown']
        for i in range(min(5, num_peers)):  # Max 5 peer
            trust_col = f'Peer_{i}_Trust'
            is_malicious = i in malicious_peers
            style = '--' if is_malicious else '-'
            color = 'red' if is_malicious else colors[i % len(colors)]
            label = f'Peer {i}' + (' (MAL)' if is_malicious else '')
            ax2.plot(rounds, df[trust_col], color=color, linestyle=style, 
                    linewidth=2, label=label)
        ax2.set_title('Trust Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # 3. Detection stats
        ax3.plot(rounds, df['Total_Malicious_Confirmed'], 'red', linewidth=3, label='Maliciosi Confermati')
        ax3.plot(rounds, df['Total_Good_Removed'], 'orange', linewidth=3, label='Buoni Rimossi')
        ax3.set_title('Detection Statistics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Round activity
        ax4.bar(rounds, df['Good_Nodes_Removed_This_Round'], alpha=0.7, color='orange', label='Buoni Rimossi')
        ax4.bar(rounds, df['Malicious_Nodes_Confirmed_This_Round'], alpha=0.9, color='red', label='Maliciosi Confermati')
        ax4.set_title('Activity per Round')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_dag_analysis.png', dpi=150, bbox_inches='tight')
        print("💾 Grafico salvato: quick_dag_analysis.png")
        plt.show()
        
        # Stats finali
        print("\n📊 STATISTICHE FINALI:")
        print(f"   • Nodi finali: {df['Total_Nodes'].iloc[-1]}")
        print(f"   • Maliciosi confermati: {df['Total_Malicious_Confirmed'].iloc[-1]}")
        print(f"   • Buoni rimossi: {df['Total_Good_Removed'].iloc[-1]}")
        
        # Trust finale
        print(f"\n🤝 TRUST FINALE:")
        for i in range(num_peers):
            trust_col = f'Peer_{i}_Trust'
            final_trust = df[trust_col].iloc[-1]
            status = "🔴" if i in malicious_peers else "🟢"
            print(f"   • Peer {i} {status}: {final_trust:.3f}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ File {csv_file} non trovato!")
        return False
    except Exception as e:
        print(f"❌ Errore: {e}")
        return False

def main():
    """Funzione principale."""
    print("⚡ Quick DAG Analysis")
    print("=" * 30)
    
    # Controlla argomenti
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "dag_trust_analysis.csv"
    
    print(f"📁 Analyzing: {csv_file}")
    
    success = quick_analysis(csv_file)
    
    if success:
        print("\n✅ Analisi completata!")
    else:
        print("\n❌ Analisi fallita!")

if __name__ == "__main__":
    main()
