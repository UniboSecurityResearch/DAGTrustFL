#!/usr/bin/env python3
"""
Quick DAG Analysis Script - PDF Version
=======================================

Script rapido per analizzare velocemente i dati DAG Trust.
Versione PDF di quick_dag_analysis.py per analisi veloci.

Uso: python quick_dag_analysis_pdf.py [file.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sys
import os
from datetime import datetime

def quick_analysis_pdf(csv_file="dag_trust_analysis.csv"):
    """Analisi rapida del file CSV con output PDF."""
    
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
        
        # Nome file PDF
        pdf_filename = 'quick_dag_analysis.pdf'
        
        with PdfPages(pdf_filename) as pdf:
            print(f"📑 Creazione PDF: {pdf_filename}")
            
            # Crea grafico rapido
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            rounds = df['Round']
            
            # 1. Evoluzione nodi
            ax1.plot(rounds, df['Total_Nodes'], 'b-', linewidth=3, marker='o', markersize=6, label='Nodi Totali')
            ax1.plot(rounds, df['Total_Tips'], 'r--', linewidth=3, marker='s', markersize=6, label='Tips')
            ax1.set_xlabel('Round', fontsize=12)
            ax1.set_ylabel('Numero di Nodi', fontsize=12)
            ax1.set_title('🔄 Evoluzione DAG', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # 2. Trust evolution (solo alcuni peer)
            colors = ['green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
            for i in range(min(8, num_peers)):  # Max 8 peer per leggibilità
                trust_col = f'Peer_{i}_Trust'
                is_malicious = i in malicious_peers
                style = '--' if is_malicious else '-'
                color = 'red' if is_malicious else colors[i % len(colors)]
                label = f'Peer {i}' + (' (MAL)' if is_malicious else ' (GOOD)')
                linewidth = 3 if is_malicious else 2
                ax2.plot(rounds, df[trust_col], color=color, linestyle=style, 
                        linewidth=linewidth, label=label, marker='o', markersize=4)
            
            ax2.set_xlabel('Round', fontsize=12)
            ax2.set_ylabel('Trust Value', fontsize=12)
            ax2.set_title('🤝 Trust Evolution', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            # 3. Detection stats
            ax3.plot(rounds, df['Total_Malicious_Confirmed'], 'red', linewidth=4, marker='o', 
                    markersize=6, label='Maliciosi Confermati')
            ax3.plot(rounds, df['Total_Good_Removed'], 'orange', linewidth=4, marker='^', 
                    markersize=6, label='Buoni Rimossi')
            ax3.set_xlabel('Round', fontsize=12)
            ax3.set_ylabel('Totale Cumulativo', fontsize=12)
            ax3.set_title('🎯 Detection Statistics', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            
            # 4. Round activity
            width = 0.35
            x_pos = np.arange(len(rounds))
            
            bars1 = ax4.bar(x_pos - width/2, df['Good_Nodes_Removed_This_Round'], 
                          width, alpha=0.8, color='orange', label='Buoni Rimossi')
            bars2 = ax4.bar(x_pos + width/2, df['Malicious_Nodes_Confirmed_This_Round'], 
                          width, alpha=0.9, color='red', label='Maliciosi Confermati')
            
            ax4.set_xlabel('Round', fontsize=12)
            ax4.set_ylabel('Nodi per Round', fontsize=12)
            ax4.set_title('📊 Activity per Round', fontsize=14, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(rounds)
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Titolo generale
            plt.suptitle(f'DAG Trust Analysis - Quick Overview\n({csv_file})', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Pagina 2: Statistiche dettagliate
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Calcola statistiche
            total_rounds = len(df)
            final_nodes = df['Total_Nodes'].iloc[-1]
            final_tips = df['Total_Tips'].iloc[-1]
            total_mal_confirmed = df['Total_Malicious_Confirmed'].iloc[-1]
            total_good_removed = df['Total_Good_Removed'].iloc[-1]
            
            # Calcola tassi
            mal_detection_rate = 0
            false_positive_rate = 0
            
            if len(malicious_peers) > 0:
                mal_detection_rate = (total_mal_confirmed / len(malicious_peers)) * 100
            
            good_peers = num_peers - len(malicious_peers)
            if good_peers > 0:
                false_positive_rate = (total_good_removed / good_peers) * 100
            
            # Trust finale per peer
            trust_finale_text = ""
            for i in range(num_peers):
                trust_col = f'Peer_{i}_Trust'
                final_trust = df[trust_col].iloc[-1]
                status = "MALICIOSO" if i in malicious_peers else "BUONO"
                trust_finale_text += f"• Peer {i} ({status}): {final_trust:.3f}\n"
            
            stats_text = f"""
QUICK DAG TRUST ANALYSIS - RISULTATI
===================================

FILE: {csv_file}
DATA ANALISI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATISTICHE GENERALI:
• Rounds totali: {total_rounds}
• Peers nel sistema: {num_peers}
• Nodi finali nel DAG: {final_nodes}
• Tips finali: {final_tips}

CONFIGURAZIONE PEERS:
• Peers maliciosi: {malicious_peers}
• Peers buoni: {[i for i in range(num_peers) if i not in malicious_peers]}

PERFORMANCE DETECTION:
• Nodi maliciosi confermati: {total_mal_confirmed}
• Nodi buoni rimossi (falsi positivi): {total_good_removed}
• Tasso detection maliciosi: {mal_detection_rate:.1f}%
• Tasso falsi positivi: {false_positive_rate:.1f}%

TRUST FINALE:
{trust_finale_text}

INTERPRETAZIONE:
{'✅ Sistema efficace!' if mal_detection_rate > 70 and false_positive_rate < 20 else '⚠️ Sistema da ottimizzare'}
• Detection Rate {'ALTA' if mal_detection_rate > 70 else 'BASSA' if mal_detection_rate < 30 else 'MEDIA'}
• False Positive Rate {'BASSO' if false_positive_rate < 10 else 'ALTO' if false_positive_rate > 30 else 'MEDIO'}
"""
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, 
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9))
            
            plt.title('Statistiche Dettagliate', fontsize=18, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"💾 PDF salvato: {pdf_filename}")
        
        # Stats finali in console
        print(f"\n📊 RISULTATI FINALI:")
        print(f"   • Nodi finali: {final_nodes}")
        print(f"   • Maliciosi confermati: {total_mal_confirmed}")
        print(f"   • Buoni rimossi: {total_good_removed}")
        print(f"   • Detection rate: {mal_detection_rate:.1f}%")
        print(f"   • False positive rate: {false_positive_rate:.1f}%")
        
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
    print("⚡ Quick DAG Analysis - PDF Edition")
    print("=" * 40)
    
    # Controlla argomenti
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("📁 Nome del file CSV (default: dag_trust_analysis.csv): ").strip()
        if not csv_file:
            csv_file = "dag_trust_analysis.csv"
    
    print(f"📁 Analyzing: {csv_file}")
    
    success = quick_analysis_pdf(csv_file)
    
    if success:
        print("\n✅ Analisi PDF completata!")
        print("📑 Controlla il file quick_dag_analysis.pdf")
    else:
        print("\n❌ Analisi fallita!")

if __name__ == "__main__":
    main()
