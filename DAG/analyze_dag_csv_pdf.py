#!/usr/bin/env python3
"""
DAG Trust Analysis Visualizer - PDF Version
===========================================

Questo script analizza i dati CSV generati dal sistema DAG Trust per federated learning
e crea visualizzazioni dettagliate dell'evoluzione del sistema nel tempo.
Tutti i grafici vengono salvati in formato PDF.

Autore: GitHub Copilot Assistant
Data: 21 luglio 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime

# Configurazione matplotlib per grafici più belli
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DAGAnalyzer:
    """Classe per analizzare e visualizzare i dati del DAG Trust system."""
    
    def __init__(self, csv_file="dag_trust_analysis.csv"):
        """
        Inizializza l'analyzer.
        
        Args:
            csv_file (str): Path del file CSV da analizzare
        """
        self.csv_file = csv_file
        self.df = None
        self.num_peers = 0
        self.output_dir = "dag_analysis_plots"
        
        # Crea directory per i grafici
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Carica i dati dal file CSV."""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Dati caricati: {len(self.df)} rounds, {len(self.df.columns)} colonne")
            
            # Determina il numero di peer
            peer_cols = [col for col in self.df.columns if col.startswith('Peer_') and col.endswith('_Trust')]
            self.num_peers = len(peer_cols)
            print(f"📊 Numero di peers rilevati: {self.num_peers}")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ File {self.csv_file} non trovato!")
            return False
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            return False
    
    def create_comprehensive_pdf_report(self):
        """Crea un singolo PDF con tutti i grafici e statistiche."""
        pdf_filename = f"{self.output_dir}/complete_dag_analysis_report.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            print("📑 Creazione PDF completo...")
            
            # 1. System Evolution
            print("   📊 System Evolution...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            rounds = self.df['Round']
            
            ax1.plot(rounds, self.df['Total_Nodes'], 'b-', linewidth=2, marker='o', label='Nodi Totali')
            ax1.plot(rounds, self.df['Total_Tips'], 'r--', linewidth=2, marker='s', label='Tips')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Numero di Nodi')
            ax1.set_title('🔄 Evoluzione del DAG')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(rounds, self.df['Good_Nodes_Removed_This_Round'], alpha=0.7, color='orange', label='Nodi Buoni Rimossi')
            ax2.bar(rounds, self.df['Malicious_Nodes_Confirmed_This_Round'], alpha=0.7, color='red', label='Nodi Maliciosi Confermati')
            ax2.bar(rounds, self.df['Corrupted_Nodes_Confirmed_This_Round'], alpha=0.7, color='purple', label='Nodi Corrotti Confermati')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Nodi per Round')
            ax2.set_title('📊 Nodi Rimossi/Confermati per Round')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(rounds, self.df['Total_Good_Removed'], 'orange', linewidth=3, label='Tot. Buoni Rimossi')
            ax3.plot(rounds, self.df['Total_Malicious_Confirmed'], 'red', linewidth=3, label='Tot. Maliciosi Confermati')
            ax3.plot(rounds, self.df['Total_Corrupted_Confirmed'], 'purple', linewidth=3, label='Tot. Corrotti Confermati')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Totale Cumulativo')
            ax3.set_title('📈 Evoluzione Cumulativa')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            if self.df['Total_Malicious_Confirmed'].max() > 0:
                malicious_detection_rate = (self.df['Total_Malicious_Confirmed'] / 
                                          (self.df['Total_Malicious_Confirmed'] + self.df['Total_Good_Removed']) * 100)
                ax4.plot(rounds, malicious_detection_rate, 'green', linewidth=3, marker='o')
                ax4.set_xlabel('Round')
                ax4.set_ylabel('Tasso di Detection (%)')
                ax4.set_title('🎯 Efficacia Detection Maliciosi')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 100)
            
            plt.suptitle('DAG Trust System Evolution', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 2. Trust Evolution
            print("   🤝 Trust Evolution...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_peers))
            
            for i in range(self.num_peers):
                trust_col = f'Peer_{i}_Trust'
                malicious_col = f'Peer_{i}_Is_Malicious'
                corrupted_col = f'Peer_{i}_Is_Corrupted'
                
                is_malicious = self.df[malicious_col].iloc[0] == 1
                is_corrupted = self.df[corrupted_col].iloc[0] == 1
                
                if is_malicious:
                    label = f'Peer {i} (Malicioso)'
                    linestyle = '--'
                    alpha = 0.8
                elif is_corrupted:
                    label = f'Peer {i} (Corrotto)'
                    linestyle = '-.'
                    alpha = 0.8
                else:
                    label = f'Peer {i} (Buono)'
                    linestyle = '-'
                    alpha = 1.0
                
                ax1.plot(rounds, self.df[trust_col], color=colors[i], linewidth=2.5, 
                        label=label, linestyle=linestyle, alpha=alpha, marker='o', markersize=4)
            
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Trust Value')
            ax1.set_title('Trust Vector Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.1)
            
            # Heatmap
            trust_matrix = []
            for i in range(self.num_peers):
                trust_col = f'Peer_{i}_Trust'
                trust_matrix.append(self.df[trust_col].values)
            
            trust_matrix = np.array(trust_matrix)
            im = ax2.imshow(trust_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Peer ID')
            ax2.set_title('Trust Values Heatmap')
            
            # Etichette peer
            for i in range(self.num_peers):
                malicious_col = f'Peer_{i}_Is_Malicious'
                corrupted_col = f'Peer_{i}_Is_Corrupted'
                is_malicious = self.df[malicious_col].iloc[0] == 1
                is_corrupted = self.df[corrupted_col].iloc[0] == 1
                
                if is_malicious:
                    ax2.text(-2, i, 'M', fontsize=12, ha='center', va='center', color='red', fontweight='bold')
                elif is_corrupted:
                    ax2.text(-2, i, 'C', fontsize=12, ha='center', va='center', color='purple', fontweight='bold')
                else:
                    ax2.text(-2, i, 'G', fontsize=12, ha='center', va='center', color='green', fontweight='bold')
            
            plt.colorbar(im, ax=ax2, label='Trust Value')
            plt.suptitle('Trust Vector Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 3. Consistency & Contributions
            print("   🏗️ Consistency & Contributions...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            for i in range(self.num_peers):
                struct_col = f'Peer_{i}_Structural_Consistency'
                malicious_col = f'Peer_{i}_Is_Malicious'
                is_malicious = self.df[malicious_col].iloc[0] == 1
                
                linestyle = '--' if is_malicious else '-'
                alpha = 0.7 if is_malicious else 1.0
                
                ax1.plot(rounds, self.df[struct_col], color=colors[i], linewidth=2, 
                        label=f'Peer {i}', linestyle=linestyle, alpha=alpha, marker='o')
            
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Structural Consistency')
            ax1.set_title('Structural Consistency nel Tempo')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.1)
            
            for i in range(self.num_peers):
                updates_col = f'Peer_{i}_Updates_Contribution'
                malicious_col = f'Peer_{i}_Is_Malicious'
                is_malicious = self.df[malicious_col].iloc[0] == 1
                
                linestyle = '--' if is_malicious else '-'
                alpha = 0.7 if is_malicious else 1.0
                
                ax2.plot(rounds, self.df[updates_col], color=colors[i], linewidth=2, 
                        label=f'Peer {i}', linestyle=linestyle, alpha=alpha, marker='s')
            
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Updates Contribution')
            ax2.set_title('Updates Contribution nel Tempo')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)
            
            # Distribuzioni finali
            final_struct = []
            final_updates = []
            labels = []
            colors_final = []
            for i in range(self.num_peers):
                struct_col = f'Peer_{i}_Structural_Consistency'
                updates_col = f'Peer_{i}_Updates_Contribution'
                malicious_col = f'Peer_{i}_Is_Malicious'
                
                final_struct.append(self.df[struct_col].iloc[-1])
                final_updates.append(self.df[updates_col].iloc[-1])
                
                is_malicious = self.df[malicious_col].iloc[0] == 1
                labels.append(f'Peer {i}')
                colors_final.append('red' if is_malicious else 'green')
            
            ax3.bar(labels, final_struct, color=colors_final, alpha=0.7)
            ax3.set_ylabel('Final Structural Consistency')
            ax3.set_title('Structural Consistency Finale')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
            
            ax4.bar(labels, final_updates, color=colors_final, alpha=0.7)
            ax4.set_ylabel('Final Updates Contribution')
            ax4.set_title('Updates Contribution Finale')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.1)
            
            plt.suptitle('Consistency & Contributions Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 4. Detection Analysis
            print("   🎯 Detection Analysis...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            total_malicious = sum(1 for i in range(self.num_peers) 
                                if self.df[f'Peer_{i}_Is_Malicious'].iloc[0] == 1)
            total_corrupted = sum(1 for i in range(self.num_peers) 
                                if self.df[f'Peer_{i}_Is_Corrupted'].iloc[0] == 1)
            
            if total_malicious > 0:
                detection_rate_mal = (self.df['Total_Malicious_Confirmed'] / total_malicious) * 100
                ax1.plot(rounds, detection_rate_mal, 'red', linewidth=3, marker='o', label='Maliciosi')
            
            if total_corrupted > 0:
                detection_rate_cor = (self.df['Total_Corrupted_Confirmed'] / total_corrupted) * 100
                ax1.plot(rounds, detection_rate_cor, 'purple', linewidth=3, marker='s', label='Corrotti')
            
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Tasso di Detection (%)')
            ax1.set_title('Tasso di Detection nel Tempo')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)
            
            # Falsi positivi
            false_positive_rate = (self.df['Total_Good_Removed'] / 
                                 (self.num_peers - total_malicious - total_corrupted)) * 100
            ax2.plot(rounds, false_positive_rate, 'orange', linewidth=3, marker='^')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Falsi Positivi (%)')
            ax2.set_title('Tasso di Falsi Positivi')
            ax2.grid(True, alpha=0.3)
            
            # Trust finale - distribuzione
            final_trust_good = []
            final_trust_bad = []
            
            for i in range(self.num_peers):
                trust_col = f'Peer_{i}_Trust'
                malicious_col = f'Peer_{i}_Is_Malicious'
                corrupted_col = f'Peer_{i}_Is_Corrupted'
                
                final_trust = self.df[trust_col].iloc[-1]
                is_malicious = self.df[malicious_col].iloc[0] == 1
                is_corrupted = self.df[corrupted_col].iloc[0] == 1
                
                if is_malicious or is_corrupted:
                    final_trust_bad.append(final_trust)
                else:
                    final_trust_good.append(final_trust)
            
            if final_trust_good and final_trust_bad:
                data_to_plot = [final_trust_good, final_trust_bad]
                bp = ax3.boxplot(data_to_plot, labels=['Nodi Buoni', 'Nodi Maliciosi/Corrotti'], patch_artist=True)
                bp['boxes'][0].set_facecolor('green')
                bp['boxes'][0].set_alpha(0.7)
                bp['boxes'][1].set_facecolor('red')
                bp['boxes'][1].set_alpha(0.7)
            
            ax3.set_ylabel('Final Trust Value')
            ax3.set_title('Distribuzione Trust Finale')
            ax3.grid(True, alpha=0.3)
            
            # Timeline eventi
            ax4.bar(rounds, self.df['Good_Nodes_Removed_This_Round'], color='orange', alpha=0.6, label='Buoni Rimossi')
            ax4.bar(rounds, self.df['Malicious_Nodes_Confirmed_This_Round'], color='red', alpha=0.8, label='Maliciosi Confermati')
            ax4.set_xlabel('Round')
            ax4.set_ylabel('Numero di Nodi')
            ax4.set_title('Timeline Eventi')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Detection Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 5. Summary Statistics Page
            print("   📋 Summary Statistics...")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Statistiche
            total_rounds = len(self.df)
            final_nodes = self.df['Total_Nodes'].iloc[-1]
            final_tips = self.df['Total_Tips'].iloc[-1]
            
            malicious_peers = []
            corrupted_peers = []
            for i in range(self.num_peers):
                if self.df[f'Peer_{i}_Is_Malicious'].iloc[0] == 1:
                    malicious_peers.append(i)
                if self.df[f'Peer_{i}_Is_Corrupted'].iloc[0] == 1:
                    corrupted_peers.append(i)
            
            total_mal_confirmed = self.df['Total_Malicious_Confirmed'].iloc[-1]
            total_cor_confirmed = self.df['Total_Corrupted_Confirmed'].iloc[-1]
            total_good_removed = self.df['Total_Good_Removed'].iloc[-1]
            
            stats_text = f"""
DAG TRUST SYSTEM - STATISTICHE RIASSUNTIVE
==========================================

SISTEMA:
• Rounds totali: {total_rounds}
• Peers nel sistema: {self.num_peers}
• Nodi finali nel DAG: {final_nodes}
• Tips finali: {final_tips}

PEERS MALICIOSI: {malicious_peers}
PEERS CORROTTI: {corrupted_peers}

DETECTION:
• Nodi maliciosi confermati: {total_mal_confirmed}
• Nodi corrotti confermati: {total_cor_confirmed}
• Nodi buoni rimossi (falsi positivi): {total_good_removed}
"""
            
            if len(malicious_peers) > 0:
                mal_detection_rate = (total_mal_confirmed / len(malicious_peers)) * 100
                stats_text += f"• Tasso detection maliciosi: {mal_detection_rate:.1f}%\n"
            
            if len(corrupted_peers) > 0:
                cor_detection_rate = (total_cor_confirmed / len(corrupted_peers)) * 100
                stats_text += f"• Tasso detection corrotti: {cor_detection_rate:.1f}%\n"
            
            good_peers = self.num_peers - len(malicious_peers) - len(corrupted_peers)
            if good_peers > 0:
                false_positive_rate = (total_good_removed / good_peers) * 100
                stats_text += f"• Tasso falsi positivi: {false_positive_rate:.1f}%\n"
            
            stats_text += f"\nTRUST FINALE:\n"
            for i in range(self.num_peers):
                trust_col = f'Peer_{i}_Trust'
                final_trust = self.df[trust_col].iloc[-1]
                
                if i in malicious_peers:
                    status = "MALICIOSO"
                elif i in corrupted_peers:
                    status = "CORROTTO"
                else:
                    status = "BUONO"
                
                stats_text += f"• Peer {i} ({status}): {final_trust:.3f}\n"
            
            stats_text += f"\nReport generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            stats_text += f"\nFile CSV: {self.csv_file}"
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, 
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            
            plt.title('Statistiche Riassuntive', fontsize=18, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"📑 PDF completo salvato: {pdf_filename}")
        return pdf_filename
    
    def generate_summary_stats(self):
        """Genera statistiche riassuntive in console."""
        print("\n" + "="*60)
        print("📊 STATISTICHE RIASSUNTIVE DEL SISTEMA DAG TRUST")
        print("="*60)
        
        # Statistiche generali
        total_rounds = len(self.df)
        final_nodes = self.df['Total_Nodes'].iloc[-1]
        final_tips = self.df['Total_Tips'].iloc[-1]
        
        print(f"\n🔄 SISTEMA:")
        print(f"   • Rounds totali: {total_rounds}")
        print(f"   • Peers nel sistema: {self.num_peers}")
        print(f"   • Nodi finali nel DAG: {final_nodes}")
        print(f"   • Tips finali: {final_tips}")
        
        # Analisi peer maliciosi
        malicious_peers = []
        corrupted_peers = []
        for i in range(self.num_peers):
            if self.df[f'Peer_{i}_Is_Malicious'].iloc[0] == 1:
                malicious_peers.append(i)
            if self.df[f'Peer_{i}_Is_Corrupted'].iloc[0] == 1:
                corrupted_peers.append(i)
        
        print(f"\n🔴 PEERS MALICIOSI: {malicious_peers}")
        print(f"🟣 PEERS CORROTTI: {corrupted_peers}")
        
        # Statistiche detection
        total_mal_confirmed = self.df['Total_Malicious_Confirmed'].iloc[-1]
        total_cor_confirmed = self.df['Total_Corrupted_Confirmed'].iloc[-1]
        total_good_removed = self.df['Total_Good_Removed'].iloc[-1]
        
        print(f"\n🎯 DETECTION:")
        print(f"   • Nodi maliciosi confermati: {total_mal_confirmed}")
        print(f"   • Nodi corrotti confermati: {total_cor_confirmed}")
        print(f"   • Nodi buoni rimossi (falsi positivi): {total_good_removed}")
        
        # Calcola tassi
        if len(malicious_peers) > 0:
            mal_detection_rate = (total_mal_confirmed / len(malicious_peers)) * 100
            print(f"   • Tasso detection maliciosi: {mal_detection_rate:.1f}%")
        
        if len(corrupted_peers) > 0:
            cor_detection_rate = (total_cor_confirmed / len(corrupted_peers)) * 100
            print(f"   • Tasso detection corrotti: {cor_detection_rate:.1f}%")
        
        good_peers = self.num_peers - len(malicious_peers) - len(corrupted_peers)
        if good_peers > 0:
            false_positive_rate = (total_good_removed / good_peers) * 100
            print(f"   • Tasso falsi positivi: {false_positive_rate:.1f}%")
        
        # Analisi trust finale
        print(f"\n🤝 TRUST FINALE:")
        for i in range(self.num_peers):
            trust_col = f'Peer_{i}_Trust'
            final_trust = self.df[trust_col].iloc[-1]
            
            status = ""
            if i in malicious_peers:
                status = "🔴"
            elif i in corrupted_peers:
                status = "🟣"
            else:
                status = "🟢"
            
            print(f"   • Peer {i} {status}: {final_trust:.3f}")
        
        print("\n" + "="*60)
    
    def create_comprehensive_report(self):
        """Crea un report completo con PDF unificato."""
        print("\n🚀 Generazione report completo PDF...")
        
        if not self.load_data():
            return False
        
        # Crea PDF unificato
        pdf_filename = self.create_comprehensive_pdf_report()
        
        # Mostra statistiche in console
        self.generate_summary_stats()
        
        # Salva statistiche in file txt
        with open(f"{self.output_dir}/summary_stats.txt", "w") as f:
            f.write(f"DAG Trust Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV File: {self.csv_file}\n\n")
            f.write(f"Total Rounds: {len(self.df)}\n")
            f.write(f"Total Peers: {self.num_peers}\n")
            f.write(f"Final Nodes: {self.df['Total_Nodes'].iloc[-1]}\n")
            f.write(f"Final Tips: {self.df['Total_Tips'].iloc[-1]}\n")
        
        print(f"\n✅ Report PDF completo generato!")
        print(f"📁 File principale: {pdf_filename}")
        print(f"📂 Directory: {self.output_dir}/")
        
        return True


def main():
    """Funzione principale."""
    print("🔍 DAG Trust Analysis Visualizer - PDF Edition")
    print("=" * 50)
    
    # Chiedi il file CSV
    csv_file = input("📁 Nome del file CSV (default: dag_trust_analysis.csv): ").strip()
    if not csv_file:
        csv_file = "dag_trust_analysis.csv"
    
    # Crea l'analyzer
    analyzer = DAGAnalyzer(csv_file)
    
    # Genera il report completo
    success = analyzer.create_comprehensive_report()
    
    if success:
        print("\n🎉 Analisi PDF completata con successo!")
    else:
        print("\n❌ Errore nell'analisi!")
    
    return success


if __name__ == "__main__":
    main()
