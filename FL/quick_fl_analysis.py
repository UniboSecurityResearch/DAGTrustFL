#!/usr/bin/env python3
"""
Quick FL Results Analysis
========================

Analisi rapida dei risultati di Federated Learning.
Versione veloce per controlli rapidi.

Uso: python quick_fl_analysis.py [file.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sys
from datetime import datetime

def quick_fl_analysis(csv_file="3000_results.csv"):
    """Analisi rapida dei risultati FL con output PDF."""
    
    try:
        # Carica dati
        raw_data = pd.read_csv(csv_file, header=None)
        print(f"✅ Caricati {len(raw_data)} epoche da {csv_file}")
        
        # Estrai metriche principali
        epochs = range(1, len(raw_data) + 1)
        accuracy = raw_data.iloc[:, 0]
        loss = raw_data.iloc[:, 1]
        
        # Calcola precision/recall medie
        precision_cols = raw_data.iloc[:, 2:12]
        recall_cols = raw_data.iloc[:, 12:22]
        avg_precision = precision_cols.mean(axis=1)
        avg_recall = recall_cols.mean(axis=1)
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        
        print(f"🎯 Accuracy finale: {accuracy.iloc[-1]:.2f}%")
        print(f"📉 Loss finale: {loss.iloc[-1]:.4f}")
        print(f"📊 F1-Score finale: {avg_f1.iloc[-1]:.3f}")
        
        # Crea PDF
        pdf_filename = 'quick_fl_analysis.pdf'
        
        with PdfPages(pdf_filename) as pdf:
            print(f"📑 Creazione PDF: {pdf_filename}")
            
            # Pagina 1: Overview principale
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy
            ax1.plot(epochs, accuracy, 'b-', linewidth=3, marker='o', markersize=5)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
            ax1.set_title('🎯 Model Accuracy Evolution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.text(0.05, 0.95, f'Final: {accuracy.iloc[-1]:.2f}%', 
                    transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # Loss
            ax2.plot(epochs, loss, 'r-', linewidth=3, marker='s', markersize=5)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('📉 Model Loss Evolution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.text(0.05, 0.95, f'Final: {loss.iloc[-1]:.4f}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            
            # Metriche aggregate
            ax3.plot(epochs, avg_precision, 'g-', linewidth=3, marker='^', markersize=4, label='Precision')
            ax3.plot(epochs, avg_recall, 'orange', linewidth=3, marker='v', markersize=4, label='Recall')
            ax3.plot(epochs, avg_f1, 'purple', linewidth=3, marker='d', markersize=4, label='F1-Score')
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Score', fontsize=12)
            ax3.set_title('📊 Average Precision, Recall & F1', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
            
            # Per-class precision finale
            final_precision = [raw_data.iloc[-1, 2+i] for i in range(10)]
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            bars = ax4.bar(range(10), final_precision, color=colors, alpha=0.8, edgecolor='black')
            ax4.set_xlabel('Class', fontsize=12)
            ax4.set_ylabel('Final Precision', fontsize=12)
            ax4.set_title('🎯 Final Precision per Class', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(10))
            ax4.set_xticklabels([f'C{i}' for i in range(10)])
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Aggiungi valori sopra le barre
            for bar, value in zip(bars, final_precision):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.suptitle(f'Federated Learning Quick Analysis\n({csv_file})', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Pagina 2: Heatmaps e statistiche
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Heatmap Precision
            precision_matrix = []
            for i in range(10):
                precision_matrix.append(raw_data.iloc[:, 2+i].values)
            precision_matrix = np.array(precision_matrix)
            
            im1 = ax1.imshow(precision_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Class')
            ax1.set_title('🔥 Precision Evolution Heatmap')
            ax1.set_yticks(range(10))
            ax1.set_yticklabels([f'C{i}' for i in range(10)])
            plt.colorbar(im1, ax=ax1, label='Precision')
            
            # Heatmap Recall
            recall_matrix = []
            for i in range(10):
                recall_matrix.append(raw_data.iloc[:, 12+i].values)
            recall_matrix = np.array(recall_matrix)
            
            im2 = ax2.imshow(recall_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Class')
            ax2.set_title('🔥 Recall Evolution Heatmap')
            ax2.set_yticks(range(10))
            ax2.set_yticklabels([f'C{i}' for i in range(10)])
            plt.colorbar(im2, ax=ax2, label='Recall')
            
            # Confronto Precision vs Recall finale
            final_recall = [raw_data.iloc[-1, 12+i] for i in range(10)]
            ax3.scatter(final_precision, final_recall, c=colors, s=100, alpha=0.8, edgecolors='black')
            for i in range(10):
                ax3.annotate(f'C{i}', (final_precision[i], final_recall[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
            
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
            ax3.set_xlabel('Final Precision')
            ax3.set_ylabel('Final Recall')
            ax3.set_title('⚖️ Precision vs Recall (Final)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 1.05)
            ax3.set_ylim(0, 1.05)
            
            # Learning rate analysis
            # Calcola "learning rate" come variazione di accuracy
            accuracy_diff = np.diff(accuracy.values)
            smoothed_diff = np.convolve(accuracy_diff, np.ones(5)/5, mode='valid')
            
            ax4.plot(range(2, len(epochs)-2), smoothed_diff, 'navy', linewidth=3, marker='o', markersize=4)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy Change (smoothed)')
            ax4.set_title('📈 Learning Progress Rate')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Detailed Analysis & Heatmaps', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Pagina 3: Statistiche riassuntive
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Calcola statistiche
            accuracy_improvement = accuracy.iloc[-1] - accuracy.iloc[0]
            loss_reduction = loss.iloc[0] - loss.iloc[-1]
            best_class_prec = np.argmax(final_precision)
            worst_class_prec = np.argmin(final_precision)
            best_class_rec = np.argmax(final_recall)
            worst_class_rec = np.argmin(final_recall)
            
            # Calcola F1 per classe
            f1_per_class = []
            for i in range(10):
                p = final_precision[i]
                r = final_recall[i]
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                f1_per_class.append(f1)
            
            stats_text = f"""
FEDERATED LEARNING - QUICK ANALYSIS REPORT
==========================================

FILE: {csv_file}
ANALISI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
EPOCHE: {len(raw_data)}

PERFORMANCE GLOBALE:
• Accuracy finale: {accuracy.iloc[-1]:.2f}%
• Loss finale: {loss.iloc[-1]:.4f}
• Precision media finale: {avg_precision.iloc[-1]:.3f}
• Recall media finale: {avg_recall.iloc[-1]:.3f}
• F1-Score medio finale: {avg_f1.iloc[-1]:.3f}

EVOLUZIONE TRAINING:
• Miglioramento accuracy: {accuracy_improvement:+.2f}%
• Riduzione loss: {loss_reduction:+.4f}
• Stabilità: {'STABILE' if accuracy.std() < 2 else 'INSTABILE'}

ANALISI PER CLASSE (FINALE):
• Migliore Precision: Classe {best_class_prec} ({final_precision[best_class_prec]:.3f})
• Peggiore Precision: Classe {worst_class_prec} ({final_precision[worst_class_prec]:.3f})
• Migliore Recall: Classe {best_class_rec} ({final_recall[best_class_rec]:.3f})
• Peggiore Recall: Classe {worst_class_rec} ({final_recall[worst_class_rec]:.3f})

DETTAGLI PER CLASSE:
"""
            
            for i in range(10):
                stats_text += f"• Classe {i}: P={final_precision[i]:.3f}, R={final_recall[i]:.3f}, F1={f1_per_class[i]:.3f}\n"
            
            # Valutazione qualitativa
            if accuracy.iloc[-1] > 95:
                quality = "🌟 ECCELLENTE"
            elif accuracy.iloc[-1] > 90:
                quality = "✅ BUONA"
            elif accuracy.iloc[-1] > 80:
                quality = "⚠️ MEDIA"
            else:
                quality = "❌ DA MIGLIORARE"
            
            balance_score = max(final_precision) - min(final_precision)
            balance = "✅ BILANCIATO" if balance_score < 0.2 else "⚠️ SBILANCIATO"
            
            stats_text += f"""
VALUTAZIONE COMPLESSIVA:
• Performance: {quality}
• Bilanciamento classi: {balance}
• Trend: {'📈 MIGLIORANDO' if accuracy_improvement > 0 else '📉 PEGGIORANDO'}

RACCOMANDAZIONI:
"""
            
            if accuracy.iloc[-1] < 85:
                stats_text += "• Aumentare numero epoche o ottimizzare iperparametri\n"
            if balance_score > 0.3:
                stats_text += "• Implementare class balancing o weighted sampling\n"
            if accuracy.std() > 3:
                stats_text += "• Migliorare stabilità training (learning rate, batch size)\n"
            if accuracy.iloc[-1] > 95:
                stats_text += "• Performance ottima! Sistema pronto per deployment\n"
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, 
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.9))
            
            plt.title('Quick Analysis Summary', fontsize=18, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"💾 PDF salvato: {pdf_filename}")
        
        # Stats in console
        print(f"\n📊 RIASSUNTO VELOCE:")
        print(f"   • Performance: {quality}")
        print(f"   • Miglioramento: {accuracy_improvement:+.2f}%")
        print(f"   • Classe migliore: {best_class_prec} (P:{final_precision[best_class_prec]:.2f})")
        print(f"   • Classe peggiore: {worst_class_prec} (P:{final_precision[worst_class_prec]:.2f})")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ File {csv_file} non trovato!")
        return False
    except Exception as e:
        print(f"❌ Errore: {e}")
        return False

def main():
    """Funzione principale."""
    print("⚡ Quick FL Results Analysis")
    print("=" * 35)
    
    # Controlla argomenti
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("📁 Nome del file CSV (default: 3000_results.csv): ").strip()
        if not csv_file:
            csv_file = "3000_results.csv"
    
    print(f"📁 Analyzing: {csv_file}")
    
    success = quick_fl_analysis(csv_file)
    
    if success:
        print("\n✅ Quick analysis completata!")
        print("📑 Controlla il file quick_fl_analysis.pdf")
    else:
        print("\n❌ Analisi fallita!")

if __name__ == "__main__":
    main()
