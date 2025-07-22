#!/usr/bin/env python3
"""
Federated Learning Results Analyzer
===================================

Questo script analizza i risultati CSV del federated learning e crea visualizzazioni
dettagliate dell'evoluzione delle metriche durante il training.

Struttura CSV:
- Column 0: Accuracy (%)
- Column 1: Loss
- Columns 2-11: Class Precision (10 classi)
- Columns 12-21: Class Recall (10 classi)

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

# Configurazione matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FLResultsAnalyzer:
    """Classe per analizzare i risultati del Federated Learning."""
    
    def __init__(self, csv_file="3000_results.csv"):
        """
        Inizializza l'analyzer.
        
        Args:
            csv_file (str): Path del file CSV da analizzare
        """
        self.csv_file = csv_file
        self.df = None
        self.output_dir = "dag_analysis_plots"
        self.num_classes = 10
        
        # Crea directory per i grafici
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Carica e struttura i dati dal file CSV."""
        try:
            # Carica il CSV senza header
            raw_data = pd.read_csv(self.csv_file, header=None)
            
            print(f"✅ Dati caricati: {len(raw_data)} epoche, {raw_data.shape[1]} colonne")
            
            # Struttura i dati
            self.df = pd.DataFrame({
                'epoch': range(1, len(raw_data) + 1),
                'accuracy': raw_data.iloc[:, 0],
                'loss': raw_data.iloc[:, 1]
            })
            
            # Aggiungi precision per ogni classe
            for i in range(self.num_classes):
                self.df[f'precision_class_{i}'] = raw_data.iloc[:, 2 + i]
            
            # Aggiungi recall per ogni classe
            for i in range(self.num_classes):
                self.df[f'recall_class_{i}'] = raw_data.iloc[:, 12 + i]
            
            # Calcola metriche aggregate
            precision_cols = [f'precision_class_{i}' for i in range(self.num_classes)]
            recall_cols = [f'recall_class_{i}' for i in range(self.num_classes)]
            
            self.df['avg_precision'] = self.df[precision_cols].mean(axis=1)
            self.df['avg_recall'] = self.df[recall_cols].mean(axis=1)
            self.df['avg_f1_score'] = 2 * (self.df['avg_precision'] * self.df['avg_recall']) / (self.df['avg_precision'] + self.df['avg_recall'])
            
            print(f"📊 Epoche analizzate: {len(self.df)}")
            print(f"🎯 Range accuracy: {self.df['accuracy'].min():.2f}% - {self.df['accuracy'].max():.2f}%")
            print(f"📉 Range loss: {self.df['loss'].min():.4f} - {self.df['loss'].max():.4f}")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ File {self.csv_file} non trovato!")
            return False
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            return False
    
    def create_comprehensive_pdf_report(self):
        """Crea un PDF completo con tutti i grafici."""
        pdf_filename = f"{self.output_dir}/fl_results_complete_analysis.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            print("📑 Creazione report PDF completo...")
            
            # 1. Overview Principale - Accuracy e Loss
            print("   📈 Main Metrics Overview...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            epochs = self.df['epoch']
            
            # Accuracy nel tempo
            ax1.plot(epochs, self.df['accuracy'], 'b-', linewidth=3, marker='o', markersize=4)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('🎯 Global Model Accuracy Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(self.df['accuracy'].min() - 1, self.df['accuracy'].max() + 1)
            
            # Loss nel tempo
            ax2.plot(epochs, self.df['loss'], 'r-', linewidth=3, marker='s', markersize=4)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('📉 Global Model Loss Evolution')
            ax2.grid(True, alpha=0.3)
            
            # Precision media nel tempo
            ax3.plot(epochs, self.df['avg_precision'], 'g-', linewidth=3, marker='^', markersize=4, label='Avg Precision')
            ax3.plot(epochs, self.df['avg_recall'], 'orange', linewidth=3, marker='v', markersize=4, label='Avg Recall')
            ax3.plot(epochs, self.df['avg_f1_score'], 'purple', linewidth=3, marker='d', markersize=4, label='Avg F1-Score')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.set_title('📊 Average Precision, Recall & F1-Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
            
            # Distribuzione Accuracy
            ax4.hist(self.df['accuracy'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(self.df['accuracy'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {self.df["accuracy"].mean():.2f}%')
            ax4.set_xlabel('Accuracy (%)')
            ax4.set_ylabel('Frequenza')
            ax4.set_title('📈 Distribuzione Accuracy')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Federated Learning - Main Metrics Overview', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 2. Per-Class Precision Analysis
            print("   🎯 Per-Class Precision Analysis...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Precision per classe nel tempo
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
            for i in range(self.num_classes):
                ax1.plot(epochs, self.df[f'precision_class_{i}'], color=colors[i], 
                        linewidth=2, marker='o', markersize=3, label=f'Class {i}', alpha=0.8)
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Precision')
            ax1.set_title('🎯 Per-Class Precision Evolution')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            
            # Heatmap Precision
            precision_matrix = []
            for i in range(self.num_classes):
                precision_matrix.append(self.df[f'precision_class_{i}'].values)
            
            precision_matrix = np.array(precision_matrix)
            im1 = ax2.imshow(precision_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Class')
            ax2.set_title('🔥 Precision Heatmap')
            ax2.set_yticks(range(self.num_classes))
            ax2.set_yticklabels([f'Class {i}' for i in range(self.num_classes)])
            plt.colorbar(im1, ax=ax2, label='Precision')
            
            # Precision finale per classe
            final_precision = [self.df[f'precision_class_{i}'].iloc[-1] for i in range(self.num_classes)]
            bars = ax3.bar(range(self.num_classes), final_precision, 
                          color=colors, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Final Precision')
            ax3.set_title('📊 Final Precision per Class')
            ax3.set_xticks(range(self.num_classes))
            ax3.set_xticklabels([f'C{i}' for i in range(self.num_classes)])
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(0, 1.1)
            
            # Aggiungi valori sopra le barre
            for bar, value in zip(bars, final_precision):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Box plot precision per classe
            precision_data = [self.df[f'precision_class_{i}'].values for i in range(self.num_classes)]
            bp = ax4.boxplot(precision_data, labels=[f'C{i}' for i in range(self.num_classes)], 
                           patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Precision')
            ax4.set_title('📦 Precision Distribution per Class')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Per-Class Precision Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 3. Per-Class Recall Analysis
            print("   🔍 Per-Class Recall Analysis...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Recall per classe nel tempo
            for i in range(self.num_classes):
                ax1.plot(epochs, self.df[f'recall_class_{i}'], color=colors[i], 
                        linewidth=2, marker='s', markersize=3, label=f'Class {i}', alpha=0.8)
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Recall')
            ax1.set_title('🔍 Per-Class Recall Evolution')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            
            # Heatmap Recall
            recall_matrix = []
            for i in range(self.num_classes):
                recall_matrix.append(self.df[f'recall_class_{i}'].values)
            
            recall_matrix = np.array(recall_matrix)
            im2 = ax2.imshow(recall_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Class')
            ax2.set_title('🔥 Recall Heatmap')
            ax2.set_yticks(range(self.num_classes))
            ax2.set_yticklabels([f'Class {i}' for i in range(self.num_classes)])
            plt.colorbar(im2, ax=ax2, label='Recall')
            
            # Recall finale per classe
            final_recall = [self.df[f'recall_class_{i}'].iloc[-1] for i in range(self.num_classes)]
            bars = ax3.bar(range(self.num_classes), final_recall, 
                          color=colors, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Final Recall')
            ax3.set_title('📊 Final Recall per Class')
            ax3.set_xticks(range(self.num_classes))
            ax3.set_xticklabels([f'C{i}' for i in range(self.num_classes)])
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(0, 1.1)
            
            # Aggiungi valori sopra le barre
            for bar, value in zip(bars, final_recall):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Box plot recall per classe
            recall_data = [self.df[f'recall_class_{i}'].values for i in range(self.num_classes)]
            bp = ax4.boxplot(recall_data, labels=[f'C{i}' for i in range(self.num_classes)], 
                           patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Recall')
            ax4.set_title('📦 Recall Distribution per Class')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Per-Class Recall Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 4. Comparative Analysis
            print("   ⚖️ Comparative Analysis...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Precision vs Recall per classe (finale)
            ax1.scatter(final_precision, final_recall, c=colors, s=100, alpha=0.7, edgecolors='black')
            for i in range(self.num_classes):
                ax1.annotate(f'C{i}', (final_precision[i], final_recall[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)  # Diagonal reference
            ax1.set_xlabel('Final Precision')
            ax1.set_ylabel('Final Recall')
            ax1.set_title('🎯 Precision vs Recall (Final Values)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 1.1)
            ax1.set_ylim(0, 1.1)
            
            # F1-Score per classe nel tempo
            for i in range(self.num_classes):
                precision_col = self.df[f'precision_class_{i}']
                recall_col = self.df[f'recall_class_{i}']
                f1_score = 2 * (precision_col * recall_col) / (precision_col + recall_col)
                ax2.plot(epochs, f1_score, color=colors[i], linewidth=2, 
                        marker='d', markersize=3, label=f'Class {i}', alpha=0.8)
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('F1-Score')
            ax2.set_title('⭐ Per-Class F1-Score Evolution')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            # Varianza delle metriche nel tempo
            precision_std = [self.df[[f'precision_class_{i}' for i in range(self.num_classes)]].iloc[j].std() 
                           for j in range(len(epochs))]
            recall_std = [self.df[[f'recall_class_{i}' for i in range(self.num_classes)]].iloc[j].std() 
                        for j in range(len(epochs))]
            
            ax3.plot(epochs, precision_std, 'g-', linewidth=3, marker='o', label='Precision Std')
            ax3.plot(epochs, recall_std, 'orange', linewidth=3, marker='s', label='Recall Std')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Standard Deviation')
            ax3.set_title('📊 Inter-Class Variance Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Learning curves confronto
            ax4.plot(epochs, self.df['accuracy'], 'b-', linewidth=3, label='Accuracy (%)', marker='o')
            
            # Normalizza loss per visualizzazione insieme ad accuracy
            normalized_loss = ((self.df['loss'].max() - self.df['loss']) / 
                             (self.df['loss'].max() - self.df['loss'].min())) * 100
            ax4.plot(epochs, normalized_loss, 'r-', linewidth=3, label='Loss (normalized)', marker='s')
            ax4.plot(epochs, self.df['avg_f1_score'] * 100, 'purple', linewidth=3, label='F1-Score (%)', marker='^')
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Percentage / Normalized Score')
            ax4.set_title('📈 Learning Curves Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Comparative Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 5. Performance Summary
            print("   📋 Performance Summary...")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Calcola statistiche finali
            final_accuracy = self.df['accuracy'].iloc[-1]
            final_loss = self.df['loss'].iloc[-1]
            final_avg_precision = self.df['avg_precision'].iloc[-1]
            final_avg_recall = self.df['avg_recall'].iloc[-1]
            final_avg_f1 = self.df['avg_f1_score'].iloc[-1]
            
            accuracy_improvement = self.df['accuracy'].iloc[-1] - self.df['accuracy'].iloc[0]
            loss_reduction = self.df['loss'].iloc[0] - self.df['loss'].iloc[-1]
            
            # Trova migliore e peggiore classe
            best_precision_class = np.argmax(final_precision)
            worst_precision_class = np.argmin(final_precision)
            best_recall_class = np.argmax(final_recall)
            worst_recall_class = np.argmin(final_recall)
            
            # Stabilità (coefficiente di variazione)
            accuracy_cv = self.df['accuracy'].std() / self.df['accuracy'].mean()
            
            summary_text = f"""
FEDERATED LEARNING - PERFORMANCE SUMMARY
========================================

FILE ANALIZZATO: {self.csv_file}
DATA ANALISI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
EPOCHE TOTALI: {len(self.df)}

METRICHE FINALI:
• Accuracy: {final_accuracy:.2f}%
• Loss: {final_loss:.4f}
• Average Precision: {final_avg_precision:.3f}
• Average Recall: {final_avg_recall:.3f}
• Average F1-Score: {final_avg_f1:.3f}

EVOLUZIONE TRAINING:
• Miglioramento Accuracy: {accuracy_improvement:+.2f}%
• Riduzione Loss: {loss_reduction:+.4f}
• Stabilità Accuracy (CV): {accuracy_cv:.4f}

PERFORMANCE PER CLASSE:
• Migliore Precision: Classe {best_precision_class} ({final_precision[best_precision_class]:.3f})
• Peggiore Precision: Classe {worst_precision_class} ({final_precision[worst_precision_class]:.3f})
• Migliore Recall: Classe {best_recall_class} ({final_recall[best_recall_class]:.3f})
• Peggiore Recall: Classe {worst_recall_class} ({final_recall[worst_recall_class]:.3f})

ANALISI DETTAGLIATA PER CLASSE:
"""
            
            # Aggiungi dettagli per ogni classe
            for i in range(self.num_classes):
                f1_final = 2 * (final_precision[i] * final_recall[i]) / (final_precision[i] + final_recall[i])
                summary_text += f"• Classe {i}: P={final_precision[i]:.3f}, R={final_recall[i]:.3f}, F1={f1_final:.3f}\n"
            
            # Valutazione qualitativa
            quality = "ECCELLENTE" if final_accuracy > 95 else "BUONA" if final_accuracy > 90 else "MEDIA" if final_accuracy > 80 else "DA MIGLIORARE"
            consistency = "STABILE" if accuracy_cv < 0.02 else "MODERATAMENTE STABILE" if accuracy_cv < 0.05 else "INSTABILE"
            
            summary_text += f"""
VALUTAZIONE QUALITATIVA:
• Performance Globale: {quality}
• Stabilità Training: {consistency}
• Balance Classes: {'BILANCIATO' if max(final_precision) - min(final_precision) < 0.2 else 'SBILANCIATO'}

RACCOMANDAZIONI:
"""
            
            if final_accuracy < 90:
                summary_text += "• Considerare aumento epoche o tuning iperparametri\n"
            if max(final_precision) - min(final_precision) > 0.3:
                summary_text += "• Implementare class balancing o weighted loss\n"
            if accuracy_cv > 0.05:
                summary_text += "• Migliorare stabilità con learning rate scheduling\n"
            if final_accuracy > 95:
                summary_text += "• Performance eccellente! Sistema pronto per produzione\n"
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.9))
            
            plt.title('Performance Summary Report', fontsize=18, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"📑 PDF completo salvato: {pdf_filename}")
        return pdf_filename
    
    def generate_summary_stats(self):
        """Genera statistiche riassuntive in console."""
        print("\n" + "="*70)
        print("📊 FEDERATED LEARNING - ANALISI RISULTATI")
        print("="*70)
        
        print(f"\n📁 File: {self.csv_file}")
        print(f"📈 Epoche: {len(self.df)}")
        
        print(f"\n🎯 METRICHE FINALI:")
        print(f"   • Accuracy: {self.df['accuracy'].iloc[-1]:.2f}%")
        print(f"   • Loss: {self.df['loss'].iloc[-1]:.4f}")
        print(f"   • Avg Precision: {self.df['avg_precision'].iloc[-1]:.3f}")
        print(f"   • Avg Recall: {self.df['avg_recall'].iloc[-1]:.3f}")
        print(f"   • Avg F1-Score: {self.df['avg_f1_score'].iloc[-1]:.3f}")
        
        print(f"\n📈 EVOLUZIONE:")
        accuracy_change = self.df['accuracy'].iloc[-1] - self.df['accuracy'].iloc[0]
        loss_change = self.df['loss'].iloc[0] - self.df['loss'].iloc[-1]
        print(f"   • Cambio Accuracy: {accuracy_change:+.2f}%")
        print(f"   • Riduzione Loss: {loss_change:+.4f}")
        
        print(f"\n🏆 TOP/WORST CLASSES:")
        final_precision = [self.df[f'precision_class_{i}'].iloc[-1] for i in range(self.num_classes)]
        final_recall = [self.df[f'recall_class_{i}'].iloc[-1] for i in range(self.num_classes)]
        
        best_prec_idx = np.argmax(final_precision)
        worst_prec_idx = np.argmin(final_precision)
        print(f"   • Best Precision: Classe {best_prec_idx} ({final_precision[best_prec_idx]:.3f})")
        print(f"   • Worst Precision: Classe {worst_prec_idx} ({final_precision[worst_prec_idx]:.3f})")
        
        print("\n" + "="*70)
    
    def create_comprehensive_report(self):
        """Crea un report completo."""
        print("\n🚀 Avvio analisi Federated Learning Results...")
        
        if not self.load_data():
            return False
        
        # Genera PDF completo
        pdf_filename = self.create_comprehensive_pdf_report()
        
        # Mostra statistiche in console
        self.generate_summary_stats()
        
        print(f"\n✅ Analisi completa generata!")
        print(f"📁 File principale: {pdf_filename}")
        print(f"📂 Directory: {self.output_dir}/")
        
        return True


def main():
    """Funzione principale."""
    print("🔬 Federated Learning Results Analyzer")
    print("=" * 50)
    
    # Chiedi il file CSV
    csv_file = input("📁 Nome del file CSV (default: 3000_results.csv): ").strip()
    if not csv_file:
        csv_file = "3000_results.csv"
    
    # Crea l'analyzer
    analyzer = FLResultsAnalyzer(csv_file)
    
    # Genera il report completo
    success = analyzer.create_comprehensive_report()
    
    if success:
        print("\n🎉 Analisi completata con successo!")
    else:
        print("\n❌ Errore nell'analisi!")
    
    return success


if __name__ == "__main__":
    main()
