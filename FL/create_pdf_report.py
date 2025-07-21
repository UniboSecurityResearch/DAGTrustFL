#!/usr/bin/env python3
"""
Script standalone per creare un report PDF professionale dai dati di cosine similarity.
Utilizza il file CSV generato dall'analisi per creare grafici professionali.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os

def create_pdf_from_csv(csv_file='cosine_similarity_report.csv', output_file='cosine_similarity_temporal_analysis.pdf'):
    """
    Crea un PDF professionale dai dati CSV del temporal evolution summary.
    
    Args:
        csv_file (str): Path al file CSV con i dati
        output_file (str): Nome del file PDF di output
    
    Returns:
        str: Path del file PDF creato, None se errore
    """
    try:
        # Carica i dati
        if not os.path.exists(csv_file):
            print(f"❌ File CSV non trovato: {csv_file}")
            return None
        
        df_temporal = pd.read_csv(csv_file)
        print(f"✅ Caricati dati da {csv_file}: {len(df_temporal)} epochs")
        
        # Verifica che abbia le colonne necessarie
        required_cols = ['Epoch', 'Clean_Avg', 'Clean_Std', 'Poisoned_Avg', 'Poisoned_Std', 
                        'Discrimination', 'Separability_Ratio', 'Optimal_Threshold', 'Threshold_Accuracy']
        
        missing_cols = [col for col in required_cols if col not in df_temporal.columns]
        if missing_cols:
            print(f"❌ Colonne mancanti nel CSV: {missing_cols}")
            return None
        
        # Setup stile professionale
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Crea il file PDF
        with PdfPages(output_file) as pdf:
            
            # ===================== PAGINA 1: Overview Metrics =====================
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('Cosine Similarity Temporal Analysis - Overview', fontsize=20, fontweight='bold', y=0.95)
            
            # Plot 1: Discrimination over time
            ax1 = plt.subplot(2, 2, 1)
            epochs = df_temporal['Epoch'].values
            discriminations = df_temporal['Discrimination'].values
            
            plt.plot(epochs, discriminations, 'o-', linewidth=3, markersize=8, 
                    color='#2E86AB', label='Discrimination Power')
            
            # Trend line
            if len(epochs) > 1:
                z = np.polyfit(epochs, discriminations, 1)
                p = np.poly1d(z)
                plt.plot(epochs, p(epochs), "--", alpha=0.8, linewidth=2, 
                        color='#A23B72', label=f'Trend: {z[0]:+.6f}/epoch')
            
            plt.title('Discrimination Power Evolution', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Discrimination (|Clean - Poisoned|)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Threshold Accuracy over time  
            ax2 = plt.subplot(2, 2, 2)
            accuracies = df_temporal['Threshold_Accuracy'].values
            
            plt.plot(epochs, accuracies * 100, 's-', linewidth=3, markersize=8,
                    color='#F18F01', label='Detection Accuracy')
            
            # Aggiungi bande di qualità
            plt.axhspan(80, 100, alpha=0.2, color='green', label='Excellent (≥80%)')
            plt.axhspan(70, 80, alpha=0.2, color='yellow', label='Good (70-80%)')
            plt.axhspan(0, 70, alpha=0.2, color='red', label='Poor (<70%)')
            
            plt.title('Detection Accuracy Evolution', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 105)
            
            # Plot 3: Similarity values comparison
            ax3 = plt.subplot(2, 2, 3)
            clean_avgs = df_temporal['Clean_Avg'].values
            poisoned_avgs = df_temporal['Poisoned_Avg'].values
            clean_stds = df_temporal['Clean_Std'].values
            poisoned_stds = df_temporal['Poisoned_Std'].values
            
            # Plot con error bars
            plt.errorbar(epochs, clean_avgs, yerr=clean_stds, 
                       fmt='o-', linewidth=3, markersize=6, capsize=5,
                       color='#4CAF50', label='Clean-Clean Similarity')
            plt.errorbar(epochs, poisoned_avgs, yerr=poisoned_stds,
                       fmt='s-', linewidth=3, markersize=6, capsize=5, 
                       color='#F44336', label='Clean-Poisoned Similarity')
            
            plt.title('Average Similarities with Standard Deviation', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Cosine Similarity', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Threshold evolution
            ax4 = plt.subplot(2, 2, 4)
            thresholds = df_temporal['Optimal_Threshold'].values
            
            plt.plot(epochs, thresholds, '^-', linewidth=3, markersize=8,
                    color='#9C27B0', label='Optimal Threshold')
            
            # Aggiungi statistiche
            mean_threshold = np.mean(thresholds)
            std_threshold = np.std(thresholds)
            
            plt.axhline(mean_threshold, color='orange', linestyle=':', linewidth=2,
                       label=f'Mean: {mean_threshold:.4f}')
            plt.fill_between(epochs, mean_threshold - std_threshold, mean_threshold + std_threshold,
                           alpha=0.2, color='orange', label=f'±1 Std: {std_threshold:.4f}')
            
            plt.title('Optimal Threshold Evolution', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Threshold Value', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # ===================== PAGINA 2: Advanced Analysis =====================
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('Advanced Metrics & Statistical Analysis', fontsize=20, fontweight='bold', y=0.95)
            
            # Plot 1: Separability Ratio
            ax1 = plt.subplot(2, 2, 1)
            sep_ratios = df_temporal['Separability_Ratio'].values
            
            plt.plot(epochs, sep_ratios, 'D-', linewidth=3, markersize=8,
                    color='#E91E63', label='Separability Ratio')
            
            # Aggiungi trend
            if len(epochs) > 1:
                z = np.polyfit(epochs, sep_ratios, 1)
                p = np.poly1d(z)
                plt.plot(epochs, p(epochs), "--", alpha=0.8, linewidth=2,
                        color='#795548', label=f'Trend: {z[0]:+.6f}/epoch')
            
            plt.title('Separability Ratio (Discrimination/Std)', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Separability Ratio', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Correlation Matrix
            ax2 = plt.subplot(2, 2, 2)
            metrics_for_corr = df_temporal[['Discrimination', 'Separability_Ratio', 
                                          'Threshold_Accuracy', 'Optimal_Threshold']].corr()
            
            im = ax2.imshow(metrics_for_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Aggiungi testo delle correlazioni
            for i in range(len(metrics_for_corr)):
                for j in range(len(metrics_for_corr)):
                    text = ax2.text(j, i, f'{metrics_for_corr.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            ax2.set_xticks(range(len(metrics_for_corr.columns)))
            ax2.set_yticks(range(len(metrics_for_corr.columns)))
            ax2.set_xticklabels(metrics_for_corr.columns, rotation=45, ha='right')
            ax2.set_yticklabels(metrics_for_corr.columns)
            ax2.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Correlation Coefficient', fontsize=12)
            
            # Plot 3: Trend Analysis
            ax3 = plt.subplot(2, 2, 3)
            
            # Calcola trend per ciascuna metrica
            metrics = ['Discrimination', 'Threshold_Accuracy', 'Separability_Ratio', 'Optimal_Threshold']
            trends = []
            
            for metric in metrics:
                if len(df_temporal) > 1:
                    slope = np.polyfit(df_temporal['Epoch'], df_temporal[metric], 1)[0]
                    trends.append(slope)
                else:
                    trends.append(0)
            
            # Crea bar plot dei trend
            colors = ['#2E86AB' if t >= 0 else '#A23B72' for t in trends]
            bars = ax3.bar(range(len(metrics)), trends, color=colors, alpha=0.7)
            
            # Aggiungi etichette
            ax3.set_xticks(range(len(metrics)))
            ax3.set_xticklabels(metrics, rotation=45, ha='right')
            ax3.set_ylabel('Trend Slope (per epoch)', fontsize=12)
            ax3.set_title('Trend Analysis Summary', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Aggiungi valori sulle barre
            max_abs_trend = max(abs(min(trends)), abs(max(trends))) if trends else 1
            for i, (bar, trend) in enumerate(zip(bars, trends)):
                height = bar.get_height()
                offset = 0.1 * max_abs_trend if height >= 0 else -0.1 * max_abs_trend
                ax3.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{trend:+.6f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontweight='bold', fontsize=10)
            
            # Plot 4: Performance Summary
            ax4 = plt.subplot(2, 2, 4)
            
            # Crea pie chart delle performance
            final_accuracy = df_temporal['Threshold_Accuracy'].iloc[-1]
            
            if final_accuracy >= 0.8:
                performance_label = 'Excellent (≥80%)'
                colors_pie = ['#4CAF50', '#E0E0E0']
            elif final_accuracy >= 0.7:
                performance_label = 'Good (70-80%)'
                colors_pie = ['#FFC107', '#E0E0E0']
            else:
                performance_label = 'Poor (<70%)'
                colors_pie = ['#F44336', '#E0E0E0']
            
            sizes = [final_accuracy * 100, (1 - final_accuracy) * 100]
            labels = [f'Correct\n{final_accuracy*100:.1f}%', f'Errors\n{(1-final_accuracy)*100:.1f}%']
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            
            ax4.set_title(f'Final Detection Performance\n{performance_label}', 
                         fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # ===================== PAGINA 3: Summary Statistics =====================
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('Summary Statistics & Recommendations', fontsize=20, fontweight='bold', y=0.95)
            
            # Calcola statistiche riassuntive
            summary_stats = df_temporal.describe()
            
            # Crea tabella delle statistiche
            ax1 = plt.subplot(2, 1, 1)
            ax1.axis('tight')
            ax1.axis('off')
            
            # Prepara i dati della tabella
            stats_to_show = ['mean', 'std', 'min', 'max']
            cols_to_show = ['Discrimination', 'Threshold_Accuracy', 'Separability_Ratio', 'Optimal_Threshold']
            
            table_data = []
            for stat in stats_to_show:
                row = [stat.upper()]
                for col in cols_to_show:
                    if col == 'Threshold_Accuracy':
                        row.append(f"{summary_stats.loc[stat, col]*100:.2f}%")
                    else:
                        row.append(f"{summary_stats.loc[stat, col]:.6f}")
                table_data.append(row)
            
            headers = ['Statistic'] + cols_to_show
            table = ax1.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            bbox=[0, 0.3, 1, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            # Stile della tabella
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#E3F2FD')
                table[(0, i)].set_text_props(weight='bold')
            
            ax1.set_title('Summary Statistics Table', fontsize=16, fontweight='bold', y=0.8)
            
            # Aggiungi raccomandazioni testuali
            ax2 = plt.subplot(2, 1, 2)
            ax2.axis('off')
            
            # Calcola raccomandazioni
            avg_accuracy = df_temporal['Threshold_Accuracy'].mean()
            avg_discrimination = df_temporal['Discrimination'].mean()
            threshold_std = df_temporal['Optimal_Threshold'].std()
            final_threshold = df_temporal['Optimal_Threshold'].iloc[-1]
            
            # Determina effectiveness
            if avg_accuracy >= 0.8 and avg_discrimination >= 0.01:
                effectiveness = "🟢 EXCELLENT - Highly recommended for production"
            elif avg_accuracy >= 0.7 and avg_discrimination >= 0.005:
                effectiveness = "🟡 GOOD - Suitable with monitoring"
            elif avg_accuracy >= 0.6:
                effectiveness = "🟠 MODERATE - Consider improvements"
            else:
                effectiveness = "🔴 POOR - Not recommended"
            
            # Determina stability
            threshold_cv = threshold_std / df_temporal['Optimal_Threshold'].mean()
            if threshold_cv < 0.01:
                stability = "✅ EXCELLENT - Threshold is very stable"
            elif threshold_cv < 0.05:
                stability = "⚠️ GOOD - Moderate variability"
            else:
                stability = "❌ POOR - Threshold is unstable"
            
            recommendations_text = f"""
FINAL RECOMMENDATIONS & CONFIGURATION

Overall Effectiveness: {effectiveness}

📋 Recommended Configuration:
   • similarity_method = 'cosine_original'
   • similarity_threshold = {final_threshold:.6f}
   • expected_accuracy = {avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)
   • discrimination_power = {avg_discrimination:.6f}

🎯 Threshold Stability: {stability}
   • Mean Threshold: {df_temporal['Optimal_Threshold'].mean():.6f}
   • Standard Deviation: {threshold_std:.6f}
   • Coefficient of Variation: {threshold_cv:.6f}

🚀 Deployment Guidance:
   • {'High accuracy - can be primary detection method' if avg_accuracy >= 0.8 else 'Moderate accuracy - combine with other methods' if avg_accuracy >= 0.7 else 'Low accuracy - use only as supplementary method'}
   • {'Threshold is stable - safe for static config' if threshold_cv < 0.01 else 'Consider adaptive threshold based on recent epochs'}
   • Total epochs analyzed: {len(df_temporal)}
   • Success rate: {len(df_temporal)}/{len(df_temporal)} epochs (100.0%)
            """
            
            ax2.text(0.05, 0.95, recommendations_text, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Aggiungi metadati al PDF
            d = pdf.infodict()
            d['Title'] = 'Cosine Similarity Temporal Analysis'
            d['Author'] = 'DAGTrustFL Analysis System'
            d['Subject'] = 'Federated Learning Parameter Similarity Analysis'
            d['Keywords'] = 'Cosine Similarity, Federated Learning, Poisoning Detection'
            d['CreationDate'] = pd.Timestamp.now()
        
        print(f"\n📊 Professional PDF report created: '{output_file}'")
        print(f"   📄 3 pages with comprehensive temporal analysis")
        print(f"   📈 Overview metrics, advanced analysis, and summary statistics")
        print(f"   📋 Correlation analysis and performance assessment")
        
        return output_file
        
    except ImportError as e:
        print(f"❌ Missing required libraries: {e}")
        print("Install with: pip install matplotlib pandas numpy")
        return None
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function per eseguire lo script standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create PDF report from cosine similarity CSV data')
    parser.add_argument('--csv', default='cosine_similarity_report.csv', 
                       help='Input CSV file (default: cosine_similarity_report.csv)')
    parser.add_argument('--output', default='cosine_similarity_temporal_analysis.pdf',
                       help='Output PDF file (default: cosine_similarity_temporal_analysis.pdf)')
    
    args = parser.parse_args()
    
    print("🚀 Creating Professional PDF Report from CSV Data")
    print("="*60)
    
    result = create_pdf_from_csv(args.csv, args.output)
    
    if result:
        print(f"✅ PDF report successfully created: {result}")
        print(f"📄 Ready for presentations and detailed analysis!")
    else:
        print(f"❌ Failed to create PDF report")
        sys.exit(1)

if __name__ == "__main__":
    main()
