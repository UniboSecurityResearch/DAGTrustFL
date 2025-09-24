#!/usr/bin/env python3
"""
Script per generare un grafico che mostra l'evoluzione nel tempo di:
- Mean_Similarity_Good
- Mean_Similarity_Bad  
- Similarity_Threshold

Utilizza i dati dal file dag_trust_analysis.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.font_manager as fm

# Configura il font Roboto
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.size'] = 16

def plot_similarity_evolution(csv_file_path, output_dir=None):
    """
    Genera un grafico dell'evoluzione delle metriche di similarità nel tempo
    
    Args:
        csv_file_path (str): Percorso al file CSV con i dati
        output_dir (str): Directory dove salvare il grafico (opzionale)
    """
    
    # Leggi i dati dal CSV
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Dati caricati con successo. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Errore: File {csv_file_path} non trovato")
        return
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return
    
    # Verifica che le colonne necessarie esistano
    required_columns = ['Round', 'Mean_Similarity_Good', 'Mean_Similarity_Bad', 'Similarity_Threshold']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Errore: Colonne mancanti nel CSV: {missing_columns}")
        print(f"Colonne disponibili: {list(df.columns)}")
        return
    
    # Configura lo stile del grafico
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colori per le diverse metriche
    colors = {
        'Mean_Similarity_Good': '#2E8B57',    # Verde scuro
        'Mean_Similarity_Bad': '#DC143C',     # Rosso cremisi
        'Similarity_Threshold': '#4169E1'     # Blu reale
    }
    
    # Calcola i limiti del range basato su Mean_Similarity_Good - Similarity_Threshold (solo sotto)
    lower_bound = df['Mean_Similarity_Good'] - df['Similarity_Threshold']
    
    # Plot del range come area colorata (solo sotto Mean_Similarity_Good)
    ax.fill_between(df['Round'], lower_bound, df['Mean_Similarity_Good'],
                    color=colors['Similarity_Threshold'], 
                    alpha=0.5, 
                    label='Percentile-based Range')
    
    # Plot delle linee principali
    ax.plot(df['Round'], df['Mean_Similarity_Good'], 
            color=colors['Mean_Similarity_Good'], 
            linewidth=3, 
            label='Mean Similarity Good',
            marker='o', 
            markersize=5,
            alpha=0.8)
    
    ax.plot(df['Round'], df['Mean_Similarity_Bad'], 
            color=colors['Mean_Similarity_Bad'], 
            linewidth=3, 
            label='Mean Similarity Bad',
            marker='s', 
            markersize=5,
            alpha=0.8)
    
    # Personalizzazione del grafico
    ax.set_xlabel('Round', fontsize=20, fontweight='bold')
    ax.set_ylabel('Similarity Value', fontsize=20, fontweight='bold')
    
    # Griglia e leggenda
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=18, framealpha=0.9)
    
    # Imposta i limiti degli assi
    ax.set_xlim(df['Round'].min() - 1, df['Round'].max() + 1)
    
    # Aumenta la dimensione dei tick
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Regola il layout
    plt.tight_layout()
    
    # Salva il grafico
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = Path(csv_file_path).parent
    
    # Salva in diversi formati
    base_name = 'similarity_evolution'
    
    # PNG ad alta risoluzione
    png_path = output_path / f"{base_name}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {png_path}")
    
    # PDF vettoriale
    pdf_path = output_path / f"{base_name}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Grafico PDF salvato in: {pdf_path}")
    
    # Mostra il grafico
    plt.show()
    
    # Genera anche un grafico aggiuntivo con subplot separati per ogni metrica
    create_subplot_analysis(df, output_path)

def create_subplot_analysis(df, output_path):
    """
    Crea un grafico aggiuntivo con subplot separati per analisi dettagliata
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    metrics = ['Mean_Similarity_Good', 'Mean_Similarity_Bad', 'Similarity_Threshold']
    colors = ['#2E8B57', '#DC143C', '#4169E1']
    titles = ['Mean Similarity Good Evolution', 'Mean Similarity Bad Evolution', 'Similarity Threshold Evolution']
    
    for i, (metric, color, title) in enumerate(zip(metrics, colors, titles)):
        ax = axes[i]
        
        # Plot principale
        ax.plot(df['Round'], df[metric], color=color, linewidth=2, marker='o', markersize=4)
        
        # Media mobile su finestra di 5 round
        if len(df) >= 5:
            rolling_mean = df[metric].rolling(window=5, center=True).mean()
            ax.plot(df['Round'], rolling_mean, color='orange', linewidth=1.5, 
                   linestyle='--', alpha=0.7, label='Media Mobile (5 round)')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Aggiungi statistiche
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        
        ax.axhline(y=mean_val, color='red', linestyle=':', alpha=0.7, label=f'Media: {mean_val:.4f}')
        
        stats_text = f"Min: {min_val:.4f}\nMax: {max_val:.4f}\nStd: {std_val:.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        if len(df) >= 5 or mean_val > 0:
            ax.legend(fontsize=9)
    
    plt.suptitle('Analisi Dettagliata delle Metriche di Similarità', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Salva l'analisi dettagliata
    detailed_path = output_path / "similarity_detailed_analysis.png"
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    print(f"Analisi dettagliata salvata in: {detailed_path}")
    
    plt.show()

def print_summary_statistics(csv_file_path):
    """
    Stampa statistiche riassuntive dei dati
    """
    df = pd.read_csv(csv_file_path)
    
    print("\n" + "="*60)
    print("STATISTICHE RIASSUNTIVE")
    print("="*60)
    
    metrics = ['Mean_Similarity_Good', 'Mean_Similarity_Bad', 'Similarity_Threshold']
    
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"  Media: {df[metric].mean():.6f}")
        print(f"  Mediana: {df[metric].median():.6f}")
        print(f"  Deviazione Standard: {df[metric].std():.6f}")
        print(f"  Minimo: {df[metric].min():.6f}")
        print(f"  Massimo: {df[metric].max():.6f}")
        print(f"  Range: {df[metric].max() - df[metric].min():.6f}")
    
    print(f"\nNumero totale di round: {len(df)}")
    print(f"Range round: {df['Round'].min()} - {df['Round'].max()}")

if __name__ == "__main__":
    # Percorso al file CSV
    csv_file = "dag_trust_analysis.csv"
    
    # Verifica che il file esista
    if not Path(csv_file).exists():
        print(f"Errore: File {csv_file} non trovato nella directory corrente")
        print("Assicurati di eseguire lo script dalla directory contenente il file CSV")
        exit(1)
    
    print("Inizio generazione grafico evoluzione similarità...")
    
    # Genera il grafico principale
    plot_similarity_evolution(csv_file)
    
    # Stampa statistiche riassuntive
    print_summary_statistics(csv_file)
    
    print("\nGenerazione grafico completata!")
