#!/usr/bin/env python3
"""
Script per generare un grafico che confronta i tempi di pubblicazione delle transazioni
per dispositivo tra DAG, Bitcoin ed Ethereum al variare del numero di dispositivi.

Il grafico mostra:
- DAG: tempo costante (linea orizzontale)
- Bitcoin: tempo costante fino a N=3, poi incremento con inclinazione alpha_bitcoin
- Ethereum: tempo costante fino a N=15-30, poi incremento con inclinazione alpha_ethereum
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_transaction_time_comparison():
    """
    Crea il grafico di confronto dei tempi di transazione.
    """
    
    # Parametri del grafico
    N_max = 100  # Numero massimo di dispositivi
    N = np.arange(1, N_max + 1)
    
    # Parametri per DAG (tempo costante)
    t_dag_base = 5  # Tempo base costante per DAG (millisecondi)
    
    # Parametri per Bitcoin
    t_bitcoin_base = 5  # Tempo base per Bitcoin (millisecondi) - stesso del DAG
    N_bitcoin_breakpoint_min = 7  # Punto di rottura minimo per Bitcoin
    N_bitcoin_breakpoint_max = 7  # Punto di rottura massimo per Bitcoin
    N_bitcoin_breakpoint = (N_bitcoin_breakpoint_min + N_bitcoin_breakpoint_max) / 2  # Punto medio
    bitcoin_increment = 5  # Incremento di 5ms per ogni dispositivo aggiuntivo oltre il breaking point
    
    # Parametri per Ethereum
    t_ethereum_base = 5  # Tempo base per Ethereum (millisecondi) - stesso del DAG
    N_ethereum_breakpoint_min = 30  # Punto di rottura minimo per Ethereum
    N_ethereum_breakpoint_max = 30  # Punto di rottura massimo per Ethereum
    N_ethereum_breakpoint = (N_ethereum_breakpoint_min + N_ethereum_breakpoint_max) / 2  # Punto medio
    ethereum_increment = 5  # Incremento di 5ms per ogni dispositivo aggiuntivo oltre il breaking point
    
    # Calcolo dei tempi per DAG (costante)
    t_dag = np.full_like(N, t_dag_base, dtype=float)
    
    # Calcolo dei tempi per Bitcoin
    t_bitcoin = np.zeros_like(N, dtype=float)
    for i, n in enumerate(N):
        if n <= N_bitcoin_breakpoint:
            t_bitcoin[i] = t_bitcoin_base
        else:
            # Incremento lineare di 5ms per ogni dispositivo aggiuntivo
            devices_over_breakpoint = n - N_bitcoin_breakpoint
            t_bitcoin[i] = t_bitcoin_base + devices_over_breakpoint * bitcoin_increment
    
    # Calcolo dei tempi per Ethereum
    t_ethereum = np.zeros_like(N, dtype=float)
    for i, n in enumerate(N):
        if n <= N_ethereum_breakpoint:
            t_ethereum[i] = t_ethereum_base
        else:
            # Incremento lineare di 5ms per ogni dispositivo aggiuntivo
            devices_over_breakpoint = n - N_ethereum_breakpoint
            t_ethereum[i] = t_ethereum_base + devices_over_breakpoint * ethereum_increment
    
    # Creazione del grafico
    plt.figure(figsize=(12, 8))
    
    # Plot delle linee
    plt.plot(N, t_dag, 'g-', linewidth=3, label='DAGTrustFL', marker='o', markersize=4, markevery=5)
    plt.plot(N, t_bitcoin, 'r-', linewidth=3, label='Bitcoin', marker='s', markersize=4, markevery=5)
    plt.plot(N, t_ethereum, 'b-', linewidth=3, label='Ethereum', marker='^', markersize=4, markevery=5)
    
    # Evidenziare i punti di rottura
    plt.axvline(x=N_bitcoin_breakpoint_max, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Bitcoin $N_b$ (N={N_bitcoin_breakpoint_max})')
    
    # Annotazione per Bitcoin breakpoint
    plt.text(N_bitcoin_breakpoint_max + 3.5, 140, 
             '$N_b$ bitcoin', ha='center', va='center', fontsize=18, fontweight='bold', 
             color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))
    
    plt.axvline(x=N_ethereum_breakpoint_max, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Ethereum $N_b$ (N={N_ethereum_breakpoint_max})')
    
    # Annotazione per Ethereum breakpoint
    plt.text(N_ethereum_breakpoint_max + 2, 70, 
             '$N_b$ ethereum', ha='center', va='center', fontsize=18, fontweight='bold', 
             color='blue', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.7))

    # Annotazioni per gli incrementi
    bitcoin_idx = int(N_bitcoin_breakpoint) + 10
    ethereum_idx = int(N_ethereum_breakpoint) + 10
    
    # Personalizzazione del grafico
    plt.xlabel('# per-round peers participating (K)', fontsize=26, fontweight='bold')
    plt.ylabel('Transactions Per Second (TPS) [ms]', fontsize=26, fontweight='bold')
    
    # Aumenta il font size dei tick
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    
    # Griglia e stile
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right', fontsize=20, framealpha=0.9)
    
    # Impostazioni degli assi
    plt.xlim(0, N_max)
    plt.ylim(0, 150)  # Limite massimo Y a 150 millisecondi
    
    # Miglioramenti estetici
    plt.tight_layout()
    
    return plt.gcf()

def save_plots():
    """
    Salva i grafici in diversi formati.
    """
    # Crea la directory per i plot se non esiste
    plot_dir = "../results_and_plots/transaction_time_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Genera il grafico
    fig = plot_transaction_time_comparison()
    
    # Salva in diversi formati
    base_filename = "transaction_time_comparison"
    
    # PNG ad alta risoluzione
    png_file = os.path.join(plot_dir, f"{base_filename}.png")
    fig.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Grafico salvato in PNG: {png_file}")
    
    # PDF per pubblicazioni
    pdf_file = os.path.join(plot_dir, f"{base_filename}.pdf")
    fig.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"✅ Grafico salvato in PDF: {pdf_file}")
    
    # SVG per editing vettoriale
    svg_file = os.path.join(plot_dir, f"{base_filename}.svg")
    fig.savefig(svg_file, bbox_inches='tight', facecolor='white')
    print(f"✅ Grafico salvato in SVG: {svg_file}")
    
    return png_file, pdf_file, svg_file

def create_comparison_table():
    """
    Crea una tabella di confronto dei tempi per valori specifici di N.
    """
    import pandas as pd
    
    # Valori di N da analizzare
    N_values = [1, 3, 5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
    
    # Parametri (stessi del grafico)
    t_dag_base = 5  # millisecondi
    t_bitcoin_base = 5  # millisecondi
    N_bitcoin_breakpoint = 4  # Punto medio tra 3 e 5
    bitcoin_increment = 5  # millisecondi per dispositivo aggiuntivo
    t_ethereum_base = 5  # millisecondi
    N_ethereum_breakpoint = 22.5  # Punto medio
    ethereum_increment = 5  # millisecondi per dispositivo aggiuntivo
    
    data = []
    for N in N_values:
        # DAG (costante)
        t_dag = t_dag_base
        
        # Bitcoin
        if N <= N_bitcoin_breakpoint:
            t_bitcoin = t_bitcoin_base
        else:
            devices_over_breakpoint = N - N_bitcoin_breakpoint
            t_bitcoin = t_bitcoin_base + devices_over_breakpoint * bitcoin_increment
        
        # Ethereum
        if N <= N_ethereum_breakpoint:
            t_ethereum = t_ethereum_base
        else:
            devices_over_breakpoint = N - N_ethereum_breakpoint
            t_ethereum = t_ethereum_base + devices_over_breakpoint * ethereum_increment
        
        data.append({
            'N_dispositivi': N,
            'DAG_tempo_ms': round(t_dag, 3),
            'Bitcoin_tempo_ms': round(t_bitcoin, 3),
            'Ethereum_tempo_ms': round(t_ethereum, 3),
            'DAG_vs_Bitcoin_ratio': round(t_dag / t_bitcoin, 3),
            'DAG_vs_Ethereum_ratio': round(t_dag / t_ethereum, 3)
        })
    
    df = pd.DataFrame(data)
    
    # Salva la tabella
    table_dir = "transaction_time_plots"
    os.makedirs(table_dir, exist_ok=True)
    
    csv_file = os.path.join(table_dir, "transaction_time_comparison_table.csv")
    df.to_csv(csv_file, index=False)
    print(f"✅ Tabella di confronto salvata: {csv_file}")
    
    # Mostra la tabella
    print("\n📊 TABELLA DI CONFRONTO TEMPI DI TRANSAZIONE:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return csv_file, df

def main():
    """
    Funzione principale per eseguire l'analisi completa.
    """
    print("🚀 Generazione grafico confronto tempi di transazione DAG vs Blockchain")
    print("=" * 70)
    
    try:
        # Genera e salva i grafici
        png_file, pdf_file, svg_file = save_plots()
        
        # Crea la tabella di confronto
        csv_file, df = create_comparison_table()
        
        print("\n✅ ANALISI COMPLETATA CON SUCCESSO!")
        print(f"📁 File generati nella cartella: transaction_time_plots/")
        print(f"   • Grafico PNG: {os.path.basename(png_file)}")
        print(f"   • Grafico PDF: {os.path.basename(pdf_file)}")
        print(f"   • Grafico SVG: {os.path.basename(svg_file)}")
        print(f"   • Tabella CSV: {os.path.basename(csv_file)}")
        
        # Mostra il grafico
        plt.show()
        
    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
        raise

if __name__ == "__main__":
    # Verifica dipendenze
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"❌ Errore: dipendenza mancante - {e}")
        print("💡 Installa con: pip install matplotlib numpy pandas")
        exit(1)
    
    main()