import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm

# Imposta un font sans-serif per tutto il grafico
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']


def load_timing_data(results_dir):
    """Load timing data from all scalability experiments"""
    results_path = Path(results_dir)
    timing_data = {}
    
    # Find all K_* folders
    for folder in results_path.glob("K_*_scalability"):
        # Extract K value from folder name
        k_str = folder.name.split('_')[1]
        try:
            k_value = int(k_str)
        except ValueError:
            continue
        
        # Look for timing results
        timing_folder = folder / 'timing_results'
        if timing_folder.exists():
            # Find timing summary CSV
            summary_files = list(timing_folder.glob("*_timing_summary.csv"))
            if summary_files:
                try:
                    # Read the metric,value format CSV
                    df = pd.read_csv(summary_files[0])
                    
                    # Convert from metric,value format to a proper DataFrame
                    timing_dict = {}
                    for _, row in df.iterrows():
                        timing_dict[row['metric']] = row['value']
                    
                    # Create a single-row DataFrame with the timing data
                    timing_df = pd.DataFrame([timing_dict])
                    timing_data[k_value] = timing_df
                    print(f"✓ Loaded timing data for K={k_value}")
                except Exception as e:
                    print(f"❌ Error loading timing data for K={k_value}: {e}")
    
    return timing_data
def main():
    results_dir = '../FL/results_scalability'
    
    print("🔍 Loading scalability experiment data...")
    
    # Load timing data
    timing_data = load_timing_data(results_dir)
    
    # Estrai solo avg_total_round_time e crea dizionario con tuple (K, tempo)
    dag_timing_points = {}
    for k_value, timing_df in timing_data.items():
        if 'avg_total_round_time' in timing_df.columns:
            avg_time = timing_df['avg_total_round_time'].iloc[0]
            dag_timing_points[k_value] = (k_value, avg_time)
            print(f"K={k_value}: avg_total_round_time = {avg_time}")
    
    print(f"\nDizionario timing points: {dag_timing_points}")
    
    # Range di N: calcolo da 1 a 200 (log(0) non definito), ma mostro l'asse X da 0
    N = np.arange(1, 201)

    # Curve di complessità
    logN    = np.log2(N)
    N_linear = N
    N_logN   = N * np.log2(N)
    N_sq     = N**2

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(N, logN,    label="O(log N)", linewidth=2, color='blue')
    plt.plot(N, N_linear, label="O(N)", linewidth=2, color='green')
    plt.plot(N, N_logN,   label="O(N log N)", linewidth=2, color='orange')
    plt.plot(N, N_sq,     label="O(N^2)", linewidth=2, color='red')
    
    # Plotta i timing data reali se disponibili
    if dag_timing_points:
        # Estrai coordinate X (K) e Y (tempo) dalle tuple e ordina per X crescente
        points = sorted(dag_timing_points.values(), key=lambda point: point[0])
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        print(f"X coords (ordinati): {x_coords}")
        print(f"Y coords (ordinati): {y_coords}")
        plt.plot(x_coords, y_coords, 'ko-', label="DAGTrustFL", linewidth=3, markersize=8)

    # Assi in inglese con font aumentati
    plt.xlabel("N (participants)", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xlim(0, 200)
    plt.ylim(1, 10**5)  # Y da 1 a 10^5
    plt.yscale('log')  # Asse Y logaritmico
    
    # Aumenta dimensione dei tick
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Griglia e legenda con font aumentato
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=17, loc='upper left')

    plt.show()

if __name__ == '__main__':
    main()
