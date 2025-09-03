# Automated Test Runner per DAGTrustFL

Questo script automatizza l'esecuzione dei test per DAGTrustFL con diversi valori di parametri.

## File creati:

1. **`automated_test_chi.py`** - Script principale che esegue tutti i test automaticamente
2. **`single_test.py`** - Script per testare una singola configurazione
3. **`README_automation.md`** - Questo file

## Come funziona lo script principale

Lo script esegue automaticamente i seguenti passi:

1. **Modifica parametri**: Aggiorna i valori di `chi`, `malicious_percentage` e `corrupted_percentage` nei file appropriati
2. **Esegue test**: Lancia in sequenza:
   - `python3 label_flipping_attack.py`
   - `python3 analyze_dag_csv_pdf.py` 
   - `python3 analyze_fl_csv_pdf.py`
3. **Organizza risultati**: Sposta tutti i file generati in una cartella con nome descrittivo

## Configurazioni eseguite

### Prima serie (malicious=10%, corrupted=20%):
- chi=0, malicious=10%, corrupted=20%
- chi=1, malicious=10%, corrupted=20%
- chi=2, malicious=10%, corrupted=20%
- chi=3, malicious=10%, corrupted=20%
- chi=4, malicious=10%, corrupted=20%

### Seconda serie (malicious=20%, corrupted=10%):
- chi=0, malicious=20%, corrupted=10%
- chi=1, malicious=20%, corrupted=10%
- chi=2, malicious=20%, corrupted=10%
- chi=3, malicious=20%, corrupted=10%
- chi=4, malicious=20%, corrupted=10%

## Uso

### Esecuzione completa di tutti i test:
```bash
cd /home/giac/DAGTrustFL/DAGTrustFL
python3 automated_test_chi.py
```

### Test di una singola configurazione (per debug):
```bash
cd /home/giac/DAGTrustFL/DAGTrustFL
python3 single_test.py <chi> <malicious_pct> <corrupted_pct>

# Esempio:
python3 single_test.py 0 0.1 0.2
```

## Struttura risultati

I risultati vengono organizzati in `FL/results/` con cartelle nominate come:
- `chi_0_mal_10_cor_20/` 
- `chi_1_mal_10_cor_20/`
- `chi_2_mal_10_cor_20/`
- etc.

Ogni cartella contiene:
- `3000_workers_selected.csv`
- `3000_results.csv`
- `dag_trust_analysis.csv`
- `dag_analysis_plots/` (cartella con tutti i grafici e PDF)

## File modificati automaticamente

1. **`DAG/DAG_functions.py`**:
   - Parametro `chi` nella inizializzazione di `TrustInformedTipSelector_NoThreshold`
   - `malicious_percentage`
   - `corrupted_percentage`

## Note importanti

- Lo script fa backup automatico modificando i file, ma assicurati di avere una copia di backup
- Ogni test può richiedere diversi minuti
- I risultati precedenti vengono spostati, non sovrascritti
- Lo script continua anche se un singolo test fallisce
- Puoi interrompere con Ctrl+C

## Controllo progresso

Lo script mostra:
- Progresso attuale (es. "Progress: 3/10")
- Parametri di ogni test
- Stato di completamento di ogni fase
- Errori eventuali (ma continua l'esecuzione)

## Debugging

Se un test fallisce:
1. Controlla i messaggi di errore mostrati
2. Usa `single_test.py` per testare quella specifica configurazione
3. Verifica che tutti i file necessari esistano
4. Controlla i permessi di scrittura nella cartella FL/
