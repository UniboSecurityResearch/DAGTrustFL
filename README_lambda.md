# Automated Test Runner per DAGTrustFL - Lambda Testing

Questo script automatizza l'esecuzione dei test per DAGTrustFL con diversi valori di lambda (peso per l'influenza della valutazione di trust storica).

## File creati:

1. **`automated_test_lambda.py`** - Script principale che esegue tutti i test con diversi valori di lambda
2. **`single_test_lambda.py`** - Script per testare un singolo valore di lambda
3. **`README_lambda.md`** - Questo file

## Come funziona lo script principale

Lo script esegue automaticamente i seguenti passi per ogni valore di lambda:

1. **Modifica parametro**: Aggiorna il valore di `lambda_` in `DAG_functions.py` (riga 25)
2. **Esegue test**: Lancia in sequenza:
   - `python3 label_flipping_attack.py`
   - `python3 analyze_dag_csv_pdf.py` 
   - `python3 analyze_fl_csv_pdf.py`
3. **Organizza risultati**: Sposta tutti i file generati in una cartella con nome descrittivo

## Valori di lambda testati

I valori di lambda testati sono: **[0.0, 0.3, 0.5, 0.7]**

### Significato di lambda:
- **lambda = 0.0**: Nessuna influenza del trust storico (solo trust corrente)
- **lambda = 0.3**: Bassa influenza del trust storico (30% storico, 70% corrente)  
- **lambda = 0.5**: Influenza bilanciata (50% storico, 50% corrente)
- **lambda = 0.7**: Alta influenza del trust storico (70% storico, 30% corrente)

La formula usata è: `trust_value_t = (1-lambda_) * trust_value + lambda_ * trust_vector[i]`

## Uso

### Esecuzione completa di tutti i test lambda:
```bash
cd /home/giac/DAGTrustFL/DAGTrustFL
python3 automated_test_lambda.py
```

### Test di un singolo valore lambda (per debug):
```bash
cd /home/giac/DAGTrustFL/DAGTrustFL
python3 single_test_lambda.py <lambda_value>

# Esempi:
python3 single_test_lambda.py 0.0
python3 single_test_lambda.py 0.5
python3 single_test_lambda.py 0.7
```

## Struttura risultati

I risultati vengono organizzati in `FL/results_lambda/` con cartelle nominate come:
- `lambda_0_0/` (per lambda = 0.0)
- `lambda_0_3/` (per lambda = 0.3)
- `lambda_0_5/` (per lambda = 0.5)
- `lambda_0_7/` (per lambda = 0.7)

Ogni cartella contiene:
- `3000_workers_selected.csv`
- `3000_results.csv`
- `dag_trust_analysis.csv`
- `dag_analysis_plots/` (cartella con tutti i grafici e PDF)

## File modificati automaticamente

1. **`DAG/DAG_functions.py`**:
   - Parametro `lambda_` alla riga 25

## Altri parametri (fissi durante i test)

Durante i test lambda, questi parametri rimangono costanti:
- `chi`: valore di default nel TrustInformedTipSelector_NoThreshold
- `malicious_percentage`: valore attuale in DAG_functions.py
- `corrupted_percentage`: valore attuale in DAG_functions.py
- `upsilon_1` e `upsilon_2`: pesi per structural consistency e updates contributions

## Note importanti

- Lo script modifica solo il parametro `lambda_`, tutti gli altri rimangono fissi
- Ogni test può richiedere diversi minuti
- I risultati precedenti vengono spostati, non sovrascritti
- Lo script continua anche se un singolo test fallisce
- Puoi interrompere con Ctrl+C

## Controllo progresso

Lo script mostra:
- Progresso attuale (es. "Progress: 2/4")
- Valore di lambda corrente
- Stato di completamento di ogni fase
- Errori eventuali (ma continua l'esecuzione)

## Debugging

Se un test fallisce:
1. Controlla i messaggi di errore mostrati
2. Usa `single_test_lambda.py` per testare quel specifico valore di lambda
3. Verifica che tutti i file necessari esistano
4. Controlla i permessi di scrittura nella cartella FL/

## Confronto con automated_test_chi.py

- **`automated_test_chi.py`**: Testa diversi valori di chi e combinazioni di percentuali malicious/corrupted
- **`automated_test_lambda.py`**: Testa solo diversi valori di lambda mantenendo tutto il resto fisso

Entrambi gli script possono essere usati indipendentemente per diversi tipi di analisi.
