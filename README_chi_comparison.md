# Chi Results Comparison Script per DAGTrustFL

Questo script analizza e confronta i risultati ottenuti dai test automatici con diversi valori del parametro `chi` e diverse percentuali di nodi malicious/corrupted.

## File creato:

**`compare_chi_results.py`** - Script per analizzare e confrontare i risultati dei test chi

## Cosa fa lo script

Analizza i CSV `dag_trust_analysis.csv` dalle cartelle dei risultati e crea grafici di confronto per:

### 1. False Positive Rate (FPR)
- **Formula**: `(Nodi buoni rimossi / Nodi buoni totali inviati) × 100`
- **Significato**: Percentuale di nodi buoni incorrettamente classificati come malicious
- **Valore ottimale**: 0% (nessun falso positivo)

### 2. True Positive Rate (TPR) 
- **Formula**: `(Nodi malicious rilevati / Nodi malicious totali inviati) × 100`
- **Significato**: Percentuale di nodi malicious correttamente identificati e rimossi
- **Valore ottimale**: 100% (tutti i malicious rilevati)

### 3. False Negative Rate (FNR)
- **Formula**: `(Nodi malicious confermati / Nodi malicious totali inviati) × 100`
- **Significato**: Percentuale di nodi malicious incorrettamente accettati nel grafo
- **Valore ottimale**: 0% (nessun falso negativo)

### 4. Overall Accuracy
- **Formula**: `(Nodi gestiti correttamente / Nodi totali inviati) × 100`
- **Significato**: Accuratezza complessiva del sistema di classificazione
- **Valore ottimale**: 100% (classificazione perfetta)

## Metodologia di calcolo

### Stima dei nodi inviati
Poiché il CSV non contiene direttamente il numero di nodi inviati per round, lo script usa questa metodologia:

1. **Conta i peer**: Dalla struttura `Peer_X_Is_Malicious` nel CSV
2. **Considera selezione casuale**: Ogni round vengono scelti K-1 nodi (9 su 10 peer) casualmente
3. **Calcola probabilità di selezione**: 
   - Per peer buoni: `(Peer buoni / Peer totali)`
   - Per peer malicious: `(Peer malicious / Peer totali)`
   - Per peer corrupted: `(Peer corrupted / Peer totali)`
4. **Stima nodi inviati**:
   - Nodi per round = K-1 = 9
   - Nodi buoni inviati = `(9 × Prob. buoni × Rounds totali)`
   - Nodi malicious inviati = `(9 × Prob. malicious × Rounds totali)`
   - Nodi corrupted inviati = `(9 × Prob. corrupted × Rounds totali)`

### Formula precisa
```
Expected_tipo_per_round = (9 × Peer_tipo) / 10
Total_tipo_inviati = Expected_tipo_per_round × Rounds_totali
```

### Metriche finali
- **Somma tutti i round**: Aggrega `Good_Nodes_Removed_This_Round` e `Malicious_Nodes_Confirmed_This_Round` di tutti i round
- **Non usa dati cumulativi**: I campi `Total_*` nel CSV sono cumulativi per round, ma noi sommiamo i valori `*_This_Round` per maggiore precisione

## Struttura cartelle richiesta

Lo script cerca cartelle con il pattern: `chi_X_mal_Y_cor_Z`

Esempio:
```
FL/results/
├── chi_0_mal_10_cor_20/
│   └── dag_trust_analysis.csv
├── chi_1_mal_10_cor_20/
│   └── dag_trust_analysis.csv
├── chi_2_mal_10_cor_20/
│   └── dag_trust_analysis.csv
├── chi_0_mal_20_cor_10/
│   └── dag_trust_analysis.csv
└── ...
```

## Uso

### Eseguire l'analisi completa:
```bash
cd /home/giac/DAGTrustFL/DAGTrustFL
python3 compare_chi_results.py
```

## Output generato

### 1. PDF Report
- **File**: `chi_comparison_plots/chi_comparison_report.pdf`
- **Contenuto**: 
  - Grafici per ogni configurazione (mal%, cor%)
  - Confronto generale tra tutte le configurazioni
  - Ranking dei valori chi per efficacia
  - Statistiche riassuntive

### 2. Grafici per configurazione
Per ogni combinazione malicious%/corrupted%, genera:
- **FPR vs Chi**: Evoluzione dei falsi positivi
- **TPR vs Chi**: Evoluzione della detection rate (true positives)
- **Overall Accuracy vs Chi**: Accuratezza complessiva del sistema
- **Vista combinata**: Tutti i metrici insieme con FNR incluso

### 3. Confronto generale
- Confronta tutti i metrici tra diverse configurazioni
- Ranking dei valori chi per performance complessiva
- Score composito basato su FPR, FNR e Accuracy

## Interpretazione risultati

### Chi ottimale
Il valore chi ottimale dovrebbe:
- **Minimizzare FPR**: Pochi nodi buoni rimossi
- **Massimizzare TPR**: Molti nodi malicious rilevati
- **Minimizzare FNR**: Pochi nodi malicious confermati
- **Massimizzare Accuracy**: Alta accuratezza complessiva

### Relazione tra TPR e FNR
- **TPR + FNR ≠ 100%** perché misurano aspetti diversi:
  - TPR: Malicious correttamente **rilevati e rimossi**
  - FNR: Malicious incorrettamente **confermati nel grafo**
- Un sistema perfetto ha **TPR = 100%** e **FNR = 0%**

### Pattern tipici
- **Chi basso (0-1)**: Ispezione superficiale, può perdere pattern malicious
- **Chi medio (2-3)**: Bilanciamento tra detection e performance
- **Chi alto (4+)**: Ispezione profonda, migliore detection ma più computazionale

## Prerequisiti

### Dipendenze Python:
```bash
pip install pandas matplotlib seaborn numpy
```

### Dati richiesti:
- Risultati generati da `automated_test_chi.py`
- File `dag_trust_analysis.csv` in ogni cartella risultato

## Note importanti

- Lo script legge l'ultimo round (riga finale) di ogni CSV per le metriche finali
- Calcola automaticamente il numero di peer malicious/corrupted/buoni dai dati
- Gestisce valori negativi convertendoli a 0 per evitare metriche inconsistenti
- Supporta multiple configurazioni contemporaneamente

## Debug

Se lo script non trova esperimenti:
1. Verifica che le cartelle seguano il pattern `chi_X_mal_Y_cor_Z`
2. Controlla che ogni cartella contenga `dag_trust_analysis.csv`
3. Assicurati di aver eseguito `automated_test_chi.py` prima

## Esempi di interpretazione

### Risultato ideale:
- FPR = 0% (nessun buono rimosso)
- FNR = 0% (nessun malicious confermato)  
- Accuracy = 100% (tutti i malicious rigettati)

### Risultato problematico:
- FPR alto = Sistema troppo aggressivo
- FNR alto = Sistema troppo permissivo
- Accuracy bassa = Scarsa capacità di detection
