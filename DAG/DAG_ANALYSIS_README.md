# DAG Trust Analysis Tools 📊

Strumenti per analizzare e visualizzare i dati CSV generati dal sistema DAG Trust per Federated Learning.

## 🚀 Quick Start

### 1. Installazione Dipendenze
```bash
pip install -r dag_analysis_requirements.txt
```

### 2. Esecuzione Analisi Completa
```bash
python analyze_dag_csv.py
```

### 3. Analisi Rapida
```bash
python quick_dag_analysis.py [file.csv]
```

## 📁 File Creati

### `analyze_dag_csv.py`
Script completo per analisi dettagliata che genera:
- **system_evolution.png**: Evoluzione generale del sistema
- **trust_evolution.png**: Evoluzione del trust vector + heatmap
- **consistency_contributions.png**: Analisi structural consistency e update contributions
- **detection_analysis.png**: Analisi efficacia detection maliciosi
- **summary_stats.txt**: Statistiche riassuntive in formato testo

### `quick_dag_analysis.py`
Script veloce per analisi rapida che genera:
- **quick_dag_analysis.png**: Panoramica rapida su 4 grafici

## 📊 Grafici Generati

### 1. System Evolution
- 🔄 **Evoluzione del DAG**: Numero totale di nodi e tips nel tempo
- 📊 **Nodi per Round**: Nodi buoni rimossi vs maliciosi/corrotti confermati per round
- 📈 **Evoluzione Cumulativa**: Totali cumulativi nel tempo
- 🎯 **Efficacia Detection**: Tasso di detection dei maliciosi

### 2. Trust Evolution
- 🤝 **Trust Vector**: Evoluzione del trust per ogni peer (linea continua=buoni, tratteggiata=maliciosi)
- 🔥 **Heatmap Trust**: Visualizzazione a colori dell'evoluzione del trust

### 3. Consistency & Contributions
- 🏗️ **Structural Consistency**: Come evolve la consistenza strutturale
- 📤 **Updates Contribution**: Come evolve il contributo degli aggiornamenti
- 📊 **Distribuzioni Finali**: Valori finali per ogni metrica

### 4. Detection Analysis
- 🎯 **Tasso Detection**: Percentuale di maliciosi/corrotti rilevati nel tempo
- ⚠️ **Falsi Positivi**: Percentuale di nodi buoni erroneamente rimossi
- 📦 **Box Plot Trust**: Confronto distribuzione trust finale buoni vs maliciosi
- ⏰ **Timeline Eventi**: Sequenza temporale degli eventi importanti

## 🎨 Legenda Grafici

| Simbolo | Significato |
|---------|-------------|
| 🟢 | Peer buono/onesto |
| 🔴 | Peer malicioso |
| 🟣 | Peer corrotto |
| Linea continua | Peer buono |
| Linea tratteggiata | Peer malicioso/corrotto |

## 📈 Interpretazione Risultati

### Trust Values
- **Vicino a 1.0**: Peer affidabile
- **Vicino a 0.0**: Peer non affidabile
- **Diminuzione nel tempo**: Peer che sta perdendo fiducia

### Detection Efficacy
- **High Detection Rate**: Sistema funziona bene
- **Low False Positives**: Pochi nodi buoni rimossi per errore
- **Trend crescente**: Sistema migliora nel tempo

### Structural Consistency
- **1.0**: Comportamento consistente
- **0.0**: Comportamento anomalo rilevato

### Updates Contribution
- **1.0**: Aggiornamenti nel percorso principale
- **0.0**: Aggiornamenti esclusi dal percorso principale
- **0.5**: Peer non ha partecipato al round

## 🔧 Personalizzazione

### Cambiare File CSV di Input
```bash
python analyze_dag_csv.py
# Inserisci il nome del file quando richiesto
```

o

```bash
python quick_dag_analysis.py mio_file.csv
```

### Modificare Directory Output
Nel file `analyze_dag_csv.py`, modifica:
```python
self.output_dir = "dag_analysis_plots"  # Cambia qui
```

## 📊 Dati CSV Analizzati

Lo script analizza le seguenti colonne dal CSV:
- `Round`: Numero del round
- `Total_Nodes`, `Total_Tips`: Stato del DAG
- `*_This_Round`: Eventi del round corrente
- `Total_*`: Contatori cumulativi
- `Peer_X_Trust`: Valore di trust per peer X
- `Peer_X_Structural_Consistency`: Consistenza strutturale
- `Peer_X_Updates_Contribution`: Contributo aggiornamenti
- `Peer_X_Is_Malicious/Corrupted`: Flag status peer

## 🐛 Troubleshooting

### Errore "File not found"
- Verifica che il file CSV sia nella directory corrente
- Controlla il nome del file (case-sensitive)

### Errore "Module not found"
```bash
pip install -r dag_analysis_requirements.txt
```

### Grafici non si aprono
- Su server remoto: i grafici vengono salvati come PNG
- Controlla la directory `dag_analysis_plots/`

## 📝 Esempio Output

```
🔍 DAG Trust Analysis Visualizer
========================================
📁 Nome del file CSV (default: dag_trust_analysis.csv): 
✅ Dati caricati: 20 rounds, 29 colonne
📊 Numero di peers rilevati: 5
📈 Creazione grafici...
💾 Salvato: dag_analysis_plots/system_evolution.png
💾 Salvato: dag_analysis_plots/trust_evolution.png
💾 Salvato: dag_analysis_plots/consistency_contributions.png
💾 Salvato: dag_analysis_plots/detection_analysis.png

============================================================
📊 STATISTICHE RIASSUNTIVE DEL SISTEMA DAG TRUST
============================================================

🔄 SISTEMA:
   • Rounds totali: 20
   • Peers nel sistema: 5
   • Nodi finali nel DAG: 45
   • Tips finali: 12

🔴 PEERS MALICIOSI: [0, 1]
🟣 PEERS CORROTTI: []

🎯 DETECTION:
   • Nodi maliciosi confermati: 8
   • Nodi corrotti confermati: 0
   • Nodi buoni rimossi (falsi positivi): 2
   • Tasso detection maliciosi: 80.0%
   • Tasso falsi positivi: 6.7%

🤝 TRUST FINALE:
   • Peer 0 🔴: 0.234
   • Peer 1 🔴: 0.187
   • Peer 2 🟢: 0.876
   • Peer 3 🟢: 0.923
   • Peer 4 🟢: 0.891

✅ Report completo generato in: dag_analysis_plots/
```

## 🎯 Conclusioni

Questi strumenti ti permettono di:
1. **Monitorare** l'evoluzione del sistema DAG Trust
2. **Valutare** l'efficacia nella detection dei peer maliciosi
3. **Identificare** trend e pattern nel comportamento del sistema
4. **Quantificare** le performance del sistema di trust

Usa `analyze_dag_csv.py` per analisi complete e `quick_dag_analysis.py` per controlli rapidi! 🚀
