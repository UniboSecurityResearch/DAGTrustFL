# DAG Trust Analysis Tools - PDF Edition 📊📑

Strumenti aggiornati per analizzare e visualizzare i dati CSV generati dal sistema DAG Trust per Federated Learning.
**Nuove versioni con supporto PDF completo!**

## 🆕 Novità - Versioni PDF

### Nuovi File Disponibili:
- **`analyze_dag_csv_pdf.py`**: Versione PDF completa - crea un singolo PDF con tutti i grafici
- **`quick_dag_analysis_pdf.py`**: Versione PDF rapida - analisi veloce in PDF

### File Originali (PNG):
- **`analyze_dag_csv.py`**: Versione originale con grafici PNG
- **`quick_dag_analysis.py`**: Versione rapida con grafici PNG

## 🚀 Quick Start

### 1. Installazione Dipendenze
```bash
pip install -r dag_analysis_requirements.txt
```

### 2. Analisi Completa PDF (⭐ RACCOMANDATO)
```bash
python analyze_dag_csv_pdf.py
```
**Output**: `complete_dag_analysis_report.pdf` (tutto in un unico PDF!)

### 3. Analisi Rapida PDF
```bash
python quick_dag_analysis_pdf.py [file.csv]
```
**Output**: `quick_dag_analysis.pdf` (panoramica veloce in PDF)

### 4. Analisi PNG (Versioni Originali)
```bash
python analyze_dag_csv.py        # Multiple PNG files
python quick_dag_analysis.py     # Single PNG file
```

## 📁 Confronto Versioni

| Script | Output | Formato | Pagine | Tempo | Uso Raccomandato |
|--------|--------|---------|--------|-------|------------------|
| `analyze_dag_csv_pdf.py` | 1 PDF completo | PDF | 5 | ~15s | ⭐ **Report finale** |
| `quick_dag_analysis_pdf.py` | 1 PDF veloce | PDF | 2 | ~5s | 🚀 **Check rapido** |
| `analyze_dag_csv.py` | 4 PNG separati | PNG | - | ~20s | 🖼️ Grafici separati |
| `quick_dag_analysis.py` | 1 PNG | PNG | - | ~3s | ⚡ Ultra veloce |

## 📑 Contenuto PDF Completo (`analyze_dag_csv_pdf.py`)

### Pagina 1: System Evolution
- 🔄 **Evoluzione del DAG**: Nodi totali e tips nel tempo
- 📊 **Nodi per Round**: Rimossi vs confermati per round  
- 📈 **Evoluzione Cumulativa**: Trend cumulativi
- 🎯 **Efficacia Detection**: Tasso di detection maliciosi

### Pagina 2: Trust Evolution
- 🤝 **Trust Vector**: Evoluzione trust per peer (buoni vs maliciosi)
- 🔥 **Heatmap Trust**: Mappa di calore dell'evoluzione trust

### Pagina 3: Consistency & Contributions
- 🏗️ **Structural Consistency**: Evoluzione consistenza strutturale
- 📤 **Updates Contribution**: Evoluzione contributi aggiornamenti
- 📊 **Distribuzioni Finali**: Valori finali per peer

### Pagina 4: Detection Analysis  
- 🎯 **Tasso Detection**: Percentuale maliciosi rilevati
- ⚠️ **Falsi Positivi**: Percentuale buoni rimossi erroneamente
- 📦 **Box Plot Trust**: Confronto trust finale
- ⏰ **Timeline Eventi**: Sequenza eventi nel tempo

### Pagina 5: Summary Statistics
- 📋 **Statistiche Complete**: Riassunto numerico dettagliato
- 🔢 **Tassi e Percentuali**: Metriche di performance
- 🤝 **Trust Finale**: Valore finale per ogni peer
- 📅 **Info Report**: Data generazione e file sorgente

## 📊 Quick Analysis PDF (`quick_dag_analysis_pdf.py`)

### Pagina 1: Overview Grafica
- 4 grafici principali in una panoramica
- Massimo 8 peer mostrati per chiarezza
- Focus sui trend principali

### Pagina 2: Statistiche Dettagliate
- Risultati numerici completi
- Interpretazione automatica (✅ Efficace / ⚠️ Da ottimizzare)
- Trust finale per tutti i peer

## 🎨 Legenda Grafici PDF

| Simbolo/Stile | Significato |
|---------------|-------------|
| Linea continua | Peer buono/onesto |
| Linea tratteggiata | Peer malicioso |
| **G** (verde) | Good peer |
| **M** (rosso) | Malicious peer |
| **C** (viola) | Corrupted peer |

## 📈 Vantaggi Versioni PDF

✅ **Un singolo file** - Tutto in un documento  
✅ **Professionale** - Pronto per presentazioni  
✅ **Portatile** - Facile da condividere  
✅ **Stampabile** - Layout ottimizzato per stampa  
✅ **Navigabile** - Pagine organizzate logicamente  
✅ **Compatto** - File size ottimizzato  

## 🔧 Personalizzazione

### Cambiare File di Input
```bash
# Metodo interattivo
python analyze_dag_csv_pdf.py
# Inserisci nome file quando richiesto

# Metodo diretto (solo quick)
python quick_dag_analysis_pdf.py mio_file.csv
```

### Cambiare Directory Output
Nel file `analyze_dag_csv_pdf.py`, modifica:
```python
self.output_dir = "dag_analysis_plots"  # Cambia qui
```

## 🐛 Troubleshooting

### PDF non si crea
- Controlla dipendenze: `pip install matplotlib pandas`
- Verifica permessi di scrittura nella directory
- Controlla spazio disco disponibile

### Grafici vuoti/strani
- Verifica formato CSV corretto
- Controlla che ci siano dati sufficienti
- Assicurati che i nomi delle colonne siano corretti

### Performance lente
- I PDF richiedono più tempo dei PNG (normale)
- Per analisi ultra-rapida usa `quick_dag_analysis.py` (PNG)

## 📊 Esempio Output Console

```bash
🔍 DAG Trust Analysis Visualizer - PDF Edition
==================================================
📁 Nome del file CSV (default: dag_trust_analysis.csv): 
✅ Dati caricati: 20 rounds, 29 colonne
📊 Numero di peers rilevati: 5
📑 Creazione PDF completo...
   📊 System Evolution...
   🤝 Trust Evolution...
   🏗️ Consistency & Contributions...
   🎯 Detection Analysis...
   📋 Summary Statistics...
📑 PDF completo salvato: dag_analysis_plots/complete_dag_analysis_report.pdf

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
   • Tasso detection maliciosi: 80.0%
   • Tasso falsi positivi: 6.7%

🤝 TRUST FINALE:
   • Peer 0 🔴: 0.234
   • Peer 1 🔴: 0.187
   • Peer 2 🟢: 0.876

✅ Report PDF completo generato!
📁 File principale: dag_analysis_plots/complete_dag_analysis_report.pdf
🎉 Analisi PDF completata con successo!
```

## 🎯 Raccomandazioni d'Uso

### Per Sviluppo/Debug:
- Usa `quick_dag_analysis_pdf.py` per controlli rapidi
- Ottimo per verificare che il sistema funzioni

### Per Presentazioni/Report:
- Usa `analyze_dag_csv_pdf.py` per analisi complete
- PDF professionale pronto per presentazioni

### Per Analisi Dettagliate:
- Usa `analyze_dag_csv.py` (PNG) se hai bisogno di grafici separati
- Utile per includere grafici specifici in documenti

### Per Debugging Ultra-Rapido:
- Usa `quick_dag_analysis.py` (PNG) per controlli istantanei
- Massima velocità per development

## 🎉 Conclusioni

Le **nuove versioni PDF** rendono l'analisi del sistema DAG Trust più professionale e condivisibile. 

**Raccomandazione**: Inizia sempre con `quick_dag_analysis_pdf.py` per una panoramica rapida, poi usa `analyze_dag_csv_pdf.py` per il report completo! 📊📑✨
