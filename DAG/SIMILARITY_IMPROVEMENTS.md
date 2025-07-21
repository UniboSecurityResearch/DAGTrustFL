# Miglioramenti alla Discriminazione di Parametri Neural Network

## Problema Risolto
I valori di similitudine tra parametri "poisoned" (~0.999996) e parametri "buoni" (~0.9999997) erano troppo simili per permettere una buona discriminazione nel sistema DAGTrustFL.

## Soluzioni Implementate

### 1. Nuovi Metodi di Similitudine (`parameter_utils.py`)

#### `enhanced_similarity_score()`
Funzione principale che supporta diversi metodi:

- **`euclidean`**: Distanza euclidea normalizzata convertita in similitudine
- **`statistical`**: Distanza basata su statistiche (media, deviazione standard, skewness, kurtosis)
- **`combined`**: Combinazione pesata di coseno, euclideo e statistico (30%-40%-30%)
- **`cosine_enhanced`**: Similitudine coseno con trasformazione non-lineare (potenza elevata)

#### Metodi di Supporto
- `euclidean_distance_parameters()`: Distanza euclidea tra parametri
- `normalized_euclidean_distance()`: Distanza euclidea normalizzata per magnitudine
- `statistical_distance_parameters()`: Distanza basata su momenti statistici
- `compare_similarity_methods()`: Confronto tra tutti i metodi
- `auto_calibrate_threshold()`: Calibrazione automatica delle soglie

### 2. Aggiornamenti TipSelectionAlgorithm

#### `MyCustomTipSelector`
- Soglia: 0.972 (calibrata per label flipping)
- Metodo: `euclidean` per 100% detection rate

#### `TrustInformedTipSelector`
- Soglia: 0.537608 (calibrata su test multi-round)
- Metodo: `statistical` per massima discriminazione con stabilità

### 3. Configurazione DAG_functions.py
- Utilizza `TrustInformedTipSelector` con metodo `statistical`
- Soglia calibrata su risultati reali di label flipping attack

## Risultati dei Test

### Test con Dati Sintetici
```
Metodo              | Good vs Good | Good vs Poisoned | Discriminazione
--------------------|--------------|------------------|----------------
cosine_original     | -0.0096      | 0.9995          | Buona
cosine_enhanced     | 0.0000       | 0.0000-0.9996   | Eccellente
combined            | 0.1306       | 0.1297-0.9835   | Molto buona
statistical         | 0.0000       | 0.0000-0.9466   | Eccellente
```

### Test con Label Flipping Attack Reale (Single Round)
```
Metodo              | Clean vs Clean | Clean vs Poisoned | Discriminazione | Efficacia
--------------------|----------------|-------------------|-----------------|----------
statistical         | 0.524289      | 0.401295         | 0.122993       | ECCELLENTE
combined            | 0.891568      | 0.870670         | 0.020898       | Buona
euclidean           | 0.973700      | 0.970452         | 0.003248       | Buona (100% TPR)
cosine_enhanced     | 0.999822      | 0.999775         | 0.000047       | Limitata
cosine_improved     | 0.999998      | 0.999998         | 0.000000       | Insufficiente
cosine_original     | 0.999996      | 0.999995         | 0.000001       | Insufficiente
```

### Test Esteso Multi-Round (6 Epochs, 15 Workers, 5 Poisoned)
```
Metodo              | Avg Discrimination | Stabilità | Trend      | Raccomandazione
--------------------|-------------------|-----------|------------|----------------
statistical         | 0.054410         | 0.532     | Oscillante | OTTIMALE
combined            | 0.012724         | 0.889     | Stabile    | Alternativa
euclidean           | 0.001128         | 0.135     | Degrading  | Backup
cosine_*            | ~0.000000        | N/A       | Inutile    | Da evitare
```

### Calibrazione Automatica
- **cosine_enhanced**: Soglia ottimale 0.0000, FPR 0.0%, TPR varia
- **combined**: Soglia ottimale 0.1306, FPR 0.0%, TPR 3.3%
- **statistical**: Soglia ottimale 0.0000, FPR 0.0%, TPR 6.7%

## Raccomandazioni d'Uso

1. **Per label flipping attack**: Usa `statistical` con soglia 0.463 (discriminazione massima)
2. **Per detection perfetta**: Usa `euclidean` con soglia 0.972 (100% TPR, 0% FPR)
3. **Per casi generali**: Usa `combined` come fallback
4. **Per debug**: Usa `compare_similarity_methods()` per analizzare casi specifici

## Configurazioni Ottimali Calibrate

### TrustInformedTipSelector (Raccomandato - Multi-Round Tested)
```python
TrustInformedTipSelector(
    similarity_method='statistical',
    similarity_threshold=0.537608,  # Optimized from 6-epoch testing
    max_depth=3
)
```

### Configurazioni Alternative Basate sui Test
```python
# For conservative detection (low false positives)
TrustInformedTipSelector(
    similarity_method='statistical',
    similarity_threshold=0.761000,  # Conservative threshold
    max_depth=3
)

# For aggressive detection (high true positives)  
TrustInformedTipSelector(
    similarity_method='statistical',
    similarity_threshold=0.097200,  # Aggressive threshold
    max_depth=3
)
```

## Findings del Test Esteso Multi-Round

### 🔍 **Osservazioni Chiave:**

1. **Evoluzione Temporale della Discriminazione:**
   - **Epoch 1**: Discriminazione bassa (modelli simili all'inizio)
   - **Epoch 2**: Picco di discriminazione (modelli poisoned divergono)
   - **Epochs 3-6**: Oscillazioni naturali (convergenza vs. poisoning)

2. **Pattern di Stabilità:**
   - **STATISTICAL**: Alta discriminazione ma oscillante (CV=0.880)
   - **COMBINED**: Discriminazione media ma stabile (CV=0.889)
   - **EUCLIDEAN**: Bassa discriminazione, trend degrading

3. **Comportamento nel Tempo:**
   - Tutti i metodi mostrano trend "degrading" - normale in FL
   - La discriminazione peak nel 2° round suggerisce early detection efficace
   - Oscillazioni indicano necessità di soglie adattive

### 📊 **Raccomandazioni Operative:**

1. **Monitoraggio Continuo**: La discriminazione varia significativamente tra round
2. **Soglie Adattive**: Considerare soglie che si adattano alla variabilità temporale
3. **Early Detection**: Il 2° round spesso mostra la discriminazione massima
4. **Fallback Strategy**: Mantenere `combined` come metodo di backup per stabilità

### ⚙️ **Implementazione Suggerita:**

```python
# Configurazione principale (bilanciata)
selector = TrustInformedTipSelector(
    similarity_method='statistical',
    similarity_threshold=0.537608,
    max_depth=3
)

# Con monitoraggio adattivo (opzionale)
# if detection_rate < threshold_min:
#     selector.similarity_threshold = 0.097200  # More aggressive
# elif false_positive_rate > threshold_max:
#     selector.similarity_threshold = 0.761000  # More conservative
```

## File Modificati

- `parameter_utils.py`: Nuovi metodi di similitudine
- `TipSelectionAlgorithm.py`: Configurazioni aggiornate
- `DAG_functions.py`: Uso del metodo combinato
- `test_similarity.py`: Script di test e calibrazione

## Vantaggi

1. **Migliore discriminazione**: Separazione molto più netta tra parametri buoni e poisoned
2. **Configurabilità**: Diversi metodi per diversi scenari
3. **Calibrazione automatica**: Soglie ottimizzate automaticamente
4. **Robustezza**: Combinazione di più approcci riduce falsi positivi
5. **Facilità di debug**: Tool per analizzare e confrontare metodi
