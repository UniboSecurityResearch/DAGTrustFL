import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_rocs(scenarios, title="Attacker detection ROC (by attacker ratio)"):
    """
    scenarios: dict
        chiave = etichetta scenario (es. '10%')
        valore = tuple (y_true, y_score)
            y_true: array-like di 0/1 (1=attaccante, 0=benigno)
            y_score: array-like di score continui (più alto = più sospetto)
    """
    plt.figure(figsize=(7, 6))
    aucs = {}

    for label, (y_true, y_score) in scenarios.items():
        y_true  = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        aucs[label] = roc_auc

        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    # baseline
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # stampa ordinata per confronto rapido
    print("AUC by scenario:")
    for k, v in sorted(aucs.items(), key=lambda kv: kv[0]):
        print(f"  {k}: {v:.4f}")

# ==========================
# ESEMPIO D’USO (sostituisci coi tuoi dati)
# ==========================
# Nota: y_true deve avere 1 per gli attaccanti, 0 per i benigni.
#       y_score è il punteggio/anomalia/trust che il tuo sistema produce (continuo).

# Simulazione minima per la struttura dei dati:
y_true_10 = [0,1,0,0,1,0,0,0,1,0]           # <- rimpiazza
y_score_10 = [0.1,0.9,0.2,0.3,0.85,0.4,0.2,0.15,0.8,0.05]

y_true_20 = [0,1,0,1,0,0,0,1,0,0]
y_score_20 = [0.05,0.92,0.12,0.8,0.2,0.18,0.1,0.77,0.22,0.16]

y_true_30 = [1,0,1,0,1,0,0,1,0,0]
y_score_30 = [0.85,0.25,0.78,0.2,0.8,0.3,0.15,0.75,0.22,0.18]

y_true_40 = [1,1,0,1,0,0,1,0,1,0]
y_score_40 = [0.9,0.82,0.28,0.77,0.2,0.18,0.8,0.3,0.75,0.22]

scenarios = {
    "10%": (y_true_10, y_score_10),
    "20%": (y_true_20, y_score_20),
    "30%": (y_true_30, y_score_30),
    "40%": (y_true_40, y_score_40),
}

plot_rocs(scenarios)
