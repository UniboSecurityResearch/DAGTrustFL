#!/usr/bin/env python3
"""
Advanced analysis script che genera raccomandazioni per soglie adattive
basate sui risultati del test esteso.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DAG'))

import numpy as np
import parameter_utils

class AdaptiveThresholdRecommender:
    """Classe per generare raccomandazioni di soglie adattive."""
    
    def __init__(self, test_results):
        """
        Args:
            test_results: Risultati dal test esteso nel formato:
                {epoch: {method: {discrimination, clean_avg, poisoned_avg, ...}}}
        """
        self.results = test_results
        self.methods = ['statistical', 'combined', 'euclidean']
    
    def analyze_temporal_patterns(self):
        """Analizza i pattern temporali della discriminazione."""
        print("="*80)
        print("ADAPTIVE THRESHOLD ANALYSIS")
        print("="*80)
        
        for method in self.methods:
            print(f"\n{method.upper()} Analysis:")
            
            # Estrai dati temporali
            epochs = sorted(self.results.keys())
            discriminations = []
            clean_avgs = []
            poisoned_avgs = []
            thresholds = []
            
            for epoch in epochs:
                if method in self.results[epoch]:
                    data = self.results[epoch][method]
                    discriminations.append(data['discrimination'])
                    clean_avgs.append(data['clean_clean_avg'])
                    poisoned_avgs.append(data['clean_poisoned_avg'])
                    threshold = (data['clean_clean_avg'] + data['clean_poisoned_avg']) / 2
                    thresholds.append(threshold)
            
            if not discriminations:
                continue
                
            # Statistiche di base
            print(f"  Discrimination stats: min={min(discriminations):.6f}, "
                  f"max={max(discriminations):.6f}, avg={np.mean(discriminations):.6f}")
            print(f"  Threshold range: {min(thresholds):.6f} - {max(thresholds):.6f}")
            
            # Raccomandazioni per soglie
            self.recommend_thresholds(method, discriminations, thresholds, clean_avgs, poisoned_avgs)
    
    def recommend_thresholds(self, method, discriminations, thresholds, clean_avgs, poisoned_avgs):
        """Genera raccomandazioni di soglie per un metodo."""
        
        # Soglia conservativa (massimo clean_avg osservato)
        conservative_threshold = max(clean_avgs) if clean_avgs else 0.5
        
        # Soglia aggressiva (minimo clean_avg osservato)
        aggressive_threshold = min(clean_avgs) if clean_avgs else 0.5
        
        # Soglia adattiva (media dei threshold ottimali)
        adaptive_threshold = np.mean(thresholds) if thresholds else 0.5
        
        # Soglia dinamica (basata sulla variabilità)
        threshold_std = np.std(thresholds) if thresholds else 0
        margin = 2 * threshold_std  # 2-sigma margin
        dynamic_threshold = adaptive_threshold - margin
        
        print(f"  Threshold Recommendations:")
        print(f"    Conservative (low FPR): {conservative_threshold:.6f}")
        print(f"    Aggressive (high TPR):  {aggressive_threshold:.6f}")
        print(f"    Adaptive (balanced):    {adaptive_threshold:.6f}")
        print(f"    Dynamic (with margin):  {dynamic_threshold:.6f}")
        
        # Calcola efficacia di ciascuna soglia
        self.evaluate_threshold_effectiveness(method, thresholds, clean_avgs, poisoned_avgs)
    
    def evaluate_threshold_effectiveness(self, method, thresholds, clean_avgs, poisoned_avgs):
        """Valuta l'efficacia delle soglie raccomandate."""
        
        if not thresholds or not clean_avgs or not poisoned_avgs:
            return
            
        # Test delle soglie raccomandate
        test_thresholds = {
            'conservative': max(clean_avgs),
            'aggressive': min(clean_avgs),
            'adaptive': np.mean(thresholds),
            'dynamic': np.mean(thresholds) - 2*np.std(thresholds)
        }
        
        print(f"  Effectiveness Analysis:")
        for thresh_name, thresh_value in test_thresholds.items():
            # Simula detection rates
            # (In un test reale, useresti dati ground truth)
            tpr_estimate = sum(1 for p_avg in poisoned_avgs if p_avg < thresh_value) / len(poisoned_avgs)
            fpr_estimate = sum(1 for c_avg in clean_avgs if c_avg < thresh_value) / len(clean_avgs)
            
            print(f"    {thresh_name:12}: threshold={thresh_value:.6f}, "
                  f"TPR≈{tpr_estimate:.2f}, FPR≈{fpr_estimate:.2f}")
    
    def generate_config_recommendations(self):
        """Genera raccomandazioni di configurazione finale."""
        print(f"\n" + "="*80)
        print("CONFIGURATION RECOMMENDATIONS")
        print("="*80)
        
        # Analizza quale metodo è più stabile nel tempo
        stability_scores = {}
        
        for method in self.methods:
            epochs = sorted(self.results.keys())
            discriminations = []
            
            for epoch in epochs:
                if method in self.results[epoch]:
                    discriminations.append(self.results[epoch][method]['discrimination'])
            
            if discriminations:
                # Stabilità = 1 / coefficient of variation
                avg_disc = np.mean(discriminations)
                std_disc = np.std(discriminations)
                cv = std_disc / avg_disc if avg_disc > 0 else float('inf')
                stability = 1 / (1 + cv)  # Normalized stability score
                stability_scores[method] = {
                    'avg_discrimination': avg_disc,
                    'stability': stability,
                    'cv': cv
                }
        
        # Ordina per combinazione di discriminazione e stabilità
        def combined_score(method_data):
            return method_data['avg_discrimination'] * method_data['stability']
        
        best_method = max(stability_scores.keys(), 
                         key=lambda m: combined_score(stability_scores[m]))
        
        print(f"Recommended Method: {best_method.upper()}")
        print(f"  Average Discrimination: {stability_scores[best_method]['avg_discrimination']:.6f}")
        print(f"  Stability Score: {stability_scores[best_method]['stability']:.3f}")
        print(f"  Coefficient of Variation: {stability_scores[best_method]['cv']:.3f}")
        
        # Genera configurazione finale
        epochs = sorted(self.results.keys())
        thresholds = []
        for epoch in epochs:
            if best_method in self.results[epoch]:
                data = self.results[epoch][best_method]
                threshold = (data['clean_clean_avg'] + data['clean_poisoned_avg']) / 2
                thresholds.append(threshold)
        
        final_threshold = np.mean(thresholds) if thresholds else 0.5
        
        print(f"\nFinal Configuration:")
        print(f"```python")
        print(f"TrustInformedTipSelector(")
        print(f"    similarity_method='{best_method}',")
        print(f"    similarity_threshold={final_threshold:.6f},")
        print(f"    max_depth=3")
        print(f")")
        print(f"```")
        
        return best_method, final_threshold

def main():
    """Simula i risultati del test per demo."""
    
    # Simulated results dal test esteso (basato sui risultati reali)
    simulated_results = {
        0: {  # Epoch 1
            'statistical': {'discrimination': 0.005066, 'clean_clean_avg': 0.0972, 'clean_poisoned_avg': 0.1022},
            'combined': {'discrimination': 0.005875, 'clean_clean_avg': 0.7396, 'clean_poisoned_avg': 0.7455},
            'euclidean': {'discrimination': 0.004526, 'clean_clean_avg': 0.9563, 'clean_poisoned_avg': 0.9517}
        },
        1: {  # Epoch 2
            'statistical': {'discrimination': 0.152550, 'clean_clean_avg': 0.3464, 'clean_poisoned_avg': 0.4989},
            'combined': {'discrimination': 0.037127, 'clean_clean_avg': 0.8620, 'clean_poisoned_avg': 0.8991},
            'euclidean': {'discrimination': 0.000541, 'clean_clean_avg': 0.9845, 'clean_poisoned_avg': 0.9839}
        },
        2: {  # Epoch 3
            'statistical': {'discrimination': 0.053106, 'clean_clean_avg': 0.6600, 'clean_poisoned_avg': 0.6069},
            'combined': {'discrimination': 0.010878, 'clean_clean_avg': 0.9380, 'clean_poisoned_avg': 0.9272},
            'euclidean': {'discrimination': 0.000468, 'clean_clean_avg': 0.9903, 'clean_poisoned_avg': 0.9898}
        },
        3: {  # Epoch 4
            'statistical': {'discrimination': 0.023865, 'clean_clean_avg': 0.5916, 'clean_poisoned_avg': 0.5677},
            'combined': {'discrimination': 0.004763, 'clean_clean_avg': 0.9252, 'clean_poisoned_avg': 0.9204},
            'euclidean': {'discrimination': 0.000508, 'clean_clean_avg': 0.9926, 'clean_poisoned_avg': 0.9921}
        },
        4: {  # Epoch 5
            'statistical': {'discrimination': 0.063255, 'clean_clean_avg': 0.6833, 'clean_poisoned_avg': 0.7465},
            'combined': {'discrimination': 0.012767, 'clean_clean_avg': 0.9446, 'clean_poisoned_avg': 0.9574},
            'euclidean': {'discrimination': 0.000313, 'clean_clean_avg': 0.9940, 'clean_poisoned_avg': 0.9936}
        },
        5: {  # Epoch 6
            'statistical': {'discrimination': 0.028620, 'clean_clean_avg': 0.7610, 'clean_poisoned_avg': 0.7896},
            'combined': {'discrimination': 0.004935, 'clean_clean_avg': 0.9598, 'clean_poisoned_avg': 0.9647},
            'euclidean': {'discrimination': 0.000411, 'clean_clean_avg': 0.9949, 'clean_poisoned_avg': 0.9945}
        }
    }
    
    # Analizza e genera raccomandazioni
    recommender = AdaptiveThresholdRecommender(simulated_results)
    recommender.analyze_temporal_patterns()
    best_method, final_threshold = recommender.generate_config_recommendations()
    
    print(f"\n" + "="*80)
    print("SUMMARY FOR DAG CONFIGURATION")
    print("="*80)
    print(f"Update DAG_functions.py with:")
    print(f"selector = TipSelectionAlgorithm.TrustInformedTipSelector(")
    print(f"    similarity_method='{best_method}',")
    print(f"    similarity_threshold={final_threshold:.6f}")
    print(f")")

if __name__ == "__main__":
    main()
