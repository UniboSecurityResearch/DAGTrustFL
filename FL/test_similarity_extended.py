#!/usr/bin/env python3
"""
Test esteso per simulare multiple round di federated learning con label flipping attack.
Questo script simula un esperimento completo usando il framework esistente per vedere
come si comportano i meccanismi di similitudine attraverso diversi round di training.
"""

import sys
import os

# Add DAG directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DAG'))

import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import DAG utilities
import parameter_utils

# Import FL components
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import replace_1_with_9, replace_4_with_6, replace_0_with_2
from federated_learning.utils import average_nn_parameters
from federated_learning.worker_selection import RandomSelectionStrategy
from client import Client
from loguru import logger

class FL_SimilarityTester:
    """Classe per testare la similitudine attraverso più round di FL."""
    
    def __init__(self, num_workers=12, num_poisoned=4, num_epochs=20, workers_per_round=8):
        self.num_workers = num_workers
        self.num_poisoned = num_poisoned
        self.num_epochs = num_epochs
        self.workers_per_round = workers_per_round
        self.poisoned_indices = list(range(num_poisoned))  # Prime N workers poisoned
        
        # Risultati per analisi dettagliata
        self.results_by_epoch = {}
        self.similarity_evolution = {method: [] for method in self.get_methods()}
        
        # Analisi approfondita per threshold detection
        self.detailed_similarities = []  # Lista di tutte le similitudini per epoch
        self.threshold_analysis = {
            'optimal_thresholds': [],
            'accuracy_per_threshold': [],
            'false_positive_rates': [],
            'false_negative_rates': []
        }
        
        # Statistiche per valutare l'efficacia
        self.detection_stats = {
            'true_positives': [],  # Poisoned correttamente identificati
            'false_positives': [], # Clean identificati come poisoned
            'true_negatives': [],  # Clean correttamente identificati
            'false_negatives': []  # Poisoned identificati come clean
        }
        
        # Setup del sistema FL
        self.setup_fl_system()
    
    def get_methods(self):
        """Ritorna i metodi di similitudine da testare."""
        return ['cosine_original']
    
    def setup_fl_system(self):
        """Setup del sistema di federated learning."""
        print(f"Setting up FL system: {self.num_workers} workers, {self.num_poisoned} poisoned")
        
        # Setup args
        self.args = Arguments(logger)
        self.args.set_num_workers(self.num_workers)
        self.args.set_num_poisoned_workers(self.num_poisoned)
        self.args.set_num_epochs(self.num_epochs)
        
        # Load and distribute data
        train_data_loader = load_train_data_loader(logger, self.args)
        test_data_loader = load_test_data_loader(logger, self.args)
        
        # Distribute data equally among workers
        distributed_train_dataset = distribute_batches_equally(train_data_loader, self.num_workers)
        distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
        
        # Apply poisoning to specified workers
        print(f"Applying label flipping to workers: {self.poisoned_indices}")
        poisoned_distributed_dataset = poison_data(
            logger, distributed_train_dataset, self.num_workers, 
            self.poisoned_indices, replace_1_with_9
        )
        
        # Create data loaders
        train_data_loaders = generate_data_loaders_from_distributed_dataset(
            poisoned_distributed_dataset, self.args.get_batch_size()
        )
        
        # Create clients
        self.clients = []
        for idx in range(self.num_workers):
            client = Client(self.args, idx, train_data_loaders[idx], test_data_loader)
            client.is_poisoned = idx in self.poisoned_indices  # Mark poisoned clients
            self.clients.append(client)
        
        print("FL system setup completed")
    
    def select_workers_for_round(self, epoch):
        """Seleziona workers per il round corrente."""
        # Simula selezione casuale ma garantisce rappresentanza di entrambi i tipi
        np.random.seed(epoch)  # Seed per riproducibilità
        
        # Assicurati di includere almeno alcuni poisoned e clean workers
        min_poisoned = min(2, self.num_poisoned)
        min_clean = min(2, self.num_workers - self.num_poisoned)
        
        # Seleziona workers poisoned
        available_poisoned = [i for i in self.poisoned_indices]
        selected_poisoned = np.random.choice(
            available_poisoned, 
            size=min(min_poisoned, len(available_poisoned), self.workers_per_round//2), 
            replace=False
        ).tolist()
        
        # Seleziona workers clean
        available_clean = [i for i in range(self.num_workers) if i not in self.poisoned_indices]
        remaining_slots = self.workers_per_round - len(selected_poisoned)
        selected_clean = np.random.choice(
            available_clean, 
            size=min(remaining_slots, len(available_clean)), 
            replace=False
        ).tolist()
        
        selected_workers = selected_poisoned + selected_clean
        np.random.shuffle(selected_workers)  # Mescola per ordine casuale
        
        return selected_workers
    
    def train_round(self, epoch, selected_workers):
        """Allena i workers selezionati per un round."""
        print(f"\n--- EPOCH {epoch+1} ---")
        print(f"Selected workers: {selected_workers}")
        
        # Identifica poisoned vs clean nella selezione
        selected_poisoned = [w for w in selected_workers if w in self.poisoned_indices]
        selected_clean = [w for w in selected_workers if w not in self.poisoned_indices]
        
        print(f"  Poisoned workers in round: {selected_poisoned}")
        print(f"  Clean workers in round: {selected_clean}")
        
        # Train selected clients
        for worker_idx in selected_workers:
            self.clients[worker_idx].train(epoch)
        
        # Collect parameters
        parameters_dict = {}
        clean_params = []
        poisoned_params = []
        
        for worker_idx in selected_workers:
            params = self.clients[worker_idx].get_nn_parameters()
            parameters_dict[worker_idx] = params
            
            if worker_idx in self.poisoned_indices:
                poisoned_params.append(params)
            else:
                clean_params.append(params)
        
        return parameters_dict, clean_params, poisoned_params, selected_workers
    
    def analyze_round_similarities(self, epoch, clean_params, poisoned_params, selected_workers):
        """Analizza le similitudini per il round corrente con analisi dettagliata."""
        if len(clean_params) < 2 or len(poisoned_params) < 1:
            print(f"  Skipping similarity analysis - insufficient samples")
            return None
        
        print(f"  Analyzing similarities: {len(clean_params)} clean, {len(poisoned_params)} poisoned")
        
        methods = self.get_methods()
        results = {}
        
        for method in methods:
            clean_clean_sims = []
            clean_poisoned_sims = []
            all_similarities = []  # Per analisi threshold
            all_labels = []        # True=clean, False=poisoned
            
            # Clean vs Clean (reference baseline)
            for i in range(len(clean_params)):
                for j in range(i+1, len(clean_params)):
                    sim = self.calculate_similarity(clean_params[i], clean_params[j], method)
                    if sim is not None:
                        clean_clean_sims.append(sim)
                        all_similarities.append(sim)
                        all_labels.append(True)  # Clean pair
            
            # Clean vs Poisoned (detection target)
            for clean_param in clean_params:
                for poisoned_param in poisoned_params:
                    sim = self.calculate_similarity(clean_param, poisoned_param, method)
                    if sim is not None:
                        clean_poisoned_sims.append(sim)
                        all_similarities.append(sim)
                        all_labels.append(False)  # Contains poisoned
            
            # Calcola statistiche base
            if clean_clean_sims and clean_poisoned_sims:
                clean_avg = np.mean(clean_clean_sims)
                clean_std = np.std(clean_clean_sims)
                poisoned_avg = np.mean(clean_poisoned_sims)
                poisoned_std = np.std(clean_poisoned_sims)
                discrimination = abs(clean_avg - poisoned_avg)
                
                # Analisi distribuzione overlap
                overlap_analysis = self.analyze_distribution_overlap(clean_clean_sims, clean_poisoned_sims)
                
                # Trova threshold ottimale per questo epoch
                optimal_threshold, threshold_stats = self.find_optimal_threshold(
                    all_similarities, all_labels, clean_avg, poisoned_avg
                )
                
                results[method] = {
                    'clean_clean_avg': clean_avg,
                    'clean_clean_std': clean_std,
                    'clean_poisoned_avg': poisoned_avg,
                    'clean_poisoned_std': poisoned_std,
                    'discrimination': discrimination,
                    'clean_clean_range': (min(clean_clean_sims), max(clean_clean_sims)),
                    'clean_poisoned_range': (min(clean_poisoned_sims), max(clean_poisoned_sims)),
                    'overlap_analysis': overlap_analysis,
                    'optimal_threshold': optimal_threshold,
                    'threshold_stats': threshold_stats,
                    'separability_ratio': discrimination / (clean_std + poisoned_std + 1e-8)
                }
                
                # Salva per analisi temporale
                epoch_data = {
                    'epoch': epoch,
                    'discrimination': discrimination,
                    'clean_avg': clean_avg,
                    'clean_std': clean_std,
                    'poisoned_avg': poisoned_avg,
                    'poisoned_std': poisoned_std,
                    'optimal_threshold': optimal_threshold,
                    'threshold_accuracy': threshold_stats['accuracy'],
                    'separability_ratio': discrimination / (clean_std + poisoned_std + 1e-8)
                }
                self.similarity_evolution[method].append(epoch_data)
                
                # Salva dettagli per analisi finale
                self.detailed_similarities.append({
                    'epoch': epoch,
                    'method': method,
                    'clean_clean_sims': clean_clean_sims,
                    'clean_poisoned_sims': clean_poisoned_sims,
                    'optimal_threshold': optimal_threshold
                })
        
        return results
    
    def calculate_similarity(self, params1, params2, method):
        """Calcola la similitudine usando il metodo specificato."""
        try:
            if method == 'cosine_original':
                return parameter_utils.cos_sim_simple(params1, params2)
            elif method == 'cosine_improved':
                return 0#parameter_utils.cosine_similarity_parameters_improved(params1, params2)
            else:
                return 0#parameter_utils.enhanced_similarity_score(params1, params2, method=method)
        except Exception as e:
            print(f"    Error calculating {method}: {e}")
            return None
    
    def aggregate_and_update(self, parameters_dict, selected_workers):
        """Aggrega i parametri e aggiorna i clients."""
        # Simple FedAvg per questo test
        parameters_list = [parameters_dict[w] for w in selected_workers]
        
        # Pesi uguali per semplicità (in realtà dovresti usare trust weights)
        trust_weights = [1.0] * len(parameters_list)
        
        # Aggrega i parametri
        aggregated_params = average_nn_parameters(parameters_list, trust_weights)
        
        # Aggiorna tutti i clients
        for client in self.clients:
            client.update_nn_parameters(aggregated_params)
        
        return aggregated_params
    
    def run_fl_experiment(self):
        """Esegue l'esperimento completo di FL."""
        print("="*80)
        print("STARTING FEDERATED LEARNING SIMILARITY EXPERIMENT")
        print("="*80)
        
        for epoch in range(self.num_epochs):
            # Seleziona workers per questo round
            selected_workers = self.select_workers_for_round(epoch)
            
            # Allena per questo round
            parameters_dict, clean_params, poisoned_params, selected_workers = self.train_round(
                epoch, selected_workers
            )
            
            # Analizza similitudini
            results = self.analyze_round_similarities(
                epoch, clean_params, poisoned_params, selected_workers
            )
            
            if results:
                self.results_by_epoch[epoch] = results
                self.print_epoch_results(epoch, results)
            
            # Aggrega e aggiorna
            aggregated_params = self.aggregate_and_update(parameters_dict, selected_workers)
            
            print(f"  Round {epoch+1} completed\n")
        
        # Analisi finale
        self.print_final_analysis()
    
    def print_epoch_results(self, epoch, results):
        """Stampa i risultati dettagliati per un singolo epoch."""
        print(f"  📊 Similarity Results for Epoch {epoch+1}:")
        
        # Ordina per discriminazione
        sorted_methods = sorted(results.keys(), key=lambda x: results[x]['discrimination'], reverse=True)
        
        for method in sorted_methods:
            r = results[method]
            
            # Simboli per indicare la qualità
            if r['discrimination'] > 0.02:
                quality_symbol = "🟢"  # Excellent
            elif r['discrimination'] > 0.01:
                quality_symbol = "🟡"  # Good
            elif r['discrimination'] > 0.005:
                quality_symbol = "🟠"  # Moderate
            else:
                quality_symbol = "🔴"  # Poor
            
            # Accuracy indicator
            acc_symbol = "✅" if r['threshold_stats']['accuracy'] >= 0.8 else "⚠️" if r['threshold_stats']['accuracy'] >= 0.7 else "❌"
            
            print(f"    {quality_symbol} {method:15}: Discrimination={r['discrimination']:.6f}, "
                  f"Sep.Ratio={r['separability_ratio']:.4f}")
            print(f"      {'':17} Clean={r['clean_clean_avg']:.4f}±{r['clean_clean_std']:.4f}, "
                  f"Poisoned={r['clean_poisoned_avg']:.4f}±{r['clean_poisoned_std']:.4f}")
            print(f"      {'':17} {acc_symbol} Threshold={r['optimal_threshold']:.4f}, "
                  f"Accuracy={r['threshold_stats']['accuracy']:.4f} ({r['threshold_stats']['accuracy']*100:.1f}%)")
            
            # Overlap analysis
            overlap = r['overlap_analysis']
            if overlap['has_overlap']:
                print(f"      {'':17} ⚠️  Distributions overlap: {overlap['overlap_range']:.4f} "
                      f"({overlap['clean_overlap_pct']:.1f}%/{overlap['poisoned_overlap_pct']:.1f}%)")
            else:
                print(f"      {'':17} ✅ Perfect separation: gap={overlap['total_separation']:.4f}")
        
        print()
    
    def print_final_analysis(self):
        """Stampa l'analisi finale dell'esperimento con visualizzazioni e report dettagliato."""
        print("\n" + "="*100)
        print("COMPREHENSIVE COSINE SIMILARITY ANALYSIS REPORT")
        print("="*100)
        
        methods = self.get_methods()
        
        # Analizza l'evoluzione nel tempo
        for method in methods:
            evolution = self.similarity_evolution[method]
            if not evolution:
                continue
                
            discriminations = [e['discrimination'] for e in evolution]
            separability_ratios = [e['separability_ratio'] for e in evolution]
            accuracies = [e['threshold_accuracy'] for e in evolution]
            
            print(f"\n🔍 {method.upper()} DETAILED ANALYSIS:")
            print("="*50)
            
            # Statistiche di base
            print(f"📊 Discrimination Statistics:")
            print(f"   Average: {np.mean(discriminations):.6f}")
            print(f"   Std Dev: {np.std(discriminations):.6f}")
            print(f"   Range: [{np.min(discriminations):.6f}, {np.max(discriminations):.6f}]")
            
            print(f"\n📊 Separability Statistics:")
            print(f"   Average: {np.mean(separability_ratios):.6f}")
            print(f"   Std Dev: {np.std(separability_ratios):.6f}")
            print(f"   Range: [{np.min(separability_ratios):.6f}, {np.max(separability_ratios):.6f}]")
            
            print(f"\n📊 Detection Accuracy Statistics:")
            print(f"   Average: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)")
            print(f"   Std Dev: {np.std(accuracies):.4f}")
            print(f"   Range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
            
            # Evolution timeline
            print(f"\n📈 Temporal Evolution:")
            for i, e in enumerate(evolution):
                trend_symbol = "📈" if i > 0 and discriminations[i] > discriminations[i-1] else "📉" if i > 0 else "🔄"
                print(f"   Epoch {e['epoch']+1:2d}: Disc={e['discrimination']:.6f}, "
                      f"Sep={e['separability_ratio']:.4f}, Acc={e['threshold_accuracy']:.4f} {trend_symbol}")
            
            # Trend analysis
            if len(discriminations) > 1:
                disc_trend = np.polyfit(range(len(discriminations)), discriminations, 1)[0]
                acc_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
                
                disc_trend_desc = "IMPROVING" if disc_trend > 0.001 else "DEGRADING" if disc_trend < -0.001 else "STABLE"
                acc_trend_desc = "IMPROVING" if acc_trend > 0.001 else "DEGRADING" if acc_trend < -0.001 else "STABLE"
                
                print(f"\n📈 Trend Analysis:")
                print(f"   Discrimination: {disc_trend_desc} ({disc_trend:+.6f}/epoch)")
                print(f"   Accuracy: {acc_trend_desc} ({acc_trend:+.6f}/epoch)")
            
            # Threshold stability analysis
            thresholds = [e['optimal_threshold'] for e in evolution]
            threshold_std = np.std(thresholds)
            threshold_cv = threshold_std / np.mean(thresholds) if np.mean(thresholds) != 0 else float('inf')
            
            print(f"\n🎯 Threshold Stability Analysis:")
            print(f"   Mean Threshold: {np.mean(thresholds):.6f}")
            print(f"   Threshold Std: {threshold_std:.6f}")
            print(f"   Coefficient of Variation: {threshold_cv:.6f}")
            
            stability_rating = "EXCELLENT" if threshold_cv < 0.01 else "GOOD" if threshold_cv < 0.05 else "POOR"
            print(f"   Stability Rating: {stability_rating}")
        
        # Crea visualizzazioni dettagliate
        print(f"\n📊 Generating comprehensive visualizations...")
        self.create_visualizations()
        
        # Crea report dettagliato
        print(f"\n📋 Generating detailed DataFrame report...")
        df_report = self.create_detailed_report()
        
        # Final recommendations and configuration
        print(f"\n🎯 FINAL RECOMMENDATIONS & CONFIGURATION")
        print("="*80)
        
        if evolution:
            final_epoch_data = evolution[-1]
            avg_discrimination = np.mean([e['discrimination'] for e in evolution])
            avg_accuracy = np.mean([e['threshold_accuracy'] for e in evolution])
            final_threshold = final_epoch_data['optimal_threshold']
            
            # Overall effectiveness assessment
            if avg_accuracy >= 0.8 and avg_discrimination >= 0.01:
                effectiveness = "🟢 EXCELLENT - Highly recommended for production"
            elif avg_accuracy >= 0.7 and avg_discrimination >= 0.005:
                effectiveness = "🟡 GOOD - Suitable with monitoring"
            elif avg_accuracy >= 0.6:
                effectiveness = "🟠 MODERATE - Consider improvements"
            else:
                effectiveness = "🔴 POOR - Not recommended"
            
            print(f"Overall Effectiveness: {effectiveness}")
            print(f"\n📋 Recommended Configuration:")
            print(f"   similarity_method = 'cosine_original'")
            print(f"   similarity_threshold = {final_threshold:.6f}")
            print(f"   expected_accuracy = {final_epoch_data['threshold_accuracy']:.4f}")
            print(f"   discrimination_power = {final_epoch_data['discrimination']:.6f}")
            
            # Practical deployment guidance
            print(f"\n🚀 Deployment Guidance:")
            
            if threshold_cv < 0.01:
                print("   ✅ Threshold is stable - safe for static configuration")
            else:
                print("   ⚠️  Consider adaptive threshold based on recent epochs")
            
            if avg_accuracy >= 0.8:
                print("   ✅ High accuracy - can be used as primary detection method")
            elif avg_accuracy >= 0.7:
                print("   ⚠️  Moderate accuracy - combine with other detection methods")
            else:
                print("   ❌ Low accuracy - use only as supplementary method")
            
            if np.std([e['threshold_accuracy'] for e in evolution]) < 0.05:
                print("   ✅ Consistent performance across epochs")
            else:
                print("   ⚠️  Performance varies significantly - monitor closely")
        
        print(f"\n📊 Experiment Summary:")
        print(f"   Total epochs analyzed: {len(self.results_by_epoch)}")
        print(f"   Workers per round: {self.workers_per_round}")
        print(f"   Total workers: {self.num_workers}")
        print(f"   Poisoned workers: {self.num_poisoned} {self.poisoned_indices}")
        print(f"   Success rate: {len(self.results_by_epoch)}/{self.num_epochs} epochs ({len(self.results_by_epoch)/self.num_epochs*100:.1f}%)")
        
        print(f"\n💾 Output Files Generated:")
        print(f"   📊 cosine_similarity_analysis.png (comprehensive visualizations)")
        print(f"   📋 cosine_similarity_report.csv (detailed temporal data)")
        print(f"   📄 cosine_similarity_temporal_analysis.pdf (professional 3-page report)")
        print(f"   🔍 Use the PDF for presentations and detailed analysis")
        
        print("\n" + "="*100)
    
    def analyze_distribution_overlap(self, clean_sims, poisoned_sims):
        """Analizza l'overlap tra le distribuzioni clean e poisoned."""
        try:
            # Calcola statistiche descrittive
            clean_min, clean_max = min(clean_sims), max(clean_sims)
            poisoned_min, poisoned_max = min(poisoned_sims), max(poisoned_sims)
            
            # Calcola overlap range
            overlap_start = max(clean_min, poisoned_min)
            overlap_end = min(clean_max, poisoned_max)
            
            if overlap_end > overlap_start:
                overlap_range = overlap_end - overlap_start
                clean_range = clean_max - clean_min
                poisoned_range = poisoned_max - poisoned_min
                
                # Percentuale di overlap
                clean_overlap_pct = overlap_range / (clean_range + 1e-8) * 100
                poisoned_overlap_pct = overlap_range / (poisoned_range + 1e-8) * 100
                
                # Separazione delle mediane
                median_separation = abs(np.median(clean_sims) - np.median(poisoned_sims))
                
                return {
                    'has_overlap': True,
                    'overlap_range': overlap_range,
                    'clean_overlap_pct': clean_overlap_pct,
                    'poisoned_overlap_pct': poisoned_overlap_pct,
                    'median_separation': median_separation,
                    'total_separation': clean_max - poisoned_min if clean_min > poisoned_max else poisoned_max - clean_min
                }
            else:
                return {
                    'has_overlap': False,
                    'overlap_range': 0,
                    'clean_overlap_pct': 0,
                    'poisoned_overlap_pct': 0,
                    'median_separation': abs(np.median(clean_sims) - np.median(poisoned_sims)),
                    'total_separation': abs(clean_min - poisoned_max) if clean_min > poisoned_max else abs(poisoned_min - clean_max)
                }
        except Exception as e:
            print(f"Error in overlap analysis: {e}")
            return {'has_overlap': True, 'overlap_range': float('inf')}
    
    def find_optimal_threshold(self, similarities, labels, clean_avg, poisoned_avg):
        """Trova il threshold ottimale per massimizzare l'accuratezza di detection."""
        if len(similarities) == 0 or len(labels) == 0:
            return (clean_avg + poisoned_avg) / 2, {'accuracy': 0, 'precision': 0, 'recall': 0}
        
        similarities = np.array(similarities)
        labels = np.array(labels)  # True=clean, False=poisoned
        
        # Test diversi threshold
        min_sim, max_sim = np.min(similarities), np.max(similarities)
        thresholds = np.linspace(min_sim, max_sim, 100)
        
        best_accuracy = 0
        best_threshold = (clean_avg + poisoned_avg) / 2
        best_stats = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        for threshold in thresholds:
            # Predizioni: similarity >= threshold -> clean (True), < threshold -> poisoned (False)
            predictions = similarities >= threshold
            
            # Calcola metriche
            tp = np.sum((predictions == True) & (labels == True))   # Clean predetto come clean
            tn = np.sum((predictions == False) & (labels == False)) # Poisoned predetto come poisoned
            fp = np.sum((predictions == True) & (labels == False))  # Poisoned predetto come clean
            fn = np.sum((predictions == False) & (labels == True))  # Clean predetto come poisoned
            
            if (tp + tn + fp + fn) > 0:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_stats = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                    }
        
        return best_threshold, best_stats
    
    def create_visualizations(self):
        """Crea visualizzazioni dettagliate dei risultati."""
        if not self.similarity_evolution['cosine_original']:
            print("No data available for visualization")
            return
        
        # Setup per multiple subplot
        fig = plt.figure(figsize=(20, 15))
        
        evolution_data = self.similarity_evolution['cosine_original']
        epochs = [d['epoch'] for d in evolution_data]
        
        # 1. Evolution of discrimination over time
        ax1 = plt.subplot(2, 3, 1)
        discriminations = [d['discrimination'] for d in evolution_data]
        plt.plot(epochs, discriminations, 'b-o', linewidth=2, markersize=8)
        plt.title('Discrimination Power Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Discrimination (|Clean_avg - Poisoned_avg|)')
        plt.grid(True, alpha=0.3)
        
        # Trend line
        if len(epochs) > 1:
            z = np.polyfit(epochs, discriminations, 1)
            p = np.poly1d(z)
            plt.plot(epochs, p(epochs), "r--", alpha=0.8, 
                    label=f'Trend: {z[0]:+.6f}/epoch')
            plt.legend()
        
        # 2. Average similarities over time
        ax2 = plt.subplot(2, 3, 2)
        clean_avgs = [d['clean_avg'] for d in evolution_data]
        poisoned_avgs = [d['poisoned_avg'] for d in evolution_data]
        
        plt.plot(epochs, clean_avgs, 'g-o', label='Clean-Clean Avg', linewidth=2, markersize=6)
        plt.plot(epochs, poisoned_avgs, 'r-s', label='Clean-Poisoned Avg', linewidth=2, markersize=6)
        
        plt.title('Average Similarities Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Optimal threshold evolution
        ax3 = plt.subplot(2, 3, 3)
        thresholds = [d['optimal_threshold'] for d in evolution_data]
        accuracies = [d['threshold_accuracy'] for d in evolution_data]
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(epochs, thresholds, 'purple', marker='o', linewidth=2, label='Optimal Threshold')
        line2 = ax3_twin.plot(epochs, accuracies, 'orange', marker='s', linewidth=2, label='Threshold Accuracy')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Optimal Threshold', color='purple')
        ax3_twin.set_ylabel('Accuracy', color='orange')
        ax3.set_title('Threshold Evolution & Accuracy', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribution analysis for latest epoch
        ax4 = plt.subplot(2, 3, 4)
        if self.detailed_similarities:
            latest_data = self.detailed_similarities[-1]
            clean_sims = latest_data['clean_clean_sims']
            poisoned_sims = latest_data['clean_poisoned_sims']
            threshold = latest_data['optimal_threshold']
            
            # Histograms
            plt.hist(clean_sims, bins=20, alpha=0.7, color='green', 
                    label=f'Clean-Clean (n={len(clean_sims)})', density=True)
            plt.hist(poisoned_sims, bins=20, alpha=0.7, color='red', 
                    label=f'Clean-Poisoned (n={len(poisoned_sims)})', density=True)
            
            # Threshold line
            plt.axvline(threshold, color='purple', linestyle='--', linewidth=2,
                       label=f'Optimal Threshold: {threshold:.4f}')
            
            plt.title(f'Distribution Analysis - Epoch {len(epochs)}', fontsize=14, fontweight='bold')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Separability ratio over time
        ax5 = plt.subplot(2, 3, 5)
        separability_ratios = [d['separability_ratio'] for d in evolution_data]
        plt.plot(epochs, separability_ratios, 'm-^', linewidth=2, markersize=8)
        plt.title('Separability Ratio Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Discrimination / (Clean_std + Poisoned_std)')
        plt.grid(True, alpha=0.3)
        
        # Higher is better indicator
        avg_separability = np.mean(separability_ratios)
        plt.axhline(avg_separability, color='orange', linestyle=':',
                   label=f'Average: {avg_separability:.3f}')
        plt.legend()
        
        # 6. Box plot comparison for all epochs
        ax6 = plt.subplot(2, 3, 6)
        all_clean_sims = []
        all_poisoned_sims = []
        
        for epoch_data in self.detailed_similarities:
            all_clean_sims.extend(epoch_data['clean_clean_sims'])
            all_poisoned_sims.extend(epoch_data['clean_poisoned_sims'])
        
        plt.boxplot([all_clean_sims, all_poisoned_sims], 
                   labels=['Clean-Clean', 'Clean-Poisoned'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        plt.title('Overall Distribution Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Cosine Similarity')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cosine_similarity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Visualization saved as 'cosine_similarity_analysis.png'")
    
    def create_detailed_report(self):
        """Crea un report dettagliato in formato DataFrame."""
        if not self.similarity_evolution['cosine_original']:
            print("No data available for detailed report")
            return None
        
        evolution_data = self.similarity_evolution['cosine_original']
        
        # Crea DataFrame per l'analisi temporale
        df_temporal = pd.DataFrame([
            {
                'Epoch': d['epoch'] + 1,
                'Clean_Avg': d['clean_avg'],
                'Clean_Std': d['clean_std'],
                'Poisoned_Avg': d['poisoned_avg'],
                'Poisoned_Std': d['poisoned_std'],
                'Discrimination': d['discrimination'],
                'Separability_Ratio': d['separability_ratio'],
                'Optimal_Threshold': d['optimal_threshold'],
                'Threshold_Accuracy': d['threshold_accuracy']
            }
            for d in evolution_data
        ])
        
        print("\n" + "="*100)
        print("DETAILED TEMPORAL ANALYSIS REPORT")
        print("="*100)
        
        print(f"\n📋 Temporal Evolution Summary:")
        print(df_temporal.round(6).to_string(index=False))
        
        # Statistiche riassuntive
        print(f"\n📊 Summary Statistics:")
        summary_stats = df_temporal.describe().round(6)
        print(summary_stats)
        
        # Analisi trend
        print(f"\n📈 Trend Analysis:")
        for column in ['Discrimination', 'Separability_Ratio', 'Threshold_Accuracy']:
            if len(df_temporal) > 1:
                correlation = df_temporal['Epoch'].corr(df_temporal[column])
                trend_slope = np.polyfit(df_temporal['Epoch'], df_temporal[column], 1)[0]
                print(f"  {column:20}: Correlation with time = {correlation:+.4f}, Slope = {trend_slope:+.6f}")
        
        # Stabilità del threshold
        threshold_std = df_temporal['Optimal_Threshold'].std()
        threshold_range = df_temporal['Optimal_Threshold'].max() - df_temporal['Optimal_Threshold'].min()
        
        print(f"\n🎯 Threshold Stability Analysis:")
        print(f"  Threshold Standard Deviation: {threshold_std:.6f}")
        print(f"  Threshold Range (Max-Min): {threshold_range:.6f}")
        print(f"  Threshold Coefficient of Variation: {threshold_std/df_temporal['Optimal_Threshold'].mean():.6f}")
        
        # Efficacia di detection
        avg_accuracy = df_temporal['Threshold_Accuracy'].mean()
        min_accuracy = df_temporal['Threshold_Accuracy'].min()
        max_accuracy = df_temporal['Threshold_Accuracy'].max()
        
        print(f"\n🎯 Detection Effectiveness:")
        print(f"  Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"  Min Accuracy: {min_accuracy:.4f} ({min_accuracy*100:.2f}%)")
        print(f"  Max Accuracy: {max_accuracy:.4f} ({max_accuracy*100:.2f}%)")
        print(f"  Accuracy Stability (std): {df_temporal['Threshold_Accuracy'].std():.4f}")
        
        # Raccomandazioni finali
        print(f"\n🔍 FINAL RECOMMENDATIONS:")
        print("="*60)
        
        final_discrimination = df_temporal['Discrimination'].iloc[-1]
        final_accuracy = df_temporal['Threshold_Accuracy'].iloc[-1]
        final_threshold = df_temporal['Optimal_Threshold'].iloc[-1]
        
        if final_accuracy >= 0.8:
            effectiveness = "EXCELLENT"
        elif final_accuracy >= 0.7:
            effectiveness = "GOOD"
        elif final_accuracy >= 0.6:
            effectiveness = "MODERATE"
        else:
            effectiveness = "POOR"
        
        print(f"✅ Detection Effectiveness: {effectiveness}")
        print(f"✅ Recommended Threshold: {final_threshold:.6f}")
        print(f"✅ Expected Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"✅ Discrimination Power: {final_discrimination:.6f}")
        
        if threshold_std < 0.01:
            print("✅ Threshold is STABLE across epochs - good for production")
        elif threshold_std < 0.05:
            print("⚠️  Threshold shows MODERATE variability - monitor in production")
        else:
            print("❌ Threshold is UNSTABLE - consider alternative approaches")
        
        # Salva il report
        df_temporal.to_csv('cosine_similarity_report.csv', index=False)
        print(f"\n💾 Detailed report saved as 'cosine_similarity_report.csv'")
        
        # Crea anche il PDF professionale
        print(f"\n📊 Creating professional PDF report...")
        pdf_file = self.create_pdf_temporal_analysis(df_temporal)
        if pdf_file:
            print(f"✅ PDF report successfully created: {pdf_file}")
        else:
            print(f"❌ PDF creation failed - using fallback PNG visualization")
        
        return df_temporal
    
    def create_pdf_temporal_analysis(self, df_temporal=None):
        """Crea un grafico PDF professionale dall'analisi temporale."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import seaborn as sns
            
            # Se non viene fornito il DataFrame, lo crea dai dati esistenti
            if df_temporal is None:
                if not self.similarity_evolution['cosine_original']:
                    print("No temporal data available for PDF creation")
                    return None
                
                evolution_data = self.similarity_evolution['cosine_original']
                df_temporal = pd.DataFrame([
                    {
                        'Epoch': d['epoch'] + 1,
                        'Clean_Avg': d['clean_avg'],
                        'Clean_Std': d['clean_std'], 
                        'Poisoned_Avg': d['poisoned_avg'],
                        'Poisoned_Std': d['poisoned_std'],
                        'Discrimination': d['discrimination'],
                        'Separability_Ratio': d['separability_ratio'],
                        'Optimal_Threshold': d['optimal_threshold'],
                        'Threshold_Accuracy': d['threshold_accuracy']
                    }
                    for d in evolution_data
                ])
            
            # Setup stile professionale
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            sns.set_palette("husl")
            
            # Crea il file PDF
            pdf_filename = 'cosine_similarity_temporal_analysis.pdf'
            
            with PdfPages(pdf_filename) as pdf:
                
                # ===================== PAGINA 1: Overview Metrics =====================
                fig = plt.figure(figsize=(16, 12))
                fig.suptitle('Cosine Similarity Temporal Analysis - Overview', fontsize=20, fontweight='bold', y=0.95)
                
                # Plot 1: Discrimination over time
                ax1 = plt.subplot(2, 2, 1)
                epochs = df_temporal['Epoch'].values
                discriminations = df_temporal['Discrimination'].values
                
                plt.plot(epochs, discriminations, 'o-', linewidth=3, markersize=8, 
                        color='#2E86AB', label='Discrimination Power')
                
                # Trend line
                if len(epochs) > 1:
                    z = np.polyfit(epochs, discriminations, 1)
                    p = np.poly1d(z)
                    plt.plot(epochs, p(epochs), "--", alpha=0.8, linewidth=2, 
                            color='#A23B72', label=f'Trend: {z[0]:+.6f}/epoch')
                
                plt.title('Discrimination Power Evolution', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Discrimination (|Clean - Poisoned|)', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Plot 2: Threshold Accuracy over time  
                ax2 = plt.subplot(2, 2, 2)
                accuracies = df_temporal['Threshold_Accuracy'].values
                
                plt.plot(epochs, accuracies * 100, 's-', linewidth=3, markersize=8,
                        color='#F18F01', label='Detection Accuracy')
                
                # Aggiungi bande di qualità
                plt.axhspan(80, 100, alpha=0.2, color='green', label='Excellent (≥80%)')
                plt.axhspan(70, 80, alpha=0.2, color='yellow', label='Good (70-80%)')
                plt.axhspan(0, 70, alpha=0.2, color='red', label='Poor (<70%)')
                
                plt.title('Detection Accuracy Evolution', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Accuracy (%)', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 105)
                
                # Plot 3: Similarity values comparison
                ax3 = plt.subplot(2, 2, 3)
                clean_avgs = df_temporal['Clean_Avg'].values
                poisoned_avgs = df_temporal['Poisoned_Avg'].values
                clean_stds = df_temporal['Clean_Std'].values
                poisoned_stds = df_temporal['Poisoned_Std'].values
                
                # Plot con error bars
                plt.errorbar(epochs, clean_avgs, yerr=clean_stds, 
                           fmt='o-', linewidth=3, markersize=6, capsize=5,
                           color='#4CAF50', label='Clean-Clean Similarity')
                plt.errorbar(epochs, poisoned_avgs, yerr=poisoned_stds,
                           fmt='s-', linewidth=3, markersize=6, capsize=5, 
                           color='#F44336', label='Clean-Poisoned Similarity')
                
                plt.title('Average Similarities with Standard Deviation', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Cosine Similarity', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Plot 4: Threshold evolution
                ax4 = plt.subplot(2, 2, 4)
                thresholds = df_temporal['Optimal_Threshold'].values
                
                plt.plot(epochs, thresholds, '^-', linewidth=3, markersize=8,
                        color='#9C27B0', label='Optimal Threshold')
                
                # Aggiungi statistiche
                mean_threshold = np.mean(thresholds)
                std_threshold = np.std(thresholds)
                
                plt.axhline(mean_threshold, color='orange', linestyle=':', linewidth=2,
                           label=f'Mean: {mean_threshold:.4f}')
                plt.fill_between(epochs, mean_threshold - std_threshold, mean_threshold + std_threshold,
                               alpha=0.2, color='orange', label=f'±1 Std: {std_threshold:.4f}')
                
                plt.title('Optimal Threshold Evolution', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Threshold Value', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # ===================== PAGINA 2: Advanced Analysis =====================
                fig = plt.figure(figsize=(16, 12))
                fig.suptitle('Advanced Metrics & Statistical Analysis', fontsize=20, fontweight='bold', y=0.95)
                
                # Plot 1: Separability Ratio
                ax1 = plt.subplot(2, 2, 1)
                sep_ratios = df_temporal['Separability_Ratio'].values
                
                plt.plot(epochs, sep_ratios, 'D-', linewidth=3, markersize=8,
                        color='#E91E63', label='Separability Ratio')
                
                # Aggiungi trend
                if len(epochs) > 1:
                    z = np.polyfit(epochs, sep_ratios, 1)
                    p = np.poly1d(z)
                    plt.plot(epochs, p(epochs), "--", alpha=0.8, linewidth=2,
                            color='#795548', label=f'Trend: {z[0]:+.6f}/epoch')
                
                plt.title('Separability Ratio (Discrimination/Std)', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Separability Ratio', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Plot 2: Correlation Matrix
                ax2 = plt.subplot(2, 2, 2)
                metrics_for_corr = df_temporal[['Discrimination', 'Separability_Ratio', 
                                              'Threshold_Accuracy', 'Optimal_Threshold']].corr()
                
                im = ax2.imshow(metrics_for_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # Aggiungi testo delle correlazioni
                for i in range(len(metrics_for_corr)):
                    for j in range(len(metrics_for_corr)):
                        text = ax2.text(j, i, f'{metrics_for_corr.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontweight='bold')
                
                ax2.set_xticks(range(len(metrics_for_corr.columns)))
                ax2.set_yticks(range(len(metrics_for_corr.columns)))
                ax2.set_xticklabels(metrics_for_corr.columns, rotation=45, ha='right')
                ax2.set_yticklabels(metrics_for_corr.columns)
                ax2.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Correlation Coefficient', fontsize=12)
                
                # Plot 3: Distribution Evolution (Box plot style)
                ax3 = plt.subplot(2, 1, 2)
                
                # Crea dati per box plot temporale  
                data_clean = []
                data_poisoned = []
                epoch_labels = []
                
                # Usa i dati dettagliati se disponibili
                if hasattr(self, 'detailed_similarities') and self.detailed_similarities:
                    for i, epoch_data in enumerate(self.detailed_similarities[::max(1, len(self.detailed_similarities)//8)]):  # Campiona per evitare troppi box
                        data_clean.extend([(i, sim) for sim in epoch_data['clean_clean_sims']])
                        data_poisoned.extend([(i, sim) for sim in epoch_data['clean_poisoned_sims']])
                        epoch_labels.append(f"E{epoch_data['epoch']+1}")
                
                if data_clean and data_poisoned:
                    # Converti in formato per plot
                    clean_epochs, clean_vals = zip(*data_clean)
                    poisoned_epochs, poisoned_vals = zip(*data_poisoned) 
                    
                    # Crea violin plot
                    positions_clean = [i - 0.2 for i in range(len(epoch_labels))]
                    positions_poisoned = [i + 0.2 for i in range(len(epoch_labels))]
                    
                    # Organizza dati per violin plot
                    clean_data_by_epoch = []
                    poisoned_data_by_epoch = []
                    
                    for i in range(len(epoch_labels)):
                        clean_epoch_data = [val for j, val in enumerate(clean_vals) if clean_epochs[j] == i]
                        poisoned_epoch_data = [val for j, val in enumerate(poisoned_vals) if poisoned_epochs[j] == i]
                        clean_data_by_epoch.append(clean_epoch_data)
                        poisoned_data_by_epoch.append(poisoned_epoch_data)
                    
                    # Plot violin plots
                    violin1 = ax3.violinplot(clean_data_by_epoch, positions=positions_clean, 
                                           widths=0.3, showmeans=True)
                    violin2 = ax3.violinplot(poisoned_data_by_epoch, positions=positions_poisoned,
                                           widths=0.3, showmeans=True)
                    
                    # Colora i violin plots
                    for pc in violin1['bodies']:
                        pc.set_facecolor('#4CAF50')
                        pc.set_alpha(0.7)
                    for pc in violin2['bodies']:
                        pc.set_facecolor('#F44336')
                        pc.set_alpha(0.7)
                    
                    ax3.set_xticks(range(len(epoch_labels)))
                    ax3.set_xticklabels(epoch_labels)
                    ax3.set_xlabel('Epochs (sampled)', fontsize=12)
                    ax3.set_ylabel('Cosine Similarity Distribution', fontsize=12)
                    ax3.set_title('Similarity Distributions Evolution Over Time', fontsize=14, fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                    
                    # Legenda custom
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor='#4CAF50', alpha=0.7, label='Clean-Clean'),
                                     Patch(facecolor='#F44336', alpha=0.7, label='Clean-Poisoned')]
                    ax3.legend(handles=legend_elements, fontsize=10)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # ===================== PAGINA 3: Summary Statistics =====================
                fig = plt.figure(figsize=(16, 12))
                fig.suptitle('Summary Statistics & Recommendations', fontsize=20, fontweight='bold', y=0.95)
                
                # Calcola statistiche riassuntive
                summary_stats = df_temporal.describe()
                
                # Crea tabella delle statistiche
                ax1 = plt.subplot(2, 1, 1)
                ax1.axis('tight')
                ax1.axis('off')
                
                # Prepara i dati della tabella
                stats_to_show = ['mean', 'std', 'min', 'max']
                cols_to_show = ['Discrimination', 'Threshold_Accuracy', 'Separability_Ratio', 'Optimal_Threshold']
                
                table_data = []
                for stat in stats_to_show:
                    row = [stat.upper()]
                    for col in cols_to_show:
                        if col == 'Threshold_Accuracy':
                            row.append(f"{summary_stats.loc[stat, col]*100:.2f}%")
                        else:
                            row.append(f"{summary_stats.loc[stat, col]:.6f}")
                    table_data.append(row)
                
                headers = ['Statistic'] + cols_to_show
                table = ax1.table(cellText=table_data, colLabels=headers,
                                cellLoc='center', loc='center',
                                bbox=[0, 0.5, 1, 0.4])
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 2)
                
                # Stile della tabella
                for i in range(len(headers)):
                    table[(0, i)].set_facecolor('#E3F2FD')
                    table[(0, i)].set_text_props(weight='bold')
                
                ax1.set_title('Summary Statistics Table', fontsize=16, fontweight='bold', pad=20)
                
                # Plot finale: Trend summary
                ax2 = plt.subplot(2, 1, 2)
                
                # Calcola trend per ciascuna metrica
                metrics = ['Discrimination', 'Threshold_Accuracy', 'Separability_Ratio', 'Optimal_Threshold']
                trends = []
                
                for metric in metrics:
                    if len(df_temporal) > 1:
                        slope = np.polyfit(df_temporal['Epoch'], df_temporal[metric], 1)[0]
                        trends.append(slope)
                    else:
                        trends.append(0)
                
                # Crea bar plot dei trend
                colors = ['#2E86AB' if t >= 0 else '#A23B72' for t in trends]
                bars = ax2.bar(range(len(metrics)), trends, color=colors, alpha=0.7)
                
                # Aggiungi etichette
                ax2.set_xticks(range(len(metrics)))
                ax2.set_xticklabels(metrics, rotation=45, ha='right')
                ax2.set_ylabel('Trend Slope (per epoch)', fontsize=12)
                ax2.set_title('Trend Analysis Summary', fontsize=16, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Aggiungi valori sulle barre
                for i, (bar, trend) in enumerate(zip(bars, trends)):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 * max(abs(min(trends)), abs(max(trends))) if height >= 0 else -0.1 * max(abs(min(trends)), abs(max(trends)))),
                           f'{trend:+.6f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Aggiungi metadati al PDF
                d = pdf.infodict()
                d['Title'] = 'Cosine Similarity Temporal Analysis'
                d['Author'] = 'DAGTrustFL Analysis System'
                d['Subject'] = 'Federated Learning Parameter Similarity Analysis'
                d['Keywords'] = 'Cosine Similarity, Federated Learning, Poisoning Detection'
                d['CreationDate'] = pd.Timestamp.now()
            
            print(f"\n📊 Professional PDF report created: '{pdf_filename}'")
            print(f"   📄 3 pages with comprehensive temporal analysis")
            print(f"   📈 Overview metrics, advanced analysis, and summary statistics")
            print(f"   📋 Correlation analysis and distribution evolution")
            
            return pdf_filename
            
        except ImportError as e:
            print(f"Missing required libraries for PDF creation: {e}")
            print("Please install: pip install matplotlib seaborn")
            return None
        except Exception as e:
            print(f"Error creating PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function per eseguire il test esteso."""
    
    # Configura il logger per ridurre il noise
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("🚀 Starting Enhanced Cosine Similarity Analysis")
    print("="*80)
    print("Configuration:")
    print("  - 20 epochs for comprehensive temporal analysis")
    print("  - Enhanced threshold detection with ROC analysis")
    print("  - Comprehensive visualizations and reporting")
    print("  - Detailed statistical analysis")
    print("="*80)
    
    # Crea e esegui il tester con configurazione potenziata
    tester = FL_SimilarityTester(
        num_workers=18,      # Più workers per test robusto
        num_poisoned=6,      # Più attackers per test realistico
        num_epochs=20,       # 20 epoche come richiesto
        workers_per_round=12 # Più workers per round
    )
    
    try:
        tester.run_fl_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
