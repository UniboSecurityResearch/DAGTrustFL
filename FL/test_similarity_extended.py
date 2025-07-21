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
    
    def __init__(self, num_workers=12, num_poisoned=4, num_epochs=5, workers_per_round=8):
        self.num_workers = num_workers
        self.num_poisoned = num_poisoned
        self.num_epochs = num_epochs
        self.workers_per_round = workers_per_round
        self.poisoned_indices = list(range(num_poisoned))  # Prime N workers poisoned
        
        # Risultati per analisi
        self.results_by_epoch = {}
        self.similarity_evolution = {method: [] for method in self.get_methods()}
        
        # Setup del sistema FL
        self.setup_fl_system()
    
    def get_methods(self):
        """Ritorna i metodi di similitudine da testare."""
        return ['cosine_original', 'cosine_improved', 'euclidean', 'statistical', 'combined', 'cosine_enhanced']
    
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
        """Analizza le similitudini per il round corrente."""
        if len(clean_params) < 2 or len(poisoned_params) < 1:
            print(f"  Skipping similarity analysis - insufficient samples")
            return None
        
        print(f"  Analyzing similarities: {len(clean_params)} clean, {len(poisoned_params)} poisoned")
        
        methods = self.get_methods()
        results = {}
        
        for method in methods:
            clean_clean_sims = []
            clean_poisoned_sims = []
            
            # Clean vs Clean
            for i in range(len(clean_params)):
                for j in range(i+1, len(clean_params)):
                    sim = self.calculate_similarity(clean_params[i], clean_params[j], method)
                    if sim is not None:
                        clean_clean_sims.append(sim)
            
            # Clean vs Poisoned
            for clean_param in clean_params:
                for poisoned_param in poisoned_params:
                    sim = self.calculate_similarity(clean_param, poisoned_param, method)
                    if sim is not None:
                        clean_poisoned_sims.append(sim)
            
            # Calcola statistiche
            if clean_clean_sims and clean_poisoned_sims:
                clean_avg = np.mean(clean_clean_sims)
                poisoned_avg = np.mean(clean_poisoned_sims)
                discrimination = abs(clean_avg - poisoned_avg)
                
                results[method] = {
                    'clean_clean_avg': clean_avg,
                    'clean_clean_std': np.std(clean_clean_sims),
                    'clean_poisoned_avg': poisoned_avg,
                    'clean_poisoned_std': np.std(clean_poisoned_sims),
                    'discrimination': discrimination,
                    'clean_clean_range': (min(clean_clean_sims), max(clean_clean_sims)),
                    'clean_poisoned_range': (min(clean_poisoned_sims), max(clean_poisoned_sims))
                }
                
                # Aggiungi all'evoluzione
                self.similarity_evolution[method].append({
                    'epoch': epoch,
                    'discrimination': discrimination,
                    'clean_avg': clean_avg,
                    'poisoned_avg': poisoned_avg
                })
        
        return results
    
    def calculate_similarity(self, params1, params2, method):
        """Calcola la similitudine usando il metodo specificato."""
        try:
            if method == 'cosine_original':
                return parameter_utils.cosine_similarity_parameters(params1, params2)
            elif method == 'cosine_improved':
                return parameter_utils.cosine_similarity_parameters_improved(params1, params2)
            else:
                return parameter_utils.enhanced_similarity_score(params1, params2, method=method)
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
        """Stampa i risultati per un singolo epoch."""
        print(f"  Similarity Results for Epoch {epoch+1}:")
        
        # Ordina per discriminazione
        sorted_methods = sorted(results.keys(), key=lambda x: results[x]['discrimination'], reverse=True)
        
        for method in sorted_methods[:3]:  # Top 3 methods
            r = results[method]
            print(f"    {method:15}: Discrimination={r['discrimination']:.6f}, "
                  f"Clean={r['clean_clean_avg']:.4f}±{r['clean_clean_std']:.4f}, "
                  f"Poisoned={r['clean_poisoned_avg']:.4f}±{r['clean_poisoned_std']:.4f}")
    
    def print_final_analysis(self):
        """Stampa l'analisi finale dell'esperimento."""
        print("\n" + "="*80)
        print("FINAL ANALYSIS - SIMILARITY EVOLUTION OVER TIME")
        print("="*80)
        
        methods = self.get_methods()
        
        # Analizza l'evoluzione nel tempo
        for method in methods:
            evolution = self.similarity_evolution[method]
            if not evolution:
                continue
                
            discriminations = [e['discrimination'] for e in evolution]
            
            print(f"\n{method.upper()}:")
            print(f"  Average discrimination: {np.mean(discriminations):.6f}")
            print(f"  Discrimination stability (std): {np.std(discriminations):.6f}")
            print(f"  Evolution: {[f'{d:.4f}' for d in discriminations]}")
            
            # Trend analysis
            if len(discriminations) > 1:
                trend = np.polyfit(range(len(discriminations)), discriminations, 1)[0]
                trend_desc = "improving" if trend > 0 else "degrading" if trend < 0 else "stable"
                print(f"  Trend: {trend_desc} ({trend:+.6f}/epoch)")
        
        # Raccomandazioni finali
        print(f"\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Trova il metodo migliore
        avg_discriminations = {}
        for method in methods:
            evolution = self.similarity_evolution[method]
            if evolution:
                avg_discriminations[method] = np.mean([e['discrimination'] for e in evolution])
        
        if avg_discriminations:
            best_method = max(avg_discriminations, key=avg_discriminations.get)
            print(f"Best performing method: {best_method.upper()}")
            print(f"Average discrimination: {avg_discriminations[best_method]:.6f}")
            
            # Calcola soglia suggerita
            best_evolution = self.similarity_evolution[best_method]
            last_epoch_data = best_evolution[-1] if best_evolution else None
            
            if last_epoch_data:
                suggested_threshold = (last_epoch_data['clean_avg'] + last_epoch_data['poisoned_avg']) / 2
                print(f"Suggested threshold: {suggested_threshold:.6f}")
                print(f"Configuration: similarity_method='{best_method}', similarity_threshold={suggested_threshold:.6f}")
        
        print(f"\nTotal epochs analyzed: {len(self.results_by_epoch)}")
        print(f"Workers per round: {self.workers_per_round}")
        print(f"Poisoned workers: {self.poisoned_indices}")

def main():
    """Main function per eseguire il test esteso."""
    
    # Configura il logger per ridurre il noise
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    # Crea e esegui il tester
    tester = FL_SimilarityTester(
        num_workers=15,      # Più workers
        num_poisoned=5,      # Più attackers
        num_epochs=6,        # Più round di FL
        workers_per_round=10 # Più workers per round
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
