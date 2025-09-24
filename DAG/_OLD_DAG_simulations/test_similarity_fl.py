#!/usr/bin/env python3
"""
Test script integrato con il sistema DAGTrustFL per testare la discriminazione
tra parametri di modelli puliti e parametri di modelli con label flipping attack.

Questo script esegue un esperimento di federated learning con label flipping
e analizza la capacità dei diversi metodi di similitudine di distinguere
tra parametri buoni e poisoned.
"""

import sys
import os

# Add FL and DAG directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FL'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from collections import OrderedDict
import parameter_utils

# Import FL components
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import replace_1_with_9
from federated_learning.worker_selection import RandomSelectionStrategy
from client import Client
from loguru import logger

def create_test_clients(num_workers=10, num_poisoned=2, replacement_method=replace_1_with_9):
    """
    Crea clients per il test, alcuni puliti e alcuni poisoned.
    
    Returns:
        tuple: (clean_clients, poisoned_clients, poisoned_indices)
    """
    # Setup minimal args for testing
    args = Arguments(logger)
    args.set_num_workers(num_workers)
    args.set_num_poisoned_workers(num_poisoned)
    
    # Load data
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)
    
    # Distribute data
    distributed_train_dataset = distribute_batches_equally(train_data_loader, num_workers)
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
    
    # Identify poisoned workers
    poisoned_indices = list(range(num_poisoned))  # First N workers are poisoned
    
    # Poison data for specified workers
    poisoned_distributed_dataset = poison_data(
        logger, distributed_train_dataset, num_workers, poisoned_indices, replacement_method
    )
    
    # Create data loaders
    train_data_loaders = generate_data_loaders_from_distributed_dataset(
        poisoned_distributed_dataset, args.get_batch_size()
    )
    
    # Create clients
    clients = []
    for idx in range(num_workers):
        client = Client(args, idx, train_data_loaders[idx], test_data_loader)
        clients.append(client)
    
    # Separate clean and poisoned clients
    clean_clients = [clients[i] for i in range(num_workers) if i not in poisoned_indices]
    poisoned_clients = [clients[i] for i in poisoned_indices]
    
    return clean_clients, poisoned_clients, poisoned_indices

def train_clients_one_epoch(clients):
    """
    Allena tutti i clients per una epoch e restituisce i loro parametri.
    
    Returns:
        list: Lista di parametri dei modelli allenati
    """
    parameters = []
    for client in clients:
        client.train(1)  # Train for 1 epoch
        parameters.append(client.get_nn_parameters())
    
    return parameters

def analyze_parameter_similarities(clean_params, poisoned_params, verbose=True):
    """
    Analizza le similitudini tra parametri puliti e poisoned usando tutti i metodi.
    
    Args:
        clean_params: Lista di parametri dei modelli puliti
        poisoned_params: Lista di parametri dei modelli poisoned
        verbose: Se stampare i risultati dettagliati
        
    Returns:
        dict: Risultati dell'analisi
    """
    results = {
        'clean_vs_clean': [],
        'clean_vs_poisoned': [],
        'poisoned_vs_poisoned': []
    }
    
    methods = ['cosine_original', 'cosine_improved', 'euclidean', 'statistical', 'combined', 'cosine_enhanced']
    
    if verbose:
        print("=== ANALISI SIMILITUDINI CON LABEL FLIPPING ATTACK ===")
        print(f"Numero modelli puliti: {len(clean_params)}")
        print(f"Numero modelli poisoned: {len(poisoned_params)}")
        print()
    
    # Clean vs Clean
    for i in range(len(clean_params)):
        for j in range(i+1, len(clean_params)):
            comparison = {}
            for method in methods:
                if method == 'cosine_original':
                    sim = parameter_utils.cosine_similarity_parameters(clean_params[i], clean_params[j])
                elif method == 'cosine_improved':
                    sim = parameter_utils.cosine_similarity_parameters_improved(clean_params[i], clean_params[j])
                else:
                    sim = parameter_utils.enhanced_similarity_score(clean_params[i], clean_params[j], method=method)
                comparison[method] = sim
            results['clean_vs_clean'].append(comparison)
    
    # Clean vs Poisoned
    for clean_param in clean_params:
        for poisoned_param in poisoned_params:
            comparison = {}
            for method in methods:
                if method == 'cosine_original':
                    sim = parameter_utils.cosine_similarity_parameters(clean_param, poisoned_param)
                elif method == 'cosine_improved':
                    sim = parameter_utils.cosine_similarity_parameters_improved(clean_param, poisoned_param)
                else:
                    sim = parameter_utils.enhanced_similarity_score(clean_param, poisoned_param, method=method)
                comparison[method] = sim
            results['clean_vs_poisoned'].append(comparison)
    
    # Poisoned vs Poisoned
    for i in range(len(poisoned_params)):
        for j in range(i+1, len(poisoned_params)):
            comparison = {}
            for method in methods:
                if method == 'cosine_original':
                    sim = parameter_utils.cosine_similarity_parameters(poisoned_params[i], poisoned_params[j])
                elif method == 'cosine_improved':
                    sim = parameter_utils.cosine_similarity_parameters_improved(poisoned_params[i], poisoned_params[j])
                else:
                    sim = parameter_utils.enhanced_similarity_score(poisoned_params[i], poisoned_params[j], method=method)
                comparison[method] = sim
            results['poisoned_vs_poisoned'].append(comparison)
    
    if verbose:
        print_similarity_analysis(results, methods)
    
    return results

def print_similarity_analysis(results, methods):
    """Stampa l'analisi delle similitudini in formato leggibile."""
    
    for method in methods:
        print(f"\n=== METODO: {method.upper()} ===")
        
        # Calcola statistiche
        clean_clean_scores = [comp[method] for comp in results['clean_vs_clean']]
        clean_poisoned_scores = [comp[method] for comp in results['clean_vs_poisoned']]
        poisoned_poisoned_scores = [comp[method] for comp in results['poisoned_vs_poisoned']]
        
        print(f"Clean vs Clean:")
        if clean_clean_scores:
            print(f"  Range: {min(clean_clean_scores):.6f} - {max(clean_clean_scores):.6f}")
            print(f"  Media: {sum(clean_clean_scores)/len(clean_clean_scores):.6f}")
        
        print(f"Clean vs Poisoned:")
        if clean_poisoned_scores:
            print(f"  Range: {min(clean_poisoned_scores):.6f} - {max(clean_poisoned_scores):.6f}")
            print(f"  Media: {sum(clean_poisoned_scores)/len(clean_poisoned_scores):.6f}")
        
        print(f"Poisoned vs Poisoned:")
        if poisoned_poisoned_scores:
            print(f"  Range: {min(poisoned_poisoned_scores):.6f} - {max(poisoned_poisoned_scores):.6f}")
            print(f"  Media: {sum(poisoned_poisoned_scores)/len(poisoned_poisoned_scores):.6f}")
        
        # Calcola discriminazione
        if clean_clean_scores and clean_poisoned_scores:
            clean_avg = sum(clean_clean_scores) / len(clean_clean_scores)
            poisoned_avg = sum(clean_poisoned_scores) / len(clean_poisoned_scores)
            discrimination = abs(clean_avg - poisoned_avg)
            print(f"  Discriminazione: {discrimination:.6f}")

def suggest_optimal_thresholds(results, methods):
    """Suggerisce soglie ottimali per ciascun metodo."""
    
    print("\n" + "="*60)
    print("RACCOMANDAZIONI SOGLIE OTTIMALI")
    print("="*60)
    
    for method in methods:
        clean_clean_scores = [comp[method] for comp in results['clean_vs_clean']]
        clean_poisoned_scores = [comp[method] for comp in results['clean_vs_poisoned']]
        
        if not clean_clean_scores or not clean_poisoned_scores:
            continue
            
        # Ordina i punteggi
        clean_clean_sorted = sorted(clean_clean_scores)
        clean_poisoned_sorted = sorted(clean_poisoned_scores)
        
        # Soglia conservativa: 5° percentile dei clean_clean
        conservative_threshold = clean_clean_sorted[max(0, int(len(clean_clean_sorted) * 0.05))]
        
        # Soglia bilanciata: punto medio tra le medie
        clean_avg = sum(clean_clean_scores) / len(clean_clean_scores)
        poisoned_avg = sum(clean_poisoned_scores) / len(clean_poisoned_scores)
        balanced_threshold = (clean_avg + poisoned_avg) / 2
        
        # Calcola accuracy con soglia bilanciata
        true_positives = sum(1 for score in clean_poisoned_scores if score < balanced_threshold)
        false_positives = sum(1 for score in clean_clean_scores if score < balanced_threshold)
        
        tpr = true_positives / len(clean_poisoned_scores) if clean_poisoned_scores else 0
        fpr = false_positives / len(clean_clean_scores) if clean_clean_scores else 0
        
        print(f"\n{method.upper()}:")
        print(f"  Soglia conservativa: {conservative_threshold:.6f}")
        print(f"  Soglia bilanciata:   {balanced_threshold:.6f}")
        print(f"  TPR (detection):     {tpr:.3f}")
        print(f"  FPR (falsi pos.):    {fpr:.3f}")

def main():
    """Main function per testare il sistema completo."""
    
    print("Inizializzazione test con label flipping attack...")
    print("="*60)
    
    # Crea clients con label flipping
    clean_clients, poisoned_clients, poisoned_indices = create_test_clients(
        num_workers=8, 
        num_poisoned=3, 
        replacement_method=replace_1_with_9
    )
    
    print(f"Creati {len(clean_clients)} clients puliti e {len(poisoned_clients)} clients poisoned")
    print(f"Indici poisoned: {poisoned_indices}")
    print("Allenamento modelli...")
    
    # Allena i modelli
    clean_params = train_clients_one_epoch(clean_clients)
    poisoned_params = train_clients_one_epoch(poisoned_clients)
    
    print("Analisi similitudini...")
    
    # Analizza le similitudini
    results = analyze_parameter_similarities(clean_params, poisoned_params)
    
    # Suggerisci soglie ottimali
    methods = ['cosine_original', 'cosine_improved', 'euclidean', 'statistical', 'combined', 'cosine_enhanced']
    suggest_optimal_thresholds(results, methods)
    
    print("\n" + "="*60)
    print("Test completato!")
    print("="*60)

if __name__ == "__main__":
    # Configura il logger per ridurre il noise
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    
    main()
