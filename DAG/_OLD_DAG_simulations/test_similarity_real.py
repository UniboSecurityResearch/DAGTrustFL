#!/usr/bin/env python3
"""
Test script semplificato per testare la discriminazione con modelli reali.
Usa dati sintetici e allena modelli reali con e senza label flipping.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import parameter_utils

class SimpleDataset(Dataset):
    """Dataset sintetico per test."""
    def __init__(self, num_samples=1000, num_features=10, num_classes=10, poisoned=False, poison_rate=0.1):
        self.num_samples = num_samples
        self.poisoned = poisoned
        
        # Genera dati sintetici
        self.data = torch.randn(num_samples, num_features)
        self.targets = torch.randint(0, num_classes, (num_samples,))
        
        # Applica label flipping se poisoned
        if poisoned:
            num_poison = int(num_samples * poison_rate)
            poison_indices = torch.randperm(num_samples)[:num_poison]
            # Label flipping: 1 -> 9 (come nel tuo sistema)
            for idx in poison_indices:
                if self.targets[idx] == 1:
                    self.targets[idx] = 9
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SimpleNN(nn.Module):
    """Rete neurale semplice per test."""
    def __init__(self, input_size=10, hidden_size=20, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, dataloader, epochs=5, lr=0.01):
    """Allena un modello per un numero specificato di epoche."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model

def create_and_train_models(num_clean=5, num_poisoned=3, epochs=5):
    """
    Crea e allena modelli puliti e poisoned.
    
    Returns:
        tuple: (clean_models_params, poisoned_models_params)
    """
    clean_params = []
    poisoned_params = []
    
    print(f"Allenamento {num_clean} modelli puliti...")
    for i in range(num_clean):
        # Dataset pulito
        dataset = SimpleDataset(num_samples=1000, poisoned=False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Modello
        model = SimpleNN()
        
        # Allenamento
        trained_model = train_model(model, dataloader, epochs=epochs)
        
        # Salva parametri
        clean_params.append(trained_model.state_dict())
        print(f"  Modello pulito {i+1}/{num_clean} completato")
    
    print(f"Allenamento {num_poisoned} modelli poisoned...")
    for i in range(num_poisoned):
        # Dataset poisoned (con label flipping)
        dataset = SimpleDataset(num_samples=1000, poisoned=True, poison_rate=0.2)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Modello
        model = SimpleNN()
        
        # Allenamento
        trained_model = train_model(model, dataloader, epochs=epochs)
        
        # Salva parametri
        poisoned_params.append(trained_model.state_dict())
        print(f"  Modello poisoned {i+1}/{num_poisoned} completato")
    
    return clean_params, poisoned_params

def analyze_models_similarity(clean_params, poisoned_params):
    """Analizza le similitudini tra modelli usando tutti i metodi."""
    
    methods = ['cosine_original', 'cosine_improved', 'euclidean', 'statistical', 'combined', 'cosine_enhanced']
    
    results = {
        'clean_vs_clean': {},
        'clean_vs_poisoned': {},
        'poisoned_vs_poisoned': {}
    }
    
    for method in methods:
        results['clean_vs_clean'][method] = []
        results['clean_vs_poisoned'][method] = []
        results['poisoned_vs_poisoned'][method] = []
    
    print("Analizzando similitudini...")
    
    # Clean vs Clean
    for i in range(len(clean_params)):
        for j in range(i+1, len(clean_params)):
            for method in methods:
                if method == 'cosine_original':
                    sim = parameter_utils.cosine_similarity_parameters(clean_params[i], clean_params[j])
                elif method == 'cosine_improved':
                    sim = parameter_utils.cosine_similarity_parameters_improved(clean_params[i], clean_params[j])
                else:
                    sim = parameter_utils.enhanced_similarity_score(clean_params[i], clean_params[j], method=method)
                results['clean_vs_clean'][method].append(sim)
    
    # Clean vs Poisoned
    for clean_param in clean_params:
        for poisoned_param in poisoned_params:
            for method in methods:
                if method == 'cosine_original':
                    sim = parameter_utils.cosine_similarity_parameters(clean_param, poisoned_param)
                elif method == 'cosine_improved':
                    sim = parameter_utils.cosine_similarity_parameters_improved(clean_param, poisoned_param)
                else:
                    sim = parameter_utils.enhanced_similarity_score(clean_param, poisoned_param, method=method)
                results['clean_vs_poisoned'][method].append(sim)
    
    # Poisoned vs Poisoned
    for i in range(len(poisoned_params)):
        for j in range(i+1, len(poisoned_params)):
            for method in methods:
                if method == 'cosine_original':
                    sim = parameter_utils.cosine_similarity_parameters(poisoned_params[i], poisoned_params[j])
                elif method == 'cosine_improved':
                    sim = parameter_utils.cosine_similarity_parameters_improved(poisoned_params[i], poisoned_params[j])
                else:
                    sim = parameter_utils.enhanced_similarity_score(poisoned_params[i], poisoned_params[j], method=method)
                results['poisoned_vs_poisoned'][method].append(sim)
    
    return results

def print_results(results):
    """Stampa i risultati dell'analisi."""
    
    methods = ['cosine_original', 'cosine_improved', 'euclidean', 'statistical', 'combined', 'cosine_enhanced']
    
    print("\n" + "="*80)
    print("RISULTATI ANALISI SIMILITUDINI CON LABEL FLIPPING ATTACK")
    print("="*80)
    
    for method in methods:
        print(f"\n=== {method.upper()} ===")
        
        clean_clean = results['clean_vs_clean'][method]
        clean_poisoned = results['clean_vs_poisoned'][method]
        poisoned_poisoned = results['poisoned_vs_poisoned'][method]
        
        if clean_clean:
            print(f"Clean vs Clean:      {min(clean_clean):.6f} - {max(clean_clean):.6f} (media: {np.mean(clean_clean):.6f})")
        if clean_poisoned:
            print(f"Clean vs Poisoned:   {min(clean_poisoned):.6f} - {max(clean_poisoned):.6f} (media: {np.mean(clean_poisoned):.6f})")
        if poisoned_poisoned:
            print(f"Poisoned vs Poisoned: {min(poisoned_poisoned):.6f} - {max(poisoned_poisoned):.6f} (media: {np.mean(poisoned_poisoned):.6f})")
        
        # Calcola discriminazione
        if clean_clean and clean_poisoned:
            clean_avg = np.mean(clean_clean)
            poisoned_avg = np.mean(clean_poisoned)
            discrimination = abs(clean_avg - poisoned_avg)
            print(f"Discriminazione:     {discrimination:.6f}")
            
            # Suggerisci soglia
            threshold = (clean_avg + poisoned_avg) / 2
            print(f"Soglia suggerita:    {threshold:.6f}")

def suggest_best_method(results):
    """Suggerisce il metodo migliore basato sulla discriminazione."""
    
    methods = ['cosine_original', 'cosine_improved', 'euclidean', 'statistical', 'combined', 'cosine_enhanced']
    discriminations = {}
    
    for method in methods:
        clean_clean = results['clean_vs_clean'][method]
        clean_poisoned = results['clean_vs_poisoned'][method]
        
        if clean_clean and clean_poisoned:
            clean_avg = np.mean(clean_clean)
            poisoned_avg = np.mean(clean_poisoned)
            discrimination = abs(clean_avg - poisoned_avg)
            discriminations[method] = discrimination
    
    if discriminations:
        best_method = max(discriminations, key=discriminations.get)
        print(f"\n" + "="*80)
        print("RACCOMANDAZIONE")
        print("="*80)
        print(f"Metodo migliore per label flipping attack: {best_method.upper()}")
        print(f"Discriminazione: {discriminations[best_method]:.6f}")
        
        # Calcola soglia ottimale per il metodo migliore
        clean_clean = results['clean_vs_clean'][best_method]
        clean_poisoned = results['clean_vs_poisoned'][best_method]
        threshold = (np.mean(clean_clean) + np.mean(clean_poisoned)) / 2
        
        print(f"Soglia raccomandata: {threshold:.6f}")
        print(f"Configurazione TipSelectionAlgorithm:")
        print(f"  similarity_method='{best_method}'")
        print(f"  similarity_threshold={threshold:.6f}")

def main():
    """Main function."""
    
    print("TEST DISCRIMINAZIONE CON LABEL FLIPPING ATTACK")
    print("="*80)
    print("Creazione e allenamento modelli...")
    
    # Crea e allena modelli
    clean_params, poisoned_params = create_and_train_models(
        num_clean=5, 
        num_poisoned=3, 
        epochs=10
    )
    
    print(f"\nModelli creati:")
    print(f"  - {len(clean_params)} modelli puliti")
    print(f"  - {len(poisoned_params)} modelli poisoned (con label flipping 1->9)")
    
    # Analizza similitudini
    results = analyze_models_similarity(clean_params, poisoned_params)
    
    # Stampa risultati
    print_results(results)
    
    # Suggerisci metodo migliore
    suggest_best_method(results)
    
    print(f"\n" + "="*80)
    print("Test completato!")

if __name__ == "__main__":
    main()
