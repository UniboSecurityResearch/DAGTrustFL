#!/usr/bin/env python3
"""
Test script to evaluate different similarity methods with neural network parameters.
This script helps determine which method provides better discrimination between
good and poisoned parameters.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import parameter_utils

def create_simple_nn_params():
    """Create a simple neural network and return its parameters."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    return model.state_dict()

def poison_parameters(original_params, poison_strength=0.01):
    """
    Create poisoned version of parameters by adding noise.
    
    Args:
        original_params: Original neural network parameters
        poison_strength: How much to modify the parameters (smaller = more subtle)
    
    Returns:
        OrderedDict: Poisoned parameters
    """
    poisoned = OrderedDict()
    for name, param in original_params.items():
        # Add random noise to simulate poisoning
        noise = torch.randn_like(param) * poison_strength
        poisoned[name] = param + noise
    return poisoned

def main():
    print("Testing similarity methods for neural network parameters...")
    print("=" * 60)
    
    # Create multiple test parameters for better calibration
    good_params_list = [create_simple_nn_params() for _ in range(5)]
    
    # Create poisoned versions with different strengths
    poisoned_params_list = []
    for good_params in good_params_list[:3]:  # Use first 3 for poisoning
        poisoned_params_list.append(poison_parameters(good_params, poison_strength=0.001))
        poisoned_params_list.append(poison_parameters(good_params, poison_strength=0.01))
    
    print(f"Created {len(good_params_list)} good parameter sets")
    print(f"Created {len(poisoned_params_list)} poisoned parameter sets")
    print()
    
    # Test individual cases
    test_cases = [
        ("Good vs Good", good_params_list[0], good_params_list[1]),
        ("Good vs Light poison", good_params_list[0], poisoned_params_list[0]),
        ("Good vs Medium poison", good_params_list[0], poisoned_params_list[1]),
    ]
    
    for case_name, params1, params2 in test_cases:
        print(f"\n{case_name}:")
        print("-" * 40)
        results = parameter_utils.compare_similarity_methods(params1, params2, verbose=False)
        
        # Print only similarity scores for clarity
        for method, score in results.items():
            if 'distance' not in method:
                print(f"  {method:20}: {score:.8f}")
    
    print("\n" + "=" * 60)
    print("Auto-calibrating thresholds...")
    print()
    
    # Auto-calibrate thresholds for different methods
    methods_to_test = ['cosine_enhanced', 'combined', 'statistical']
    
    for method in methods_to_test:
        print(f"\nCalibrating {method}:")
        try:
            optimal_threshold, fpr, tpr = parameter_utils.auto_calibrate_threshold(
                good_params_list, poisoned_params_list, method=method, target_fpr=0.05
            )
            print(f"  Recommended threshold: {optimal_threshold:.6f}")
        except Exception as e:
            print(f"  Error calibrating {method}: {e}")
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("- Use the auto-calibrated thresholds in your TipSelectionAlgorithm")
    print("- cosine_enhanced typically provides the best discrimination")
    print("- Monitor false positive rates in your actual federated learning setup")
    
if __name__ == "__main__":
    main()
