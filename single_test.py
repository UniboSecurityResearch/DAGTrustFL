#!/usr/bin/env python3
"""
Single test runner for DAGTrustFL - useful for testing the automation.
"""

import sys
import os

# Add the parent directory to the path to import the main script
sys.path.append('/home/giac/DAGTrustFL/DAGTrustFL')

from automated_test_chi import run_experiment_set

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 single_test.py <chi_value> <malicious_pct> <corrupted_pct>")
        print("Example: python3 single_test.py 0 0.1 0.2")
        sys.exit(1)
    
    try:
        chi_value = int(sys.argv[1])
        malicious_pct = float(sys.argv[2])
        corrupted_pct = float(sys.argv[3])
        
        print(f"Running single test with:")
        print(f"  chi = {chi_value}")
        print(f"  malicious_percentage = {malicious_pct}")
        print(f"  corrupted_percentage = {corrupted_pct}")
        
        run_experiment_set(chi_value, malicious_pct, corrupted_pct)
        
        print("Single test completed!")
        
    except ValueError:
        print("Error: Invalid parameter values")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
