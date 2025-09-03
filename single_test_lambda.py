#!/usr/bin/env python3
"""
Single lambda test runner for DAGTrustFL - useful for testing the automation.
"""

import sys
import os

# Add the parent directory to the path to import the main script
sys.path.append('/home/giac/DAGTrustFL/DAGTrustFL')

from automated_test_lambda import run_experiment_set

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 single_test_lambda.py <lambda_value>")
        print("Example: python3 single_test_lambda.py 0.5")
        sys.exit(1)
    
    try:
        lambda_value = float(sys.argv[1])
        
        if lambda_value < 0.0 or lambda_value > 1.0:
            print("Warning: lambda value should typically be between 0.0 and 1.0")
        
        print(f"Running single lambda test with:")
        print(f"  lambda = {lambda_value}")
        
        run_experiment_set(lambda_value)
        
        print("Single lambda test completed!")
        
    except ValueError:
        print("Error: Invalid lambda value (should be a number)")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
