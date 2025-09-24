# DAGTrustFL

A novel architecture that integrates Directed Acyclic Graph (DAG) structures with Federated Learning to enable trust evaluation and malicious participant detection in distributed machine learning environments.

## Overview

DAGTrustFL combines the immutable and transparent properties of DAG-based systems with federated learning to create a trustworthy distributed training framework. The system automatically detects and mitigates malicious participants while maintaining privacy and decentralization principles.

## Article
Please cite the article if you use this repository: @cite.....

## Repository Structure

```
DAGTrustFL/
├── DAG/                           # DAG Trust System Implementation
│   ├── DAG_functions.py          # Core DAG operations and trust vector computation
│   ├── TipSelectionAlgorithm.py  # Custom tip selection with similarity analysis
│   ├── DAG.py                    # DAG graph implementation
│   ├── DAGNode.py                # Single node of a DAG implementation
│   ├── Peer.py                   # Peer network simulation
│   ├── parameter_utils.py        # Custom library for performing parameters similarities (distance metrics)
│   └── analyze_dag_csv_pdf.py    # DAG analysis and visualization tools
│
├── FL/                            # Federated Learning Implementation
│   ├── server.py                 # FL server with DAG integration
│   ├── client.py                 # FL client implementation
│   ├── label_flipping_attack.py  # Main experiment script for adversarial scenarios
│   ├── analyze_fl_csv_pdf.py     # Comprehensive FL results analysis
│   ├── analyze_dag_csv_pdf.py    # Comprehensive DAG results analysis and visualization
│   ├── federated_learning/       # Core FL algorithms and utilities
│   ├── data_loaders/             # Dataset handling and distribution
│   └── results_scalability/      # Scalability test results
│
├── automated_tests/               # Automated Hyperparameter Testing
│   ├── compare_chi_results.py    # χ (chi) parameter comparison analysis
│   ├── compare_lambda_results.py # λ (lambda) parameter comparison analysis
│   ├── compare_theta_results.py  # θ (theta) parameter comparison analysis
│   ├── compare_varsigma_results.py # ς (varsigma) parameter comparison analysis
│   ├── compare_percentages_*.py  # Malicious percentage comparison studies
│   ├── automated_test_chi.py     # Automated χ parameter testing
│   ├── automated_test_lambda.py  # Automated λ parameter testing
│   ├── automated_test_theta.py   # Automated θ parameter testing
│   ├── automated_test_varsigma.py # Automated ς parameter testing
│   └── automated_test_scalability.py # System scalability testing
│
├── plots/                         # Visualization and Analysis Scripts
│   ├── plot_scalability_graph.py # System scalability analysis plots
│   ├── plot_trust_analysis.py    # Trust evaluation visualization
│   ├── plot_transaction_time_comparison.py # Performance timing analysis
│   ├── plot_model_accuracy_vs_threat.py # Accuracy under attack analysis
│   ├── chi_comparison_plots/     # χ parameter analysis results
│   ├── lambda_comparison_plots/  # λ parameter analysis results  
│   ├── theta_comparison_plots/   # θ parameter analysis results
│   ├── varsigma_comparison_plots/ # ς parameter analysis results
│   └── model_accuracy_comparison/ # ML model performance comparisons
│
├── requirements.pip              # Python dependencies
└── README.md                     # This documentation
```


## Quick Start

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/UniboSecurityResearch/DAGTrustFL.git
   cd DAGTrustFL
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.pip
   ```

### Running an Experiment

To conduct a federated learning experiment with label flipping attacks:

```bash
python FL/label_flipping_attack.py
```

This script will:
1. Initialize the DAG trust system
2. Set up federated learning clients with configurable malicious participants
3. Execute training rounds with trust evaluation
4. Generate comprehensive analysis reports and visualizations
5. Save results in CSV format for further analysis

## Automated Hyperparameter Testing

The repository includes comprehensive automated testing for DAGTrustFL hyperparameters:

### Individual Parameter Testing
```bash
# Test different χ (chi) values for tip selection algorithm
python automated_tests/automated_test_chi.py

# Test different λ (lambda) values for trust computation
python automated_tests/automated_test_lambda.py

# Test different θ (theta) values for DAG structure
python automated_tests/automated_test_theta.py

# Test different ς (varsigma) values for similarity threshold
python automated_tests/automated_test_varsigma.py

# Test system scalability with different participant counts
python automated_tests/automated_test_scalability.py
```

### Results Comparison
After running tests, compare results using scripts in `automated_tests/`:
```bash
# Compare χ parameter results across different values
python automated_tests/compare_chi_results.py

# Compare λ parameter performance
python automated_tests/compare_lambda_results.py

# Compare malicious percentage impact
python automated_tests/compare_percentages_results.py
```

## Visualization and Analysis

The `plots/` directory contains specialized visualization tools:

### Performance Analysis
```bash
# Generate scalability analysis graphs
python plots/plot_scalability_graph.py

# Visualize trust evolution and detection
python plots/plot_trust_analysis.py

# Compare transaction timing performance
python plots/plot_transaction_time_comparison.py

# Analyze model accuracy under different threat levels
python plots/plot_model_accuracy_vs_threat.py
```

### Parameter Analysis Visualization
The system automatically generates comparison plots in respective directories:
- `chi_comparison_plots/` - χ parameter analysis
- `lambda_comparison_plots/` - λ parameter analysis  
- `theta_comparison_plots/` - θ parameter analysis
- `varsigma_comparison_plots/` - ς parameter analysis
- `model_accuracy_comparison/` - ML performance comparisons

### Configuration

Key parameters can be modified in `DAG/DAG_functions.py`:
- `N`: Number of federated learning total participants (peers)
- `R`: Number of federated learning training rounds
- `malicious_percentage`: Fraction of malicious participants (0.0-1.0) [M%]
- `corrupted_percentage`: Fraction of corrupted participants (0.0-1.0) [C%]
- `λ (lambda)`

Enter the DAG directory and read the README to learn how to modify other parameters.

### Manual Analysis and Visualization

After running experiments, analyze results using:

```bash
# DAG trust analysis
python FL/analyze_dag_csv_pdf.py

# FL performance analysis
python FL/fl_results_analyzer.py
```

## Experiment Workflow

1. **Initialization**: DAG structure is created with genesis node
2. **Client Registration**: Participants join the federated learning network
3. **Training Rounds**: 
   - Clients perform local training
   - Parameter updates are submitted to DAG
   - Trust evaluation using similarity analysis
   - Malicious detection and filtering
4. **Aggregation**: Server aggregates trusted updates, and generate a milestone in the DAG
5. **Analysis**: Comprehensive evaluation of system performance and security

## Contributing

This is a research repository maintained by the University of Bologna Security Research group. For questions or collaboration inquiries, please contact Giacomo Gori at g.gori@unibo.it

## License

This project is part of ongoing research at the University of Bologna. Please refer to the institution's guidelines for usage and distribution.
