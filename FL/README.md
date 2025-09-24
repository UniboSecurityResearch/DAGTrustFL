# DAGTrustFL - FL Component

This directory contains the federated learning implementation of DAGTrustFL, a system that integrates Directed Acyclic Graph (DAG) structures with Federated Learning to enable trust evaluation and malicious participant detection.

## Installation

1) Create a virtual environment (Python 3.7+)
```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.pip
```

3) For visualization and analysis features, matplotlib is required:
```bash
pip install matplotlib
```

## Quick Start

### Setup

Before running experiments, complete the initial setup:

1) **Generate data distribution**:
   ```bash
   python3 generate_data_distribution.py
   ```
   This downloads datasets and generates static distribution for consistent experiments.

2) **Generate default models**:
   ```bash
   python3 generate_default_models.py
   ```
   This creates instances of all models used in experiments.

### Running the Main Experiment

**Primary execution script**:
```bash
python3 label_flipping_attack.py
```

This script orchestrates a single complete DAGTrustFL experiment, as done in the paper.

## Experimental Components

### Core Experiments

- **`label_flipping_attack.py`** - Main experiment script with DAG trust integration

### Analysis Scripts

- **`analyze_fl_csv_pdf.py`** - FL results analysis and PDF report generation
- **`analyze_dag_csv_pdf.py`** - DAG trust analysis and visualization

## Configuration

### Hyperparameters

Main FL hyperparameters can be modified in:
- **`federated_learning/arguments.py`** - General FL settings
- **`server.py`** - Setting of K (participants per round, in function of N), malicious and corrupted indices.

### Recommended Settings

**Fashion-MNIST (CNN)**:
- Batch size: 4
- Learning rate: 0.001
- Epochs: 100
- Momentum: 0.9
- Scheduler step size: 10
- Scheduler gamma: 0.1
- Min LR: 1e-10

## Results and Output

### Generated Files

The experiment generates several output files:
- **`3000_results.csv`** - FL training metrics and results
- **`3000_workers_selected.csv`** - Worker selection each round
- **`dag_trust_analysis.csv`** - DAG trust vectors and participant scores

## Directory Structure

```
FL/
├── federated_learning/          # Core FL implementation
├── data_loaders/               # Dataset handling utilities
├── data/                      # Training and test datasets
├── default_models/            # Model templates
├── results/                   # Experiment output files
└── logs/                      # Execution logs
```

## Advanced Usage

### Scalability Testing

For automated experiments across different parameters, use the ../automated_tests folder:
```bash
python3 ../automated_tests/automated_test_*.py*
```

### Custom Experiments

Modify experiment parameters in the respective scripts or through command-line arguments. See individual script documentation for available options.

## Integration with DAG Component

This FL component works in conjunction with the DAG trust system located in `../DAG/`.
