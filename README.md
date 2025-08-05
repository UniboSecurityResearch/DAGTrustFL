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
│   └── parameter_utils.py        # Custom library for performing parameters similarities
|                                    (i. e cosine similarity)
│
├── FL/                            # Federated Learning Implementation
│   ├── server.py                 # FL server with DAG integration
│   ├── client.py                 # FL client implementation
│   ├── label_flipping_attack.py  # Main experiment script for adversarial scenarios
│   ├── analyze_fl_csv_pdf.py     # Comprehensive FL results analysis
│   └── analyze_dag_csv_pdf.py    # Comprehensive DAG results analysis and visualization
│
├── requirements.pip              # Python dependencies
└── README.md                     # This documentation
```

## Key Features

- **Trust-Based Participant Evaluation**: Continuous assessment of participant behavior using DAG structural consistency
- **Malicious Detection**: Automatic identification of label flipping attacks and model poisoning
- **Similarity-Based Filtering**: Advanced cosine similarity analysis for parameter validation
- **Comprehensive Analytics**: Built-in visualization and analysis tools for experiment evaluation
- **Scalable Architecture**: Supports multiple participants with configurable attack scenarios

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

### Configuration

Key parameters can be modified in `DAG/DAG_functions.py`:
- `X`: Number of federated learning total participants (peers)
- `R`: Number of federated learning traning rounds
- `malicious_percentage`: Fraction of malicious participants (0.0-1.0)
- `corrupted_percentage`: Fraction of corrupted participants (0.0-1.0)

### Analysis and Visualization

After running experiments, analyze results using:

```bash
# Comprehensive DAG trust analysis
python FL/analyze_dag_csv_pdf.py

# Federated learning performance analysis
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
4. **Aggregation**: Server aggregates trusted updates
5. **Analysis**: Comprehensive evaluation of system performance and security

## Output Files

- `dag_trust_analysis.csv`: Trust vectors, structural consistency, and detection statistics
- `3000_results.csv`: Federated learning metrics (accuracy, loss, precision, recall)
- `complete_dag_analysis_report.pdf`: Comprehensive DAG system analysis
- `fl_results_analysis.pdf`: Federated learning performance report

## Research Applications

This framework is designed for research in:
- Trustworthy federated learning
- Byzantine fault tolerance in distributed ML
- Privacy-preserving collaborative learning
- Malicious participant detection
- Decentralized trust systems

## Contributing

This is a research repository maintained by the University of Bologna Security Research group. For questions or collaboration inquiries, please contact Giacomo Gori at g.gori@unibo.it

## License

This project is part of ongoing research at the University of Bologna. Please refer to the institution's guidelines for usage and distribution.
