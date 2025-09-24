# DAGTrustFL - DAG Component
Implementation of the DAG (Directed Acyclic Graph) architecture for enabling trust evaluation in Federated Learning environments.

## Directory Structure

**`frames/`** - Used for generating GIF visualizations of the DAG evolution

**`_OLD_DAG_simulations/`** - Legacy scripts for basic graph behavior simulation:
- Run `python3 Sim_basic.py` to generate the simulated graph
- `Sim_corrupted.py` includes corrupted behavior simulation
- Set malicious percentage and number of peers/nodes/rounds inside the scripts

## Core Components

The Python scripts contain the essential components for the DAG trust simulation:

**`DAG_functions.py`** - Main configuration and trust computation:
- Settings for M% (malicious percentage), C% (corrupted percentage), N (nodes), R (rounds)
- Parameters: lambda, upsilon
- Interaction point between DAG and FL components
- Responsible for trust vector computation

**`TipSelectionAlgorithm.py`** - Trust-weighted selection:
- Settings for varsigma and chi parameters
- Implementation of the trust-weighted tip selection algorithm
- Core algorithm from the research paper

**`DAG.py`** - DAG structure implementation:
- DAG graph structure and operations
- Theta parameter configuration
- Node management and graph maintenance

**`DAGNode.py`** - Individual node implementation:
- Single node properties and behavior
- Node state management

**`Peer.py`** - Peer implementation.

## Usage

### Basic DAG Simulation
```bash
python3 Sim_basic.py
```

### Corrupted Behavior Simulation
```bash
python3 Sim_corrupted.py
```

### Integration with FL Component
The DAG system integrates with the FL component in `../FL/` through `DAG_functions.py`, providing real-time trust evaluation during federated learning rounds. To use the entire FL+DAG system, launche the script on the `../FL/` directory:

```bash
python3 ../FL/label_flipping_attack.py
```

## Configuration

Main parameters can be adjusted in the respective component files:
- Trust computation parameters in `DAG_functions.py`
- Selection algorithm parameters in `TipSelectionAlgorithm.py`
- Graph structure parameters in `DAG.py`

