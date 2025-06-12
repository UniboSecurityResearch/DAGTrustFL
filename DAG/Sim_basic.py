import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import DAG
import DAGNode
import Peer
import TipSelectionAlgorithm
from PIL import Image
import shutil
import dag_utils
import gif

matplotlib.use('Qt5Agg')

# Simulation of a DAG with X peers, each creating Y nodes
X = 5  # Number of peers
Y = 4  # Number of nodes per peer
R = 5  # Number of FL rounds
malicious_percentage = 0.35  # Percentage of malicious peers (e.g., 0.4 = 40%)

# Initialize the DAG and tip selector
dag = DAG.DAG()
selector = TipSelectionAlgorithm.MyCustomTipSelector()

# Create X peers
peers = [Peer.Peer(dag, selector, name=f"Peer{i}") for i in range(X)]

# Randomly select malicious peers based on the percentage
num_malicious = int(X * malicious_percentage)
malicious_indices = np.random.choice(X, num_malicious, replace=False)
for idx in malicious_indices:
    peers[idx].malicious = True

# Initial node
genesis = DAGNode.DAGNode(np.zeros((3, 3)))
dag.add_node(genesis, [])
pos = nx.spring_layout(dag.graph, seed=42)  # initial layout
dag_utils.draw_dag(dag, step=0, pos=pos)
last_node_prev_round = genesis
longest_ppath = set()
# Each peer adds Y nodes
step = 1
added_node_ids = []
frames_info = []  # To save frame info if needed
total_good_removed = 0
total_malicious_confirmed = 0

# START SIMULATION
for round in range(R):
    for _ in range(Y):
        for peer in peers:
            if getattr(peer, "malicious", False):
                # Generate a malicious matrix with very high values
                data_matrix = dag_utils.generate_data_matrix(malicious=True)
                node = DAGNode.DAGNode(data_matrix)
                node.malicious = True  # Mark the node as malicious
            else:
                # Generate a normal matrix
                data_matrix = dag_utils.generate_data_matrix()
                node = DAGNode.DAGNode(data_matrix)
            
            node_id = peer.add_node(node)
            added_node_ids.append(node_id)
            pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
            dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
            step += 1

    # Show the matrix of a specific node
    #sample_id = added_node_ids[7]
    #print(f"\nMatrix of node {sample_id[:9]}:\n{dag.get_node_data(sample_id)}")
    print(f"\n\nEND OF ROUND {round+1}\nTotal nodes: {len(dag.nodes)} - Total tips: {len(dag.tips)}")
    
    # Find the longest path among all tips (if any) up to the genesis
    old_path = set(longest_ppath)
    longest_ppath = set()
    for tip in dag.tips:
        ppath = dag_utils.search_path_from_tip_to_lastroundFL(dag, tip, last_node_prev_round.id)
        if len(ppath) > len(longest_ppath):
            longest_ppath = ppath
    # Redraw the last frame highlighting only the nodes of the longest path tip->genesis in green
    longest_ppath = longest_ppath.union(old_path)  # Union with the old path to keep previous green nodes
    dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
    step += 1
    # Before removing, gather info on malicious nodes confirmed and good nodes eliminated
    num_malicious_confirmed = sum(1 for n in longest_ppath if getattr(dag.nodes.get(n, None), "malicious", False))
    
    # Remove all nodes not in longest_ppath from the DAG
    nodes_to_remove = [n for n in list(dag.graph.nodes) if n not in longest_ppath]
    num_good_removed = sum(1 for n in nodes_to_remove if not getattr(dag.nodes.get(n, None), "malicious", False))
    print(f"Malicious nodes confirmed: {num_malicious_confirmed - total_malicious_confirmed}")
    print(f"Good nodes eliminated: {num_good_removed}")
    for n in nodes_to_remove:
        dag.graph.remove_node(n)
        dag.nodes.pop(n, None)
        dag.tips.discard(n)
    total_good_removed += num_good_removed
    total_malicious_confirmed = num_malicious_confirmed
    # CREATING MILESTONE
    # Save the last node found in the longest path for the next round
    last_node_prev_round = None
    last_node = None
    for node_id in longest_ppath:
        if all(parent not in longest_ppath for parent in dag.graph.predecessors(node_id)):
            last_node = node_id
            break
    if last_node is not None:
        last_node_prev_round = dag.nodes[last_node]
    print("NEW MILESTONE: " + str(last_node_prev_round.id))
    print("\n\n")
print(f"Total Malicious nodes confirmed in this training: {total_malicious_confirmed}")
print(f"Total Good nodes eliminated in this training: {total_good_removed}")
gif.make_gif()





