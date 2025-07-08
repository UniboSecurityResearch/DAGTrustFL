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

# Simulazione di un DAG con X peer, cada uno fa Y nodi
X= 10  # Numero di peer
Y= 1   # Numero di nodi per peer
R= 5  # Numero di round di FL
malicious_percentage = 0.1  # Percentuale di peer malevoli
corrupted_percentage = 0.4  # Percentuale di peer corrotti


# Inizializza il DAG e il selettore di tip
dag = DAG.DAG()
selector = TipSelectionAlgorithm.MyCustomTipSelector()

# Crea X peer
peers = [Peer.Peer(dag, selector, name=f"Peer{i}") for i in range(X)]

# Seleziona casualmente i peer malevoli in base alla percentuale
num_malicious = int(X * malicious_percentage)
malicious_indices = np.random.choice(X, num_malicious, replace=False)
for idx in malicious_indices:
    peers[idx].malicious = True

# Seleziona casualmente i peer corrotti in base alla percentuale
num_corrupted = int(X * corrupted_percentage)
corrupted_indices = np.random.choice([i for i in range(X) if i not in malicious_indices], num_corrupted, replace=False)
for idx in corrupted_indices:
    peers[idx].corrupted = True

# Nodo iniziale
genesis = DAGNode.DAGNode(np.zeros((3, 3)))
dag.add_node(genesis, [])
pos = nx.spring_layout(dag.graph, seed=42)  # layout iniziale
dag_utils.draw_dag(dag, step=0, pos=pos)
last_node_prev_round = genesis
longest_ppath = set()
step = 1
added_node_ids = []
frames_info = []  # Per salvare info dei frame se serve
total_good_removed = 0
total_malicious_confirmed = 0
total_corrupted_confirmed = 0

# Ogni peer aggiunge Y nodi
for round in range(R):
    for _ in range(Y):
        for peer in peers:
            is_malicious = getattr(peer, "malicious", False)
            is_corrupted = getattr(peer, "corrupted", False)
            print(is_malicious, is_corrupted)
            data_matrix = dag_utils.generate_data_matrix(malicious=is_malicious, corrupted=is_corrupted)
            node = DAGNode.DAGNode(data_matrix)
            node.malicious = is_malicious
            node.corrupted = is_corrupted
            node_id = peer.add_node(node)
            added_node_ids.append(node_id)
            pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
            dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
            step += 1

    print(f"\n\nFINE {round+1} ROUND\nTotale nodi: {len(dag.nodes)} - Totale tips: {len(dag.tips)}")
    
    old_path = set(longest_ppath)
    # Heaviest subgraph computations
    longest_ppath = set()
    for tip in dag.tips:
        ppath = dag_utils.search_path_from_tip_to_lastroundFL(dag, tip, last_node_prev_round.id)
        #TODO
        # Aggiungere check del past cone come dal tip selection algorithm
        if len(ppath) > len(longest_ppath):
            longest_ppath = ppath
    longest_ppath = longest_ppath.union(old_path)
    dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
    step += 1

    # Before removing, gather info on malicious/corrupted nodes confirmed and good nodes eliminated
    num_malicious_confirmed = sum(1 for n in longest_ppath if getattr(dag.nodes.get(n, None), "malicious", False))
    num_corrupted_confirmed = sum(1 for n in longest_ppath if getattr(dag.nodes.get(n, None), "corrupted", False))
    nodes_to_remove = [n for n in list(dag.graph.nodes) if n not in longest_ppath]
    num_good_removed = sum(1 for n in nodes_to_remove if not getattr(dag.nodes.get(n, None), "malicious", False) and not getattr(dag.nodes.get(n, None), "corrupted", False))
    print(f"Malicious nodes confirmed: {num_malicious_confirmed - total_malicious_confirmed}")
    print(f"Corrupted nodes confirmed: {num_corrupted_confirmed - total_corrupted_confirmed}")
    print(f"Good nodes eliminated: {num_good_removed}")

    # Remove nodes outside of the heaviest subgraph
    for n in nodes_to_remove:
        dag.graph.remove_node(n)
        dag.nodes.pop(n, None)
        dag.tips.discard(n)
    total_good_removed += num_good_removed
    total_malicious_confirmed = num_malicious_confirmed
    total_corrupted_confirmed = num_corrupted_confirmed


    #TODO
    # Aggiungere calcoli trust del nodo centrale (structural consistency e updates contributions)
    # Così da avere i valori per il weighted FedAvg



    
    #CREATING MILESTONE
    #TODO
    #Da cambiare: adesso la milestone è un nuovo nodo che contiene il global model
    # e viene aggiunto al DAG come un normale nodo, ma in futuro potrebbe essere
    # un nodo speciale o un'operazione di aggregazione
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
print(f"Total Corrupted nodes confirmed in this training: {total_corrupted_confirmed}")
print(f"Total Good nodes eliminated in this training: {total_good_removed}")
gif.make_gif()





