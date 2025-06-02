import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import DAG
import DAGNode
import Peer
import TipSelectionAlgorithm
import os
from PIL import Image
import shutil

def draw_dag(dag, step, pos, folder="frames", highlight_nodes=None):
    # Rimuovi i file vecchi dalla cartella frames
    if step == 0:  # Solo al primo step
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    dag.update_cumulative_weights()  # Aggiorna i pesi cumulativi
    plt.figure(figsize=(13, 7))
    # Colori dei nodi in base alla conferma, malevolenza e highlight
    node_colors = []
    for node_id in dag.graph.nodes:
        node = dag.nodes[node_id]
        if highlight_nodes is not None and node_id in highlight_nodes:
            if getattr(node, "corrupted", False):  # Controlla se il nodo è corrotto
                node_colors.append("lightgreen")  # Verde chiaro per confermati ma corrotti
            elif getattr(node, "malicious", False):  # Malevolo ma confermato
                node_colors.append("#FFA1A1")  # Rosso chiaro per highlight malevolo
            else:
                node_colors.append("limegreen")  # Verde per highlight
        else:
            if dag.is_confirmed(node_id):
                if getattr(node, "corrupted", False):  # Controlla se il nodo è corrotto
                    node_colors.append("#98690A")  # Orange più scuro per corrotti confermati
                elif getattr(node, "malicious", False):  # Malevolo ma confermato
                    node_colors.append("darkred")
                else:
                    node_colors.append("royalblue")  # Blu scuro per buoni confermati
            elif getattr(node, "malicious", False):  # Controlla se il nodo è malevolo
                node_colors.append("red")  # Rosso per nodi malevoli
            elif getattr(node, "corrupted", False):  # Controlla se il nodo è corrotto
                node_colors.append("orange")
            else:
                node_colors.append("skyblue")    # Azzurro per non buoni confermati

    # Disegna nodi e archi
    nx.draw(
        dag.graph,
        pos,
        with_labels=False,
        node_size=2000,
        node_color=node_colors,
        edge_color="gray",
        arrows=True
    )
    # Etichette: ID abbreviato + CW + media matrice, in grassetto
    labels = {}
    for node_id in dag.graph.nodes:
        node = dag.nodes[node_id]
        matrix_mean = np.mean(node.data)
        labels[node_id] = f"{node_id[:4]}\nCW={node.cw}\nμ={matrix_mean:.2f}"

    nx.draw_networkx_labels(
        dag.graph,
        pos,
        labels=labels,
        font_size=12,
        font_weight="bold"
    )

    plt.title(f"DAG Evolution - Step {step}")
    plt.axis("off")
    plt.savefig(f"{folder}/frame_{step:03d}.png")
    plt.close()

# ======== GIF Generation ========
def make_gif(frame_folder="frames", output_gif="dag_evolution.gif", duration=400):
    images = []
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    for filename in frame_files:
        img = Image.open(os.path.join(frame_folder, filename))
        images.append(img.copy())
        img.close()
    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF salvata come {output_gif}")

def generate_data_matrix(size=(3, 3), malicious=False, corrupted=False):
    if malicious:
        return np.random.rand(*size) * 100
    elif corrupted:
        # You can define a different pattern for corrupted if you want
        # for now, it's just a random matrix to make it similar to the normal ones
        return np.random.rand(*size)
    else:
        return np.random.rand(*size)

def search_path_from_tip_to_lastroundFL(dag, tip, genesis_id):
    """
    Efficiently traverses all paths from the tip to the given genesis_id and returns a set of all nodes involved.
    Uses memoization to avoid redundant traversals in large DAGs.
    """
    nodes_in_paths = set()
    memo = {}

    def dfs(current, path):
        # Avoid cycles
        if current in path:
            return set()
        # Memoization: if already computed, return cached result
        if current in memo:
            return memo[current]
        path.append(current)
        if current == genesis_id:
            nodes = set(path)
            nodes_in_paths.update(nodes)
            path.pop()
            memo[current] = nodes
            return nodes
        else:
            parents = list(dag.graph.successors(current))
            if not parents:
                path.pop()
                memo[current] = set()
                return set()
            total_nodes = set()
            for parent in parents:
                parent_nodes = dfs(parent, path)
                total_nodes.update(parent_nodes)
            if total_nodes:
                total_nodes.add(current)
                nodes_in_paths.update(total_nodes)
            path.pop()
            memo[current] = total_nodes
            return total_nodes

    dfs(tip, [])
    return nodes_in_paths

matplotlib.use('Qt5Agg')

# Simulazione di un DAG con X peer, cada uno fa Y nodi
X= 8  # Numero di peer
Y= 2   # Numero di nodi per peer
R= 4  # Numero di round di FL
malicious_percentage = 0.2  # Percentuale di peer malevoli
corrupted_percentage = 0.2  # Percentuale di peer corrotti


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
draw_dag(dag, step=0, pos=pos)
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
            data_matrix = generate_data_matrix(malicious=is_malicious, corrupted=is_corrupted)
            node = DAGNode.DAGNode(data_matrix)
            node.malicious = is_malicious
            node.corrupted = is_corrupted
            node_id = peer.add_node(node)
            added_node_ids.append(node_id)
            pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
            draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
            step += 1

    print(f"\n\nFINE {round+1} ROUND\nTotale nodi: {len(dag.nodes)} - Totale tips: {len(dag.tips)}")
    
    old_path = set(longest_ppath)
    longest_ppath = set()
    for tip in dag.tips:
        ppath = search_path_from_tip_to_lastroundFL(dag, tip, last_node_prev_round.id)
        if len(ppath) > len(longest_ppath):
            longest_ppath = ppath
    longest_ppath = longest_ppath.union(old_path)
    draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
    step += 1

    # Before removing, gather info on malicious/corrupted nodes confirmed and good nodes eliminated
    num_malicious_confirmed = sum(1 for n in longest_ppath if getattr(dag.nodes.get(n, None), "malicious", False))
    num_corrupted_confirmed = sum(1 for n in longest_ppath if getattr(dag.nodes.get(n, None), "corrupted", False))
    nodes_to_remove = [n for n in list(dag.graph.nodes) if n not in longest_ppath]
    num_good_removed = sum(1 for n in nodes_to_remove if not getattr(dag.nodes.get(n, None), "malicious", False) and not getattr(dag.nodes.get(n, None), "corrupted", False))
    print(f"Malicious nodes confirmed: {num_malicious_confirmed - total_malicious_confirmed}")
    print(f"Corrupted nodes confirmed: {num_corrupted_confirmed - total_corrupted_confirmed}")
    print(f"Good nodes eliminated: {num_good_removed}")

    for n in nodes_to_remove:
        dag.graph.remove_node(n)
        dag.nodes.pop(n, None)
        dag.tips.discard(n)
    total_good_removed += num_good_removed
    total_malicious_confirmed = num_malicious_confirmed
    total_corrupted_confirmed = num_corrupted_confirmed

    #CREATING MILESTONE
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
make_gif()





