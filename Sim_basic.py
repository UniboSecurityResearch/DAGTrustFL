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
        if highlight_nodes is not None and node_id in highlight_nodes:
            node_colors.append("limegreen")  # Verde per highlight
        else:
            node = dag.nodes[node_id]
            if getattr(node, "malicious", False):  # Controlla se il nodo è malevolo
                node_colors.append("red")  # Rosso per nodi malevoli
            elif dag.is_confirmed(node_id):
                node_colors.append("royalblue")  # Blu scuro per confermati
            else:
                node_colors.append("skyblue")    # Azzurro per non confermati

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

def generate_data_matrix(size=(3, 3), malicious=False):
    if malicious== False:
        return np.random.rand(*size)
    else:
        return np.random.rand(*size) * 100

def search_path_from_tip_to_lastroundFL(dag, tip, genesis_id):
    """
    Recursively traverses all paths from the tip to the given genesis_id and returns a set of all nodes involved.
    If genesis_id is not reachable from tip, the function will not loop infinitely.
    """
    nodes_in_paths = set()
    visited = set()  # Prevent revisiting nodes in the same path

    def dfs(current, path):
        # Avoid cycles or infinite loops
        if current in path:
            return
        path.append(current)
        if current == genesis_id:
            nodes_in_paths.update(path)
        else:
            parents = list(dag.graph.successors(current))
            if not parents:
                # Dead end, genesis_id not reachable from this path
                path.pop()
                return
            for parent in parents:
                dfs(parent, path)
        path.pop()

    dfs(tip, [])
    return nodes_in_paths

matplotlib.use('Qt5Agg')

# Simulazione di un DAG con X peer, cada uno fa Y nodi
X= 4  # Numero di peer
Y= 3   # Numero di nodi per peer
R= 6  # Numero di round di FL
malicious_percentage = 0.3  # Imposta la percentuale di peer malevoli (es: 0.4 = 40%)

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

# Nodo iniziale
genesis = DAGNode.DAGNode(np.zeros((3, 3)))
dag.add_node(genesis, [])
pos = nx.spring_layout(dag.graph, seed=42)  # layout iniziale
draw_dag(dag, step=0, pos=pos)
last_node_prev_round = genesis
longest_ppath = set()
# Ogni peer aggiunge Y nodi
step = 1
added_node_ids = []
frames_info = []  # Per salvare info dei frame se serve
for round in range(R):
    for _ in range(Y):
        for peer in peers:
            if getattr(peer, "malicious", False):
                # Genera una matrice malevola con valori molto alti
                data_matrix = generate_data_matrix(malicious=True)
                node = DAGNode.DAGNode(data_matrix)
                node.malicious = True  # Marca il nodo come malevolo
            else:
                # Genera una matrice normale
                data_matrix = generate_data_matrix()
                node = DAGNode.DAGNode(data_matrix)
            
            node_id = peer.add_node(node)
            added_node_ids.append(node_id)
            pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
            draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
            step += 1

    # Mostra la matrice di un nodo specifico
    #sample_id = added_node_ids[7]
    #print(f"\nMatrice del nodo {sample_id[:9]}:\n{dag.get_node_data(sample_id)}")
    print(f"\n\nFINE {round+1} ROUND\nTotale nodi: {len(dag.nodes)} - Totale tips: {len(dag.tips)}")
    
    # Trova il path più lungo tra tutti i tip (se esistono) fino al genesis
    old_path= set(longest_ppath)
    longest_ppath = set()
    for tip in dag.tips:
        ppath = search_path_from_tip_to_lastroundFL(dag, tip, last_node_prev_round.id)
        if len(ppath) > len(longest_ppath):
            longest_ppath = ppath
    # Ridisegna l'ultimo frame evidenziando solo i nodi del path più lungo tip->genesis in verde
    longest_ppath = longest_ppath.union(old_path)  # Unione con il vecchio path per mantenere i nodi precedenti
    draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
    step+=1

    # Remove all nodes not in longest_ppath from the DAG
    nodes_to_remove = [n for n in list(dag.graph.nodes) if n not in longest_ppath]
    for n in nodes_to_remove:
        dag.graph.remove_node(n)
        dag.nodes.pop(n, None)
        dag.tips.discard(n)

    #CREATING MILESTONE
    # Save the last node found in the longest path for the next round
    last_node_prev_round = None
    last_node = None
    for node_id in longest_ppath:
        if all(parent not in longest_ppath for parent in dag.graph.predecessors(node_id)):
            last_node = node_id
            break
    if last_node is not None:
        last_node_prev_round = dag.nodes[last_node]
    print("NEW MILESTONE: "+ str(last_node_prev_round.id))
    print("\n\n")
make_gif()





