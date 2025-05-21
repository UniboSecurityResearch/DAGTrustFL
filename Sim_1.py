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

def draw_dag(dag, step, pos, folder="frames"):
    # Rimuovi i file vecchi dalla cartella frames
    if step == 0:  # Solo al primo step
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    dag.update_cumulative_weights()  # Aggiorna i pesi cumulativi
    plt.figure(figsize=(13, 7))
    # Colori dei nodi in base alla conferma e alla malevolenza
    node_colors = []
    for node_id in dag.graph.nodes:
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
def make_gif(frame_folder="frames", output_gif="dag_evolution.gif", duration=900):
    images = []
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    for filename in frame_files:
        img = Image.open(os.path.join(frame_folder, filename))
        images.append(img.copy())
        img.close()
    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF salvata come {output_gif}")

def generate_data_matrix(size=(3, 3)):
    return np.random.rand(*size)

matplotlib.use('Qt5Agg')

# Simulazione di un DAG con X peer, cada uno fa Y nodi
X= 10  # Numero di peer
Y= 5   # Numero di nodi per peer
malicious_percentage = 0.4  # Imposta la percentuale di peer malevoli (es: 0.4 = 40%)

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

# Ogni peer aggiunge Y nodi
step = 1
added_node_ids = []

for _ in range(Y):
    for peer in peers:
        if getattr(peer, "malicious", False):
            # Genera una matrice malevola con valori molto alti
            data_matrix = np.random.rand(3, 3) * 100
            node = DAGNode.DAGNode(data_matrix)
            node.malicious = True  # Marca il nodo come malevolo
        else:
            # Genera una matrice normale
            data_matrix = generate_data_matrix()
            node = DAGNode.DAGNode(data_matrix)
        
        node_id = peer.add_node(node)
        added_node_ids.append(node_id)
        pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
        draw_dag(dag, step=step, pos=pos)
        step += 1

# Mostra la matrice di un nodo specifico
#sample_id = added_node_ids[7]
#print(f"\nMatrice del nodo {sample_id[:9]}:\n{dag.get_node_data(sample_id)}")

print(f"Totale nodi: {len(dag.nodes)} - Totale tips: {len(dag.tips)}")
make_gif()

#CREATING MILESTONE AT THE ENF OF EVERY FL ROUND??