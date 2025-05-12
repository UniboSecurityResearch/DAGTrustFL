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

def draw_dag(dag, step, pos, folder="frames"):
    plt.figure(figsize=(10, 7))
    nx.draw(dag.graph, pos, with_labels=False, node_size=300, node_color="skyblue", edge_color="gray", arrows=True)
    labels = {n: n[:4] for n in dag.graph.nodes()}
    nx.draw_networkx_labels(dag.graph, pos, labels, font_size=6)
    plt.title(f"DAG Evolution - Step {step}")
    plt.axis("off")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/frame_{step:03d}.png")
    plt.close()

# ======== GIF Generation ========
def make_gif(frame_folder="frames", output_gif="dag_evolution.gif", duration=1000):
    images = []
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    for filename in frame_files:
        img = Image.open(os.path.join(frame_folder, filename))
        images.append(img.copy())
        img.close()
    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF salvata come {output_gif}")

matplotlib.use('Qt5Agg')

# Simulazione di un DAG con 5 peer e 25 nodi
# Inizializza il DAG e il selettore di tip
dag = DAG.DAG()
selector = TipSelectionAlgorithm.IOTALikeTipSelector()

# Crea 5 peer
peers = [Peer.Peer(dag, selector, name=f"Peer{i}") for i in range(5)]

# Nodo iniziale
genesis = DAGNode.DAGNode(np.zeros((3, 3)))
dag.add_node(genesis, [])
pos = nx.spring_layout(dag.graph, seed=42)  # layout iniziale
draw_dag(dag, step=0, pos=pos)

# Ogni peer aggiunge 5 nodi (25 totali)
step=1
added_node_ids = []
for _ in range(5):
    for peer in peers:
        node_id = peer.add_node()
        added_node_ids.append(node_id)
        pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
        draw_dag(dag, step=step, pos=pos)
        step += 1

print(f"Numero totale di nodi: {len(dag.nodes)}")
print(f"Numero di tip attuali: {len(dag.tips)}")

# Mostra la matrice di un nodo specifico
sample_id = added_node_ids[7]
print(f"\nMatrice del nodo {sample_id[:8]}:\n{dag.get_node_data(sample_id)}")

print(f"Totale nodi: {len(dag.nodes)} - Totale tips: {len(dag.tips)}")
make_gif()