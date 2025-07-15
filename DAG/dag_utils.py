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

def draw_dag(dag, step, pos, folder="../DAG/frames", highlight_nodes=None):
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
    # Etichette: ID abbreviato + CW + media matrice + peer number, in grassetto
    labels = {}
    for node_id in dag.graph.nodes:
        node = dag.nodes[node_id]
        matrix_mean = np.mean(node.data)
        
        # Get peer information
        peer_info = ""
        if hasattr(node, 'peer') and node.peer is not None:
            if hasattr(node.peer, 'name'):
                # Extract peer number from name (e.g., "Peer0" -> "P0", "CentralNode" -> "CN")
                if "Peer" in node.peer.name:
                    peer_num = node.peer.name.replace("Peer", "")
                    peer_info = f"\nP{peer_num}"
                elif "Central" in node.peer.name:
                    peer_info = "\nCN"
                else:
                    peer_info = f"\n{node.peer.name[:2]}"
        
        labels[node_id] = f"{node_id[:4]}\nCW={node.cw}\nμ={matrix_mean:.2f}{peer_info}"

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
