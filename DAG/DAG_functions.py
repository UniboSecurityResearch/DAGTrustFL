import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import DAG
import DAGNode
import Peer
import TipSelectionAlgorithm
import parameter_utils
from PIL import Image
import shutil
import dag_utils
import gif
import csv
import os

#matplotlib.use('Qt5Agg')
# Simulazione di un DAG con X peer, cada uno fa Y nodi
X= 5  # Numero di peer
R= 50  # Numero di round di FL
malicious_percentage = 0.2  # Percentuale di peer malevoli
corrupted_percentage = 0  # Percentuale di peer corrotti

trust_vector = [1] * X
structural_consistencies = []
updates_contributions = []
upsilon_1 = 0.5  # Weight for structural consistency
upsilon_2 = 0.5  # Weight for updates contributions
lambda_ = 0.7  # Weight for influence of historical trust evaluations
dag = None
selector = None
peers = []
central_node = None
malicious_indices = []
num_malicious = 0
num_corrupted = 0
corrupted_indices = []
last_node_prev_round = None
longest_ppath = set()
step = 1
added_node_ids = []
frames_info = []  # Per salvare info dei frame se serve
total_good_removed = 0
total_malicious_confirmed = 0
total_corrupted_confirmed = 0
pos = None
csv_filename = None  # Nome del file CSV per salvare i dati


def init_csv_logging(filename="dag_trust_analysis.csv"):
    """Inizializza il file CSV per salvare i dati dei round."""
    global csv_filename
    csv_filename = filename
    
    # Crea l'header del CSV
    headers = [
        'Round', 'Total_Nodes', 'Total_Tips', 
        'Good_Nodes_Removed_This_Round', 'Malicious_Nodes_Confirmed_This_Round', 
        'Corrupted_Nodes_Confirmed_This_Round',
        'Total_Good_Removed', 'Total_Malicious_Confirmed', 'Total_Corrupted_Confirmed',
        'Similarity_Threshold', 'Mean_Similarity_Good', 'Mean_Similarity_Bad'
    ]
    
    # Aggiungi colonne per il trust vector e le metriche per ogni peer
    for i in range(X):
        headers.extend([
            f'Peer_{i}_Trust', f'Peer_{i}_Structural_Consistency', 
            f'Peer_{i}_Updates_Contribution', f'Peer_{i}_Is_Malicious', 
            f'Peer_{i}_Is_Corrupted'
        ])
    
    # Scrivi l'header
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    
    print(f"CSV logging initialized: {csv_filename}")


def save_round_data(epoch, num_good_removed, num_malicious_confirmed_this_round, num_corrupted_confirmed_this_round):
    """Salva i dati del round corrente nel file CSV."""
    global csv_filename, dag, trust_vector, structural_consistencies, updates_contributions
    global total_good_removed, total_malicious_confirmed, total_corrupted_confirmed, peers, selector

    if csv_filename is None:
        print("Warning: CSV logging not initialized. Call init_csv_logging() first.")
        return

    # Calcola la soglia di similitudine attuale
    similarity_threshold = selector.get_threshold()

    # Calcola le similitudini medie dei nodi buoni e cattivi
    good_similarities = []
    bad_similarities = []


    for node_str in dag.nodes:
        node = dag.nodes[node_str]
        if node.peer.malicious == False and node.epoch == epoch:
            print("NODO BUONO:", node.id[:6], "DEL PEER", node.peer.id)
            sim_g = []
            sim_b = []
            for node_other_str in dag.nodes:
                node_other = dag.nodes[node_other_str]
                if node_other.epoch == epoch:
                    print("CONFRONTA CON:", node_other.id[:6], "DEL PEER", node_other.peer.id)
                    if node_other.malicious == False:
                        # Calcola la similitudine rispetto gli altri nodi buoni del round (se esiste la funzione nel selettore)
                        #sim_g.append(selector.cs_evaluate_similarity(node.id, dag, node_other.data))
                        sim_g.append(parameter_utils.cos_sim_simple(node.data, node_other.data))
                        print("Peer numero", node.peer.id, "ha similitudini buone:", sim_g)
                    else:
                        # Calcola la similitudine rispetto agli altri nodi malevoli del round
                        sim_b.append(parameter_utils.cos_sim_simple(node.data, node_other.data))
                        print("Peer numero", node.peer.id, "ha similitudini cattive:", sim_b)
            #print("Peer numero", node.peer.id, "ha similitudini buone:", sim_g, "e cattive:", sim_b)
            good_similarities.append(np.mean(sim_g)) if sim_g else good_similarities.append(0)
            bad_similarities.append(np.mean(sim_b)) if sim_b else bad_similarities.append(0)
    print(f"Epoch {epoch} Good similarities:", good_similarities)
    print(f"Epoch {epoch} Bad similarities:", bad_similarities)


    mean_sim_good = np.mean(good_similarities) if good_similarities else None
    mean_sim_bad = np.mean(bad_similarities) if bad_similarities else None

    # Prepara i dati base del round
    row_data = [
        epoch + 1,  # Round (1-indexed)
        len(dag.nodes),  # Total nodes
        len(dag.tips),   # Total tips
        num_good_removed,  # Good nodes removed this round
        num_malicious_confirmed_this_round,  # Malicious nodes confirmed this round
        num_corrupted_confirmed_this_round,   # Corrupted nodes confirmed this round
        total_good_removed,  # Total good removed
        total_malicious_confirmed,  # Total malicious confirmed
        total_corrupted_confirmed,   # Total corrupted confirmed
        similarity_threshold, mean_sim_good, mean_sim_bad
    ]
    
    # Aggiungi i dati per ogni peer
    for i in range(X):
        trust_val = trust_vector[i] if i < len(trust_vector) else 0.0
        struct_consistency = structural_consistencies[i] if i < len(structural_consistencies) else 1.0
        updates_contrib = updates_contributions[i] if i < len(updates_contributions) else 1.0
        is_malicious = 1 if i in malicious_indices else 0
        is_corrupted = 1 if i in corrupted_indices else 0
        
        row_data.extend([
            trust_val, struct_consistency, updates_contrib, is_malicious, is_corrupted
        ])
    
    # Scrivi i dati nel CSV
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)
    
    print(f"Round {epoch + 1} data saved to {csv_filename}")


def init_DAG(initial_params):
    global dag, selector, peers, central_node, malicious_indices, num_malicious, num_corrupted, corrupted_indices, last_node_prev_round, pos
    #print(initial_params[0])
    # Inizializza il DAG e il selettore di tip
    dag = DAG.DAG()
    selector = TipSelectionAlgorithm.TrustInformedTipSelector(
        similarity_method='cosine',
    )

    # Crea X peer
    peers = [Peer.Peer(dag, selector, name=f"Peer{i}", id=i) for i in range(X)]
    central_node = Peer.Peer(dag, selector, name="CentralNode", id=X)  # Nodo centrale per aggregazione

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

    # Genesis node - use the first client's parameters as template for structure
    if initial_params and len(initial_params) > 0:
        # Get the first client's parameters as a template and create a neutral version
        first_client_params = list(initial_params.values())[0]
        try:
            import parameter_utils
            # Use the safer copy_parameter_structure function instead of random generation
            genesis_params = parameter_utils.copy_parameter_structure(first_client_params)
        except (ImportError, Exception) as e:
            print(f"Warning: Could not create genesis parameters from template: {e}")
            # Fallback to simple matrix if parameter_utils not available
            genesis_params = dag_utils.generate_data_matrix(malicious=False, corrupted=False)
    else:
        # Fallback to simple matrix
        genesis_params = dag_utils.generate_data_matrix(malicious=False, corrupted=False)
    
    genesis = DAGNode.DAGNode(genesis_params, peer=central_node, epoch=0)
    central_node.manual_add_node(genesis, [])
    pos = nx.spring_layout(dag.graph, seed=42)  # layout iniziale
    dag_utils.draw_dag(dag, step=0, pos=pos)
    last_node_prev_round = genesis
    #second_node = DAGNode.DAGNode(np.zeros((3, 3)), peer=central_node)
    #dag.add_node(second_node, [])
    print("INITIALIZED GRAPH")
    
    # Inizializza il logging CSV
    init_csv_logging()
    
    return X,R,num_malicious,malicious_indices.tolist()


def get_trust_vector(parameters_clients, epoch):
    global dag, selector, peers, longest_ppath, step, added_node_ids, total_good_removed, total_malicious_confirmed, total_corrupted_confirmed, trust_vector, structural_consistencies, updates_contributions, last_node_prev_round, pos
    
    #GIACOMO: Take the actual neural network parameters and put them in the DAG nodes
    for peer in peers:
        if peer.id in parameters_clients.keys():
            print("NODO NUMERO " + str(peer.id))
            is_malicious = getattr(peer, "malicious", False)
            is_corrupted = getattr(peer, "corrupted", False)
            
            # Use the actual neural network parameters from the client
            actual_parameters = parameters_clients[peer.id]
            
            # For testing purposes, we can still modify parameters for malicious/corrupted peers
            # but in the real system, the detection should work with actual parameter differences
            if is_malicious:
                # For malicious peers, we could potentially modify their parameters
                # but let's use the actual parameters for now to test real detection
                print("MALICIOUS")
                pass
            if is_corrupted:
                # For corrupted peers, same approach
                print("CORRUPTED")
                pass
                
            node = DAGNode.DAGNode(actual_parameters, peer=peer, epoch=epoch)
            node.malicious = is_malicious
            node.corrupted = is_corrupted
            node_id = peer.add_node(node)
            added_node_ids.append(node_id)
            pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
            dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
            step += 1

    print(f"\n\nFINE {epoch+1} ROUND\nTotale nodi: {len(dag.nodes)} - Totale tips: {len(dag.tips)}")
    
    old_path = set(longest_ppath)
    # Heaviest subgraph computations
    unconfirmed_tips = [tip for tip in dag.tips if not dag.is_confirmed(tip)]
    #TODO:
    # compute 2 longest_paths, so that the milestone can connect to both
    # and reduce drastically the probability of good nodes being removed
    longest_ppath = compute_longest_path(dag, selector, unconfirmed_tips, last_node_prev_round, old_path)
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
    
    # Salva i dati prima del calcolo del trust vector
    num_malicious_confirmed_this_round = num_malicious_confirmed - total_malicious_confirmed
    num_corrupted_confirmed_this_round = num_corrupted_confirmed - total_corrupted_confirmed
    
    # Develop trust vector for each peer based on structural consistency and updates contributions
    for i, peer in enumerate(peers):
        structural_consistency = 1  # Default to trust
        updates_contribution = 1  # Default to trust
        # Find the node created by this specific peer in the longest path
        peer_node_in_path = None
        for node_id in longest_ppath:
            node = dag.nodes[node_id]
            if hasattr(node, 'peer') and node.peer == peer and node.epoch == epoch:
                peer_node_in_path = node_id
                break
            #TODO: to add efficiency, stop when you find last_node_prev_round node
            #but it depends on the fact that the order in longest_ppath is the same as in the graph -> to be checked
        
        # If this peer has a node in the longest path and last FL round, check its past cone similarity
        if peer_node_in_path is not None:
            nodes_outside = selector.check_past_cone_similarity([peer_node_in_path], dag)
            if len(nodes_outside) > 0:
                structural_consistency = 0  # Corrupted behavior detected in this peer's node past cone
                print(f"Peer {i} ({peer.name}): Corrupted behavior detected in node {peer_node_in_path[:6]}")
        else:
            # If peer has no node in the longest path, check if they have any node in the graph
            peer_has_node = False
            for node_id in dag.nodes:
                node = dag.nodes[node_id]
                if hasattr(node, 'peer') and node.peer == peer and node.epoch == epoch:
                    peer_has_node = True
                    # The update was outside the heaviest subgraph, so updates contributions is 0
                    print(f"Peer {i} ({peer.name}): Node outside heaviest graph in node {node_id[:6]} (not in longest path)")
                    updates_contribution = 0
                    # Check this peer's node even if not in longest path
                    nodes_outside = selector.check_past_cone_similarity([node_id], dag)
                    if len(nodes_outside) > 0:
                        structural_consistency = 0  # Corrupted behavior detected
                        print(f"Peer {i} ({peer.name}): Corrupted behavior detected outside heaviest graph in node {node_id[:6]} (not in longest path)")
                    break
            
            if not peer_has_node:
                print(f"Peer {i} ({peer.name}): No node found in the graph for this FL round")
                #The peer didn't participate in this round, so we assume update contributions is 0.5
                updates_contribution = trust_vector[i] if i < len(trust_vector) else 0.5
        if i >= len(updates_contributions):
            updates_contributions.append(updates_contribution)
            structural_consistencies.append(structural_consistency)
        else:
            # Update existing values    
            updates_contributions[i] = updates_contribution
            structural_consistencies[i] = structural_consistency
        trust_value = upsilon_1 * structural_consistency + upsilon_2 * updates_contribution
        trust_value_t = (1-lambda_) * trust_value + lambda_ * (trust_vector[i] if i < len(trust_vector) else trust_value)  # Historical trust influence
        trust_vector[i] = trust_value_t
    print(f"Structural Consistency vector: {structural_consistencies}")
    print(f"Update Contributions vector: {updates_contributions}")
    print(f"Trust vector: {trust_vector}")
    
    # Salva i dati del round nel CSV
    save_round_data(epoch, num_good_removed, num_malicious_confirmed_this_round, num_corrupted_confirmed_this_round)
    
    # Remove nodes outside of the heaviest subgraph
    for n in nodes_to_remove:
        dag.graph.remove_node(n)
        dag.nodes.pop(n, None)
        dag.tips.discard(n)
    total_good_removed += num_good_removed
    total_malicious_confirmed = num_malicious_confirmed
    total_corrupted_confirmed = num_corrupted_confirmed
    
    if epoch == R/2:
        # Snellisci il grafo eliminando tutti i nodi fino all'ultima milestone
        print(f"Pruning graph at epoch {epoch + 1} (R/2)")
        
        # Trova tutti i nodi da rimuovere (tutti tranne quelli dalla milestone in poi)
        nodes_to_prune = []
        milestone_found = False
        
        # Trova l'ultima milestone (dovrebbe essere last_node_prev_round)
        if last_node_prev_round:
            # Trova tutti i nodi che precedono l'ultima milestone
            for node_id in list(dag.graph.nodes):
                node = dag.nodes.get(node_id)
                if node and node.id != last_node_prev_round.id:
                    # Verifica se questo nodo è un predecessore dell'ultima milestone
                    if node.epoch < last_node_prev_round.epoch:
                        nodes_to_prune.append(node_id)
        
        # Rimuovi i nodi trovati
        for node_id in nodes_to_prune:
            if node_id in dag.graph.nodes:
                dag.graph.remove_node(node_id)
                dag.nodes.pop(node_id, None)
                dag.tips.discard(node_id)
                longest_ppath.discard(node_id)  # Remove from longest path set too
        
        print(f"Removed {len(nodes_to_prune)} nodes during graph pruning")
        print(f"Remaining nodes: {len(dag.nodes)}")
        
    if epoch == R:
        print("*****END OF TRUST VECTOR COMPUTATION*****")
        print("Total good nodes removed:", total_good_removed)
        print("Total malicious nodes confirmed:", total_malicious_confirmed)
        print("Total corrupted nodes confirmed:", total_corrupted_confirmed)
    
    # Update similarity threshold logarithmically
    # From 0.33 at round 1 to 0.68 at round 200 (sempre basato su 200 round come riferimento)
    # Crescita molto più lenta usando logaritmo con base più alta e smorzamento
    min_threshold = 0.33
    max_threshold = 0.68
    target_rounds = 200  # Sempre basato su 200 round come riferimento
    current_round = epoch + 1
    
    # Crescita logaritmica molto più lenta: usiamo log con base più alta e un esponente frazionario
    # Formula: threshold = min + (max - min) * (log(current_round + 10) / log(target_rounds + 10))^0.6
    # Il +10 rallenta la crescita iniziale, l'esponente 0.6 rallenta la crescita complessiva
    if current_round <= target_rounds:
        # Crescita molto graduale con smorzamento
        base_progress = np.log(current_round + 10) / np.log(target_rounds + 10)
        threshold_progress = np.power(base_progress, 0.6)  # Esponente < 1 per crescita più lenta
        new_threshold = min_threshold + (max_threshold - min_threshold) * threshold_progress
        selector.update_threshold(new_threshold)
        print(f"Updated similarity threshold to: {new_threshold:.4f} (round {current_round}/{R}, based on 200-round scale, slow growth)")
    else:
        # Se il round corrente supera 200, mantieni il threshold massimo
        selector.update_threshold(max_threshold)
        print(f"Updated similarity threshold to: {max_threshold:.4f} (round {current_round}/{R}, max threshold reached)")
    
    return trust_vector, updates_contributions


def milestone(new_nn_params, epoch):
    global dag, central_node, longest_ppath, added_node_ids, step, last_node_prev_round, pos
    
    #CREATING MILESTONE
    last_node = None
    for node_id in longest_ppath:
        try:
            if all(parent not in longest_ppath for parent in dag.graph.predecessors(node_id)):
                last_node = node_id
                break
        except Exception as e:
            print(f"Error accessing predecessors for node {node_id}: {e}")
            last_node = node_id
            continue
    if last_node is not None:
        last_node_prev_round = dag.nodes[last_node]
    
    # Use the actual aggregated neural network parameters for the milestone
    milestone = DAGNode.DAGNode(new_nn_params, peer=central_node, epoch=epoch)
    central_node.manual_add_node(new_node=milestone, selected_parents=[last_node_prev_round.id])
    longest_ppath.add(milestone.id)
    print("NEW MILESTONE: " + str(milestone.id))
    print("\n\n")
    added_node_ids.append(node_id)
    last_node_prev_round = milestone
    pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
    dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
    step+=1
    return

def compute_longest_path(dag, selector, unconfirmed_tips, last_node_prev_round, old_path):
    """
    Compute the longest path in the DAG by evaluating tips and checking for corrupted behavior.
    
    Args:
        dag: The DAG instance
        selector: The tip selection algorithm instance
        unconfirmed_tips: List of unconfirmed tip nodes
        last_node_prev_round: The last node from the previous round
        old_path: Set of nodes from previous longest path
        
    Returns:
        set: The longest path as a set of node IDs
    """
    longest_ppath = set()
    
    for tip in unconfirmed_tips:
        # Check past cone similarity
        nodes_to_remove = selector.check_past_cone_similarity([tip], dag)
        print("Evaluating longest path from tip: " + str(tip))
        if len(nodes_to_remove) == 0:
            ppath = dag_utils.search_path_from_tip_to_lastroundFL(dag, tip, last_node_prev_round.id)
            if len(ppath) > len(longest_ppath):
                longest_ppath = ppath
                print("Found new longest path from tip: " + str(tip))
        else:
            print("CORRUPTED")
    
    # Union with old path
    longest_ppath = longest_ppath.union(old_path)
    return longest_ppath