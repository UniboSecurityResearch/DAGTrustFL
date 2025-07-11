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
R= 4  # Numero di round di FL
malicious_percentage = 0.2  # Percentuale di peer malevoli
corrupted_percentage = 0.2  # Percentuale di peer corrotti

trust_vector = [1] * X
structural_consistencies = []
updates_contributions = []
upsilon_1 = 0.5  # Weight for structural consistency
upsilon_2 = 0.5  # Weight for updates contributions
lambda_ = 0.5  # Weight for influence of historical trust evaluations


# Inizializza il DAG e il selettore di tip
dag = DAG.DAG()
selector = TipSelectionAlgorithm.TrustInformedTipSelector()

# Crea X peer
peers = [Peer.Peer(dag, selector, name=f"Peer{i}") for i in range(X)]
central_node = Peer.Peer(dag, selector, name="CentralNode")  # Nodo centrale per aggregazione

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
genesis = DAGNode.DAGNode(dag_utils.generate_data_matrix(malicious=False, corrupted=False), peer=central_node)
central_node.manual_add_node(genesis, [])
#dag.add_node(genesis, [])
pos = nx.spring_layout(dag.graph, seed=42)  # layout iniziale
dag_utils.draw_dag(dag, step=0, pos=pos)
last_node_prev_round = genesis
#second_node = DAGNode.DAGNode(np.zeros((3, 3)), peer=central_node)
#dag.add_node(second_node, [])
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
            #print(is_malicious, is_corrupted)
            data_matrix = dag_utils.generate_data_matrix(malicious=is_malicious, corrupted=is_corrupted)
            node = DAGNode.DAGNode(data_matrix, peer=peer)
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
    unconfirmed_tips = [tip for tip in dag.tips if not dag.is_confirmed(tip)]
    for tip in unconfirmed_tips:
        #Check past cone similarity
        nodes_to_remove= selector.check_past_cone_similarity([tip], dag)
        print("Evaluating longest path from tip: " + str(tip))
        if(len(nodes_to_remove) == 0):
            ppath = dag_utils.search_path_from_tip_to_lastroundFL(dag, tip, last_node_prev_round.id)
            if len(ppath) > len(longest_ppath):
                longest_ppath = ppath
        else:
            print("CORRUPTED")
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
    
    # Develop trust vector for each peer based on structural consistency and updates contributions
    #TODO
    #Fare in modo che controlli solo l'ultimo round del grafo, e non tutto il grafo!
    for i, peer in enumerate(peers):
        structural_consistency = 1  # Default to trust
        updates_contribution = 1  # Default to trust
        # Find the node created by this specific peer in the longest path
        peer_node_in_path = None
        for node_id in longest_ppath:
            node = dag.nodes[node_id]
            if hasattr(node, 'peer') and node.peer == peer:
                peer_node_in_path = node_id
                break
        
        # If this peer has a node in the longest path, check its past cone similarity
        if peer_node_in_path is not None:
            nodes_to_remove = selector.check_past_cone_similarity([peer_node_in_path], dag)
            if len(nodes_to_remove) > 0:
                structural_consistency = 0  # Corrupted behavior detected in this peer's node past cone
                print(f"Peer {i} ({peer.name}): Corrupted behavior detected in node {peer_node_in_path[:6]}")
        else:
            # If peer has no node in the longest path, check if they have any node in the graph
            peer_has_node = False
            for node_id in dag.nodes:
                node = dag.nodes[node_id]
                if hasattr(node, 'peer') and node.peer == peer:
                    peer_has_node = True
                    # The update was outside the heaviest subgraph, so updates contributions is 0
                    updates_contribution = 0
                    # Check this peer's node even if not in longest path
                    nodes_to_remove = selector.check_past_cone_similarity([node_id], dag)
                    if len(nodes_to_remove) > 0:
                        structural_consistency = 0  # Corrupted behavior detected
                        print(f"Peer {i} ({peer.name}): Corrupted behavior detected outside heaviest graph in node {node_id[:6]} (not in longest path)")
                    break
            
            if not peer_has_node:
                print(f"Peer {i} ({peer.name}): No node found in the graph")
                #The peer didn't participate in this round, so we assume update contributions is 0.5
                updates_contribution = 0.5
        updates_contributions.append(updates_contribution)
        structural_consistencies.append(structural_consistency)
        trust_value = upsilon_1 * structural_consistency + upsilon_2 * updates_contribution
        trust_value_t = (1-lambda_) * trust_value + lambda_ * (trust_vector[i] if i < len(trust_vector) else 1)  # Historical trust influence
        trust_vector[i] = trust_value_t
        #TODO
        #Usare questi valori per calcolare la weighted aggregation del modello globale
        #print(f"Peer {i} ({peer.name}): Trust value = {trust_value}")
    print(f"Structural Consistency vector: {structural_consistencies}")
    print(f"Update Contributions vector: {updates_contributions}")
    print(f"Trust vector: {trust_vector}")
    
    
    # Remove nodes outside of the heaviest subgraph
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
    #TODO
    #Mettere qua i valori del global model aggregato
    milestone = DAGNode.DAGNode(dag_utils.generate_data_matrix(malicious=False, corrupted=False), peer=central_node)
    central_node.manual_add_node(milestone, [last_node_prev_round.id])
    longest_ppath.add(milestone.id)
    print(longest_ppath)
    print("NEW MILESTONE: " + str(last_node_prev_round.id))
    print("\n\n")
    added_node_ids.append(node_id)
    pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
    dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
    step+=1
    

print(f"Total Malicious nodes confirmed in this training: {total_malicious_confirmed}")
print(f"Total Corrupted nodes confirmed in this training: {total_corrupted_confirmed}")
print(f"Total Good nodes eliminated in this training: {total_good_removed}")
gif.make_gif()





