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
X= 5  # Numero di peer
R= 50  # Numero di round di FL
malicious_percentage = 0.34  # Percentuale di peer malevoli
corrupted_percentage = 0  # Percentuale di peer corrotti

trust_vector = [1] * X
structural_consistencies = []
updates_contributions = []
upsilon_1 = 0.5  # Weight for structural consistency
upsilon_2 = 0.5  # Weight for updates contributions
lambda_ = 0.5  # Weight for influence of historical trust evaluations
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


def init_DAG(initial_params):
    global dag, selector, peers, central_node, malicious_indices, num_malicious, num_corrupted, corrupted_indices, last_node_prev_round, pos
    #print(initial_params[0])
    # Inizializza il DAG e il selettore di tip
    dag = DAG.DAG()
    selector = TipSelectionAlgorithm.TrustInformedTipSelector(
        similarity_method='euclidean',
        similarity_threshold=0.537608
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
                updates_contribution = 0.5
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
    
    
    # Remove nodes outside of the heaviest subgraph
    for n in nodes_to_remove:
        dag.graph.remove_node(n)
        dag.nodes.pop(n, None)
        dag.tips.discard(n)
    total_good_removed += num_good_removed
    total_malicious_confirmed = num_malicious_confirmed
    total_corrupted_confirmed = num_corrupted_confirmed

    if epoch == R:
        print("*****END OF TRUST VECTOR COMPUTATION*****")
        print("Total good nodes removed:", total_good_removed)
        print("Total malicious nodes confirmed:", total_malicious_confirmed)
        print("Total corrupted nodes confirmed:", total_corrupted_confirmed)
    
    return trust_vector, updates_contributions


def milestone(new_nn_params, epoch):
    global dag, central_node, longest_ppath, added_node_ids, step, last_node_prev_round, pos
    
    #CREATING MILESTONE
    last_node = None
    for node_id in longest_ppath:
        if all(parent not in longest_ppath for parent in dag.graph.predecessors(node_id)):
            last_node = node_id
            break
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