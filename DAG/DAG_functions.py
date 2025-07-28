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
# Simulation of a DAG with X peers, each one makes Y nodes
X= 10  # Number of peers
R= 200  # Number of FL rounds
malicious_percentage = 0.1  # Percentage of malicious peers
corrupted_percentage = 0.3  # Percentage of corrupted peers

trust_vector = [1] * X
structural_consistencies = []
updates_contributions = []
upsilon_1 = 0.2  # Weight for structural consistency
upsilon_2 = 0.8  # Weight for updates contributions
lambda_ = 0.6  # Weight for influence of historical trust evaluations
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
frames_info = []  # To save frame info if needed
total_good_removed = 0
total_malicious_confirmed = 0 #The total refers to the total number from last pruning of the graph
total_corrupted_confirmed = 0 #The total refers to the total number from last pruning of the graph
pos = None
csv_filename = None  # CSV filename to save data
checkpoint_malicious_conf = 0 #To be summed to total_malicious_confirmed to get the total number of malicious nodes confirmed
checkpoint_corrupted_conf = 0 #To be summed to total_corrupted_confirmed to get the total number of corrupted nodes confirmed
mean_sim_good = 0.4
mean_sim_bad = 0.2

def init_csv_logging(filename="dag_trust_analysis.csv", logger=None):
    """Initialize the CSV file to save round data."""
    global csv_filename
    csv_filename = filename
    
    # Create CSV header
    headers = [
        'Round', 'Total_Nodes', 'Total_Tips', 
        'Good_Nodes_Removed_This_Round', 'Malicious_Nodes_Confirmed_This_Round', 
        'Corrupted_Nodes_Confirmed_This_Round',
        'Total_Good_Removed', 'Total_Malicious_Confirmed', 'Total_Corrupted_Confirmed',
        'Similarity_Threshold', 'Mean_Similarity_Good', 'Mean_Similarity_Bad'
    ]
    
    # Add columns for trust vector and metrics for each peer
    for i in range(X):
        headers.extend([
            f'Peer_{i}_Trust', f'Peer_{i}_Structural_Consistency', 
            f'Peer_{i}_Updates_Contribution', f'Peer_{i}_Is_Malicious', 
            f'Peer_{i}_Is_Corrupted'
        ])
    
    # Write header
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    
    if logger:
        logger.info(f"CSV logging initialized: {csv_filename}")


def save_round_data(epoch, num_good_removed, num_malicious_confirmed_this_round, num_corrupted_confirmed_this_round, logger=None):
    """Save current round data to CSV file."""
    global csv_filename, dag, trust_vector, structural_consistencies, updates_contributions, mean_sim_bad, mean_sim_good
    global total_good_removed, total_malicious_confirmed, total_corrupted_confirmed, peers, selector, checkpoint_corrupted_conf, checkpoint_malicious_conf

    if csv_filename is None:
        if logger:
            logger.warning("CSV logging not initialized. Call init_csv_logging() first.")
        return

    # Calculate current similarity threshold
    similarity_threshold = selector.get_threshold()

    # Calculate mean similarities of good and bad nodes
    good_similarities = []
    bad_similarities = []

    for node_str in dag.nodes:
        node = dag.nodes[node_str]
        if node.peer.malicious == False and node.epoch == epoch:
            sim_g = []
            sim_b = []
            for node_other_str in dag.nodes:
                node_other = dag.nodes[node_other_str]
                if node_other.epoch == epoch:
                    if node_other.malicious == False:
                        # Calculate similarity with respect to other good nodes of the round
                        sim_g.append(parameter_utils.cos_sim_simple(node.data, node_other.data, logger=logger))
                    else:
                        # Calculate similarity with respect to other malicious nodes of the round
                        sim_b.append(parameter_utils.cos_sim_simple(node.data, node_other.data, logger=logger))
            good_similarities.append(np.mean(sim_g)) if sim_g else good_similarities.append(0)
            bad_similarities.append(np.mean(sim_b)) if sim_b else bad_similarities.append(0)
    
    if logger:
        logger.info(f"Epoch {epoch} Good similarities: {good_similarities}")
        logger.info(f"Epoch {epoch} Bad similarities: {bad_similarities}")

    mean_sim_good = np.mean(good_similarities) if good_similarities else None
    mean_sim_bad = np.mean(bad_similarities) if bad_similarities else None
    
    # Find the range where majority of values fall instead of using mean
    if len(good_similarities) > 0 and len(bad_similarities) > 0 and epoch > 2 and mean_sim_bad > 0:
        # Combine all similarity values
        all_similarities = good_similarities + bad_similarities
        all_similarities = [sim for sim in all_similarities if not np.isnan(sim)]  # Remove NaN values
        
        if len(all_similarities) > 0:
            # Calculate the range containing the majority of values (e.g., 25th to 75th percentile)
            q25 = np.percentile(all_similarities, 25)
            q75 = np.percentile(all_similarities, 75)
            # Use the range width as the outlier detection threshold
            # This represents the spread where most values fall
            majority_range = q75 - q25 - 0.05
            
            # Alternative: use a percentile-based threshold (e.g., 10th percentile as lower bound)
            # majority_threshold = np.percentile(all_similarities, 10)
            if majority_range > 0.0001:
                selector.update_outlier_detection(majority_range, logger=logger)
                if logger:
                    logger.info(f"Similarity range (Q25-Q75): {q25:.4f} - {q75:.4f}, Range width: {majority_range:.4f}")
        else:
            if logger:
                logger.warning("No valid similarity values found for threshold update")
    
    if logger:
        logger.info(f"Updated outlier range to: {selector.get_threshold():.4f}")

    # Prepare base round data
    row_data = [
        epoch + 1,  # Round (1-indexed)
        len(dag.nodes),  # Total nodes
        len(dag.tips),   # Total tips
        num_good_removed,  # Good nodes removed this round
        num_malicious_confirmed_this_round,  # Malicious nodes confirmed this round
        num_corrupted_confirmed_this_round,   # Corrupted nodes confirmed this round
        total_good_removed,  # Total good removed
        (total_malicious_confirmed + checkpoint_malicious_conf),  # Total malicious confirmed
        (total_corrupted_confirmed + checkpoint_corrupted_conf),   # Total corrupted confirmed
        similarity_threshold, mean_sim_good, mean_sim_bad
    ]
    
    # Add data for each peer
    for i in range(X):
        trust_val = trust_vector[i] if i < len(trust_vector) else 0.0
        struct_consistency = structural_consistencies[i] if i < len(structural_consistencies) else 1.0
        updates_contrib = updates_contributions[i] if i < len(updates_contributions) else 1.0
        is_malicious = 1 if i in malicious_indices else 0
        is_corrupted = 1 if i in corrupted_indices else 0
        
        row_data.extend([
            trust_val, struct_consistency, updates_contrib, is_malicious, is_corrupted
        ])
    
    # Write data to CSV
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)
    
    if logger:
        logger.info(f"Round {epoch + 1} data saved to {csv_filename}")


def init_DAG(initial_params, logger):
    global dag, selector, peers, central_node, malicious_indices, num_malicious, num_corrupted, corrupted_indices, last_node_prev_round, pos, checkpoint_corrupted_conf, checkpoint_malicious_conf
    
    # Initialize DAG and tip selector
    dag = DAG.DAG()
    selector = TipSelectionAlgorithm.TrustInformedTipSelector_NoThreshold(
        similarity_method='cosine',
    )
    # Create X peers
    peers = [Peer.Peer(dag, selector, name=f"Peer{i}", id=i) for i in range(X)]
    central_node = Peer.Peer(dag, selector, name="CentralNode", id=X)  # Central node for aggregation

    # Randomly select malicious peers based on percentage
    num_malicious = int(X * malicious_percentage)
    malicious_indices = np.random.choice(X, num_malicious, replace=False)
    for idx in malicious_indices:
        peers[idx].malicious = True

    # Randomly select corrupted peers based on percentage
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
            if logger:
                logger.warning(f"Could not create genesis parameters from template: {e}")
            # Fallback to simple matrix if parameter_utils not available
            genesis_params = dag_utils.generate_data_matrix(malicious=False, corrupted=False)
    else:
        # Fallback to simple matrix
        genesis_params = dag_utils.generate_data_matrix(malicious=False, corrupted=False)
    
    genesis = DAGNode.DAGNode(genesis_params, peer=central_node, epoch=0)
    central_node.manual_add_node(genesis, [],logger=logger)
    pos = nx.spring_layout(dag.graph, seed=42)  # initial layout
    dag_utils.draw_dag(dag, step=0, pos=pos)
    last_node_prev_round = genesis
    
    if logger:
        logger.info("INITIALIZED GRAPH")
    
    # Initialize CSV logging
    init_csv_logging(logger=logger)
    
    return X,R,num_malicious,malicious_indices.tolist()


def get_trust_vector(parameters_clients, epoch, logger):
    global dag, selector, peers, longest_ppath, step, added_node_ids, total_good_removed, total_malicious_confirmed, total_corrupted_confirmed, trust_vector, structural_consistencies, updates_contributions, last_node_prev_round, pos, checkpoint_malicious_conf, checkpoint_corrupted_conf
    
    # Take the actual neural network parameters and put them in the DAG nodes
    for peer in peers:
        if peer.id in parameters_clients.keys():
            print("\nNODE NUMBER " + str(peer.id))
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
            if is_corrupted:
                # For corrupted peers, same approach
                print("CORRUPTED")
                pass
                
            node = DAGNode.DAGNode(actual_parameters, peer=peer, epoch=epoch)
            node.malicious = is_malicious
            node.corrupted = is_corrupted
            
            node_id = peer.add_node(node,logger=logger)
            added_node_ids.append(node_id)
            pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
            dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
            step += 1

    if logger:
        logger.info(f"\n\nEND OF ROUND {epoch+1}\nTotal nodes: {len(dag.nodes)} - Total tips: {len(dag.tips)}")
    
    old_path = set(longest_ppath)
    # Heaviest subgraph computations
    unconfirmed_tips = [tip for tip in dag.nodes if not dag.is_confirmed(tip)]
    
    try:
        longest_ppath = compute_longest_path(dag, selector, unconfirmed_tips, last_node_prev_round, old_path, logger)
        dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
        step += 1
    except Exception as e:
        if logger:
            logger.error(f"Error computing longest path: {e}")
        longest_ppath = old_path  # Fallback to previous path

    # Before removing, gather info on malicious/corrupted nodes confirmed and good nodes eliminated
    num_malicious_confirmed = sum(1 for n in longest_ppath if getattr(dag.nodes.get(n, None), "malicious", False))
    num_corrupted_confirmed = sum(1 for n in longest_ppath if getattr(dag.nodes.get(n, None), "corrupted", False))
    nodes_to_remove = [n for n in list(dag.graph.nodes) if n not in longest_ppath]
    num_good_removed = sum(1 for n in nodes_to_remove if not getattr(dag.nodes.get(n, None), "malicious", False) and not getattr(dag.nodes.get(n, None), "corrupted", False))
    num_malicious_confirmed_this_round = num_malicious_confirmed - total_malicious_confirmed
    num_corrupted_confirmed_this_round = num_corrupted_confirmed - total_corrupted_confirmed
    
    if logger:
        logger.info(f"Malicious nodes confirmed: {num_malicious_confirmed_this_round}")
        logger.info(f"Corrupted nodes confirmed: {num_corrupted_confirmed_this_round}")
        logger.info(f"Good nodes eliminated: {num_good_removed}")
    
    # Develop trust vector for each peer based on structural consistency and updates contributions
    for i, peer in enumerate(peers):
        structural_consistency = 1  # Default to trust
        updates_contribution = 1  # Default to trust
        # Find the node created by this specific peer in the longest path
        peer_node_in_path = None
        for node_id in longest_ppath:
            if node_id in dag.nodes:  # Safety check
                node = dag.nodes[node_id]
                if hasattr(node, 'peer') and node.peer == peer and node.epoch == epoch:
                    peer_node_in_path = node_id
                    break
        
        # If this peer has a node in the longest path and last FL round, check its past cone similarity
        if peer_node_in_path is not None:
            try:
                nodes_outside = selector.check_past_cone_similarity([peer_node_in_path], mean_sim_good, dag, logger=logger)
                if (dag.nodes[peer_node_in_path].malicious or dag.nodes[peer_node_in_path].corrupted) and len(nodes_outside) == 0:
                    logger.error(f"Peer {i} : MALICIOUS OR CORRUPTED, BUT NODES_OUTSIDE: {len(nodes_outside)}")
                if len(nodes_outside) > 0:
                    structural_consistency = 0  # Corrupted behavior detected in this peer's node past cone
                    if logger:
                        logger.info(f"Peer {i} : Corrupted behavior detected in node {peer_node_in_path[:6]}")
            except Exception as e:
                if logger:
                    logger.error(f"Error checking past cone for peer {i}: {e}")
                structural_consistency = 0.5  # Conservative default
        else:
            # If peer has no node in the longest path, check if they have any node in the graph
            peer_has_node = False
            for node_id in list(dag.nodes.keys()):  # Create a list copy to avoid concurrent modification
                if node_id in dag.nodes:  # Safety check
                    node = dag.nodes[node_id]
                    if hasattr(node, 'peer') and node.peer == peer and node.epoch == epoch:
                        peer_has_node = True
                        # The update was outside the heaviest subgraph, so updates contributions is 0
                        if logger:
                            logger.info(f"Peer {i} : Node outside heaviest graph in node {node_id[:6]} (not in longest path)")
                        updates_contribution = 0
                        # Check this peer's node even if not in longest path
                        try:
                            nodes_outside = selector.check_past_cone_similarity([node_id], mean_sim_good, dag, logger=logger)
                            if (dag.nodes[node_id].malicious or dag.nodes[node_id].corrupted) and len(nodes_outside) == 0:
                                logger.error(f"Peer {i} : MALICIOUS OR CORRUPTED, BUT NODES_OUTSIDE: {len(nodes_outside)}")
                            if len(nodes_outside) > 0:
                                structural_consistency = 0  # Corrupted behavior detected
                                if logger:
                                    logger.info(f"Peer {i} : Corrupted behavior detected outside heaviest graph in node {node_id[:6]} (not in longest path)")
                        except Exception as e:
                            if logger:
                                logger.error(f"Error checking past cone for peer {i} outside path: {e}")
                            structural_consistency = 0.5  # Conservative default
                        break
            
            if not peer_has_node:
                if logger:
                    logger.info(f"Peer {i} : No node found in the graph for this FL round")
                    if i in parameters_clients.keys():
                        logger.error(f"Peer {i} : No node found in the graph for this FL round, but parameters were provided")
                # The peer didn't participate in this round, so we assume update contributions is 0.5
                updates_contribution = trust_vector[i] if i < len(trust_vector) else 0.5
        
        # Ensure vectors are properly sized
        while len(updates_contributions) <= i:
            updates_contributions.append(1.0)
        while len(structural_consistencies) <= i:
            structural_consistencies.append(1.0)
        while len(trust_vector) <= i:
            trust_vector.append(1.0)
            
        # Update existing values    
        updates_contributions[i] = updates_contribution
        structural_consistencies[i] = structural_consistency
        trust_value = upsilon_1 * structural_consistency + upsilon_2 * updates_contribution
        trust_value_t = (1-lambda_) * trust_value + lambda_ * trust_vector[i]  # Historical trust influence
        trust_vector[i] = trust_value_t
    
    if logger:
        logger.info(f"Structural Consistency vector: {structural_consistencies}")
        logger.info(f"Update Contributions vector: {updates_contributions}")
        logger.info(f"Trust vector: {trust_vector}")
    
    # Save round data to CSV
    save_round_data(epoch, num_good_removed, num_malicious_confirmed_this_round, num_corrupted_confirmed_this_round, logger=logger)
    
    # Remove nodes outside of the heaviest subgraph - do this safely
    nodes_removed = 0
    for n in nodes_to_remove:
        try:
            if n in dag.graph.nodes:
                dag.graph.remove_node(n)
                nodes_removed += 1
            if n in dag.nodes:
                dag.nodes.pop(n, None)
            dag.tips.discard(n)
        except Exception as e:
            if logger:
                logger.error(f"Error removing node {n}: {e}")
            continue
    
    if logger:
        logger.info(f"Successfully removed {nodes_removed} nodes")
    total_good_removed += num_good_removed
    total_malicious_confirmed = num_malicious_confirmed
    total_corrupted_confirmed = num_corrupted_confirmed
    
    # Pruning logic with safety checks
    if epoch > 0 and ((R//8) != 0) and (epoch % (R//8) == 0):
        # Streamline the graph by removing all nodes up to the last milestone
        if logger:
            logger.info(f"Pruning graph at epoch {epoch + 1}")
        
        # Find all nodes to remove (all except those from milestone onwards)
        nodes_to_prune = []
        # Find the last milestone (should be last_node_prev_round)
        if last_node_prev_round:
            # Find all nodes that precede the last milestone
            for node_id in list(dag.graph.nodes):
                if node_id in dag.nodes:  # Safety check
                    node = dag.nodes.get(node_id)
                    if node and node.id != last_node_prev_round.id:
                        # Check if this node is a predecessor of the last milestone
                        if hasattr(node, 'epoch') and node.epoch < last_node_prev_round.epoch:
                            nodes_to_prune.append(node_id)
        
        # Remove found nodes safely
        pruned_count = 0
        for node_id in nodes_to_prune:
            try:
                if node_id in dag.graph.nodes:
                    dag.graph.remove_node(node_id)
                    pruned_count += 1
                if node_id in dag.nodes:
                    dag.nodes.pop(node_id, None)
                dag.tips.discard(node_id)
                longest_ppath.discard(node_id)  # Remove from longest path set too
                if node_id in added_node_ids:
                    added_node_ids.remove(node_id)
            except Exception as e:
                if logger:
                    logger.error(f"Error pruning node {node_id}: {e}")
                continue
        
        checkpoint_malicious_conf += total_malicious_confirmed
        checkpoint_corrupted_conf += total_corrupted_confirmed
        total_malicious_confirmed = 0
        total_corrupted_confirmed = 0
        
        if logger:
            logger.info(f"Removed {pruned_count} nodes during graph pruning")
            logger.info(f"Remaining nodes: {len(dag.nodes)}")
    
    return trust_vector, updates_contributions


def milestone(new_nn_params, epoch, logger):
    global dag, central_node, longest_ppath, added_node_ids, step, last_node_prev_round, pos
    
    try:
        # CREATING MILESTONE
        last_node = None
        for node_id in longest_ppath:
            try:
                if node_id in dag.graph.nodes:  # Safety check
                    if all(parent not in longest_ppath for parent in dag.graph.predecessors(node_id)):
                        last_node = node_id
                        break
            except Exception as e:
                if logger:
                    logger.error(f"Error accessing predecessors for node {node_id}: {e}")
                last_node = node_id
                continue
        
        if last_node is not None and last_node in dag.nodes:
            last_node_prev_round = dag.nodes[last_node]
        
        # Use the actual aggregated neural network parameters for the milestone
        milestone_node = DAGNode.DAGNode(new_nn_params, peer=central_node, epoch=epoch)
        
        # Safely add milestone
        if last_node_prev_round:
            central_node.manual_add_node(new_node=milestone_node, selected_parents=[last_node_prev_round.id],logger=logger)
        else:
            # Fallback if no previous milestone found
            central_node.manual_add_node(new_node=milestone_node, selected_parents=[],logger=logger)

        longest_ppath.add(milestone_node.id)
        if logger:
            logger.info("NEW MILESTONE: " + str(milestone_node.id))
            logger.info("\n\n")
        added_node_ids.append(milestone_node.id)
        last_node_prev_round = milestone_node
        
        
        pos = nx.spring_layout(dag.graph, seed=42, pos=pos, fixed=list(pos.keys()))
        dag_utils.draw_dag(dag, step=step, pos=pos, highlight_nodes=longest_ppath)
        step += 1
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating milestone: {e}")
        return


def compute_longest_path(dag, selector, unconfirmed_tips, last_node_prev_round, old_path, logger):
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
    
    try:
        for tip in unconfirmed_tips:
            if tip not in dag.nodes:  # Safety check
                continue
                
            # Check past cone similarity
            try:
                nodes_to_remove = selector.check_past_cone_similarity([tip], mean_sim_good, dag)
                if logger:
                    logger.info(f"Tip {tip} past cone similarity check: {nodes_to_remove}")
                if len(nodes_to_remove) == 0:
                    if last_node_prev_round:
                        ppath = dag_utils.search_path_from_tip_to_lastroundFL(dag, tip, last_node_prev_round.id)
                        if len(ppath) > len(longest_ppath):
                            longest_ppath = ppath
                            if logger:
                                logger.info("Found new longest path from tip: " + str(tip))
                else:
                    if logger:
                        logger.info("CORRUPTED")
            except Exception as e:
                if logger:
                    logger.error(f"Error evaluating tip {tip}: {e}")
                continue

        # Union with old path - safely
        if old_path:
            # Only include nodes that still exist in the DAG
            valid_old_nodes = {node for node in old_path if node in dag.nodes}
            longest_ppath = longest_ppath.union(valid_old_nodes)
        
    except Exception as e:
        if logger:
            logger.error(f"Error in compute_longest_path: {e}")
        # Fallback to old path if available
        if old_path:
            longest_ppath = {node for node in old_path if node in dag.nodes}
    
    return longest_ppath