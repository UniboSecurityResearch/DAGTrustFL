import DAG
import numpy as np
import random
import parameter_utils

class TipSelectionAlgorithm:
    def select_tips(self, dag: DAG) -> list:
        """To be implemented by specific algorithms."""
        raise NotImplementedError

class RandomTipSelector(TipSelectionAlgorithm):
    def select_tips(self, dag: DAG) -> list:
        tips = dag.get_tips()
        if len(tips) >= 2:
            return random.sample(tips, 2)
        elif tips:
            return [tips[0], tips[0]]
        else:
            return []

class IOTALikeTipSelector:
    def select_tips_old(self, dag: DAG) -> list:
        tips = dag.get_tips()
        unconfirmed = [t for t in tips if not dag.is_confirmed(t)]
        all_nodes = list(dag.graph.nodes())
        if len(unconfirmed) >= 2:
            return random.sample(unconfirmed, 2)
        elif len(unconfirmed) == 1:
            fallback_candidates = [n for n in all_nodes if n != unconfirmed[0]]
            if fallback_candidates:
                fallback = random.choice(fallback_candidates)
            else:
                fallback = unconfirmed[0]  # only possible choice
            return [unconfirmed[0], fallback]
        else:
            # if all tips are confirmed or none are unconfirmed
            if len(all_nodes) >= 2:
                return random.sample(all_nodes, 2)
            else:
                return [all_nodes[0], all_nodes[0]]  # safe fallback

    def select_tips(self, dag: DAG) -> list:
        dag.update_cumulative_weights()
        # Consider all nodes in the DAG that are not yet confirmed (CW <= threshold)
        unconfirmed = [n for n in dag.graph.nodes if not dag.is_confirmed(n)]
        if len(unconfirmed) >= 2:
            return random.sample(unconfirmed, 2)
        elif len(unconfirmed) == 1:
            fallback_candidates = [n for n in dag.graph.nodes if n != unconfirmed[0]]
            if fallback_candidates:
                fallback = random.choice(fallback_candidates)
            else:
                fallback = unconfirmed[0]  # only possible choice
            return [unconfirmed[0], fallback]
        else:
            # all nodes are confirmed → fallback to any nodes
            all_nodes = list(dag.graph.nodes)
            if len(all_nodes) >= 2:
                return random.sample(all_nodes, 2)
            else:
                return [all_nodes[0], all_nodes[0]]

class OLD_MyCustomTipSelector(TipSelectionAlgorithm):

    def __init__(self, max_depth=4, similarity_threshold=0.4, cw_penalty_factor=0.05):
        self.max_depth = max_depth #chi, depth of the past cone to inspect
        self.similarity_threshold = similarity_threshold
        self.cw_penalty_factor = cw_penalty_factor  # Factor to penalize high CW nodes

    # Cosine similarity between two matrices - also considers magnitude!
    # This method is used in Zhu et Al. (2024)
    def cosine_similarity(self, m1, m2):
        v1 = m1.flatten()
        v2 = m2.flatten()
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # Check if either norm is zero to avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0  # Zero similarity if one of the vectors is null
        # Standard cosine similarity
        cosine_sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
        # Penalize magnitude differences
        magnitude_penalty = 1 - abs(norm_v1 - norm_v2) / max(norm_v1, norm_v2)
        # Return a combination of cosine similarity and magnitude penalty
        return cosine_sim * magnitude_penalty
    
    def euclidean_distance(self, m1, m2):
        #TODO
        v1 = m1.flatten()
        v2 = m2.flatten()
        return np.linalg.norm(v1 - v2)

    # Evaluate similarity score recursively up to a maximum depth
    # It evaluates also the past cone of the transaction, to check for
    # corrupted behaviors
    # But in a different way than TrustInformedTipSelector
    # It uses a weighted average of the similarities, where the weight is based on depth
    # and cumulative weight (CW) of the nodes
    # The similarity is calculated using cosine similarity, and nodes with high CW are penalized
    # to avoid selecting nodes that are too heavy in the DAG.
    # So the check for corrupted behavior, it only consider its matrix with all the past cone
    # It does not compare the matrix of parents node with their own parents!
    def evaluate_similarity(self, node_id, dag: DAG, new_matrix):
        visited = set()
        queue = [(node_id, 0)]
        weighted_similarities = []
        total_weights = 0

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > self.max_depth:
                continue
            visited.add(current)
            current_matrix = dag.nodes[current].data

            # Calculate similarity between the new matrix and the current node's matrix
            sim = self.cosine_similarity(new_matrix, current_matrix)

            # Penalize nodes with high CW
            cw_penalty = 1 / (1 + self.cw_penalty_factor * dag.nodes[current].cw)
            if not np.isnan(sim):
                weight = 2.0 if depth == 0 else (1 / (1 + depth**2))
                if sim < self.similarity_threshold:
                    print(f"Node {current} excluded due to low similarity")
                    continue
                weighted_similarity = weight * sim * cw_penalty
                weighted_similarities.append(weighted_similarity)
                total_weights += weight
            else:
                print(f"Similarity is NaN for node {current} at depth {depth}")
            for parent in dag.graph.successors(current):
                # Traverse parents (successors in this graph structure)
                queue.append((parent, depth + 1))
        # Calculate the weighted average similarity
        return sum(weighted_similarities) / total_weights if total_weights > 0 else 0

    def select_tips(self, dag: DAG, new_matrix, corrupted=False) -> list:
        # Step 1: find all tips (CW < threshold)
        dag.update_cumulative_weights()
        candidate_tips = [
            n for n in dag.graph.nodes
            if not dag.is_confirmed(n)
        ]
        #print(f"Candidate tips: {candidate_tips}")
        # Step 2: calculate similarity (including parents) for each tip
        scored = []
        for tip in candidate_tips:
            score = self.evaluate_similarity(tip, dag, new_matrix)
            #print(f"Tip: {tip}, Score: {score}")
            if score is not None:
                scored.append((tip, score))
        # Sort by descending score
        if corrupted:
            # If corrupted, the order is reversed to prioritize lower scores, i.e. malicious nodes
            scored.sort(key=lambda x: x[1], reverse=False)
            # Lower threshold for corrupted nodes
            # Step 3: select the worst unconfirmed nodes
            selected = [tip for tip, _ in scored if _ >= 0.0001][:2]
        else:
            scored.sort(key=lambda x: x[1], reverse=True)
            # Step 3: select the best unconfirmed nodes
            selected = [tip for tip, _ in scored if _ >= self.similarity_threshold][:2]
        # Step 4: if less than 2, add fallback from confirmed nodes with CW > threshold
        if len(selected) < 2:
            confirmed_nodes = [
                n for n in dag.graph.nodes
                if dag.is_confirmed(n) and n not in selected
            ]
            # Filter confirmed nodes for those that are similar
            confirmed_scores = []
            for node in confirmed_nodes:
                score = self.evaluate_similarity(node, dag, new_matrix)
                if score is not None:
                    confirmed_scores.append((node, score))
            if corrupted:
                # If corrupted, we want to select the worst confirmed nodes
                confirmed_scores.sort(key=lambda x: x[1], reverse=False)
            else:
                # If not corrupted, we want to select the best confirmed nodes      
                confirmed_scores.sort(key=lambda x: x[1], reverse=True)
            #print(f"Confirmed nodes: {confirmed_scores}")
            # Add the most similar confirmed nodes
            selected += [node for node, _ in confirmed_scores[:2 - len(selected)]]
        # Step 5: if still less than 2, pick any other node as final fallback
        if len(selected) < 2:
            backup_nodes = [n for n in dag.graph.nodes if n not in selected]
            random.shuffle(backup_nodes)
            selected += backup_nodes[:2 - len(selected)]
        return selected[:2]
    
    def check_past_cone_similarity(self, selected_nodes, dag: DAG, debug=0):
        """
        Check if any node in the past cone of selected nodes has low similarity with its parents.
        Returns a list of nodes that should be removed from selection.
        """
        nodes_to_remove = []
        for node_id in selected_nodes:
            visited = set()
            queue = [(node_id, 0)]
            should_remove_node = False
            
            while queue and not should_remove_node:
                current, depth = queue.pop(0)
                if current in visited or depth > self.max_depth:
                    continue
                visited.add(current)
                current_matrix = dag.nodes[current].data
                
                # Get parent nodes of current node
                parent_nodes = list(dag.graph.successors(current))
                
                # Compare current node's matrix with each parent's matrix
                for parent in parent_nodes:
                    parent_matrix = dag.nodes[parent].data
                    sim = self.cosine_similarity(current_matrix, parent_matrix)
                    
                    if not np.isnan(sim) and sim < self.similarity_threshold:
                        if debug == 1:
                            print(f"Node {node_id} excluded due to low similarity ({sim:.4f}) between node {current} and its parent {parent}")
                        should_remove_node = True
                        break
                
                # Continue traversing to parents for next iteration
                for parent in parent_nodes:
                    queue.append((parent, depth + 1))
            
            if should_remove_node:
                nodes_to_remove.append(node_id)
        
        return nodes_to_remove



class TrustInformedTipSelector(TipSelectionAlgorithm):
    #TODO:
    # make the similarity threshold variable based on the epoch
    def __init__(self, max_depth=3, similarity_threshold=0.30, cw_penalty_factor=0.05, similarity_method='cosine'):
        self.max_depth = max_depth # depth of the past cone to inspect
        self.similarity_threshold = similarity_threshold  # Optimized from extended FL testing
        self.cw_penalty_factor = cw_penalty_factor  # Factor to penalize high CW nodes
        self.similarity_method = similarity_method  # Method for similarity calculation

    def update_threshold(self, new_threshold, logger=None):
        """
        Update the similarity threshold dynamically.
        
        :param new_threshold: New threshold value
        :type new_threshold: float
        """
        self.similarity_threshold = new_threshold
        if logger:
            logger.info(f"Updated similarity threshold to {self.similarity_threshold}")
        else:
            if logger:
                logger.info(f"Updated similarity threshold to {self.similarity_threshold}")

    def get_threshold(self):
        """
        Get the current similarity threshold.
        
        :return: Current threshold value
        :rtype: float
        """
        return self.similarity_threshold

    def cs_evaluate_similarity(self, node_id, dag: DAG, new_parameters, logger=None):
        current_parameters = dag.nodes[node_id].data
        if dag.nodes[node_id].peer.name == "CentralNode":
            return 1.0 - (self.cw_penalty_factor * dag.nodes[node_id].cw)
        else:
            return (parameter_utils.cos_sim_simple(current_parameters, new_parameters) - (self.cw_penalty_factor * dag.nodes[node_id].cw))

    def evaluate_similarity(self, node_id, dag: DAG, new_parameters, logger=None):
        weight=1
        cw_penalty = 1
        # Calculate similarity between the new parameters and the current node's parameters
        current_parameters = dag.nodes[node_id].data
        
        # Debug: log information about the parameters
        if logger:
            logger.debug(f"Evaluating similarity for node {node_id}")
            logger.debug(f"  Node peer name: {dag.nodes[node_id].peer.name}")
        
        if type(new_parameters) != type(current_parameters):
            if logger:
                logger.warning(f"new_parameters and current_parameters are not of the same type: {type(new_parameters)} vs {type(current_parameters)}")
            else:
                if logger:
                    logger.warning(f"  WARNING: new_parameters and current_parameters are not of the same type: {type(new_parameters)} vs {type(current_parameters)}")
            # Return a neutral score for incompatible data types
            return 0.5
        
        # Check if both parameters are neural network parameters
        if hasattr(new_parameters, '__len__') and hasattr(current_parameters, '__len__'):
            if hasattr(new_parameters, 'shape') and logger:
                logger.debug(f"  New parameters shape: {new_parameters.shape}")
            if hasattr(current_parameters, 'shape') and logger:
                logger.debug(f"  Current parameters shape: {current_parameters.shape}")
        
        # Check if current_parameters looks like a neural network parameter set
        if not isinstance(current_parameters, (dict, type(new_parameters))):
            if logger:
                logger.warning(f"current_parameters is not a dict or same type as new_parameters")
                logger.debug(f"  Current parameters content preview: {str(current_parameters)[:100]}...")
            else:
                if logger:
                    logger.warning(f"  WARNING: current_parameters is not a dict or same type as new_parameters")
                    logger.debug(f"  Current parameters content preview: {str(current_parameters)[:100]}...")
            # Return a neutral score for incompatible data types
            return 0.5
        
        sim = parameter_utils.cos_sim_simple(new_parameters, current_parameters)
        weighted_similarity = weight * sim * cw_penalty
        return weighted_similarity if not np.isnan(sim) else -1
    
    def check_past_cone_similarity(self, selected_nodes, dag: DAG, debug=1, logger=None):
        """
        Check if any node in the past cone of selected nodes has low similarity with its parents (corrupted behavior).
        Returns a list of nodes that should be removed from selection.
        """
        nodes_to_remove = []
        for node_id in selected_nodes:
            visited = set()
            queue = [(node_id, 0)]
            should_remove_node = False
            if dag.nodes[node_id].peer.name != "CentralNode":
                while queue and not should_remove_node:
                    current, depth = queue.pop(0)
                    if current in visited or depth > self.max_depth:
                        continue
                    visited.add(current)
                    current_parameters = dag.nodes[current].data
                    # Get parent nodes of current node
                    parent_nodes = list(dag.graph.successors(current))
                    if logger:
                        logger.debug(f"NAME: {dag.nodes[current].peer.name}")
                    else:
                        if logger:
                            logger.debug(f"NAME: {dag.nodes[current].peer.name}")
                    if dag.nodes[current].peer.name != "CentralNode":
                        # Compare current node's parameters with each parent's parameters
                        for parent in parent_nodes:
                            if(dag.nodes[parent].peer.name == "CentralNode"):
                                # Skip the central node, as it is not a peer and does not have parameters
                                sim=1.0
                            else:
                                parent_parameters = dag.nodes[parent].data
                                sim = parameter_utils.cos_sim_simple(current_parameters,parent_parameters)
                                if not np.isnan(sim) and sim < self.similarity_threshold:
                                    if debug == 1:
                                        if logger:
                                            logger.debug(f"Node {node_id} excluded due to low similarity ({sim:.4f}) between node {current} and its parent {parent}")
                                        else:
                                            if logger:
                                                logger.debug(f"Node {node_id} excluded due to low similarity ({sim:.4f}) between node {current} and its parent {parent}")
                                    should_remove_node = True
                                    break
                
                # Continue traversing to parents for next iteration
                for parent in parent_nodes:
                    queue.append((parent, depth + 1))
            
            if should_remove_node:
                nodes_to_remove.append(node_id)
        
        return nodes_to_remove

    def select_tips(self, dag: DAG, new_parameters, corrupted=False, logger=None) -> list:
        # Step 1: find all tips (CW < Theta)
        dag.update_cumulative_weights()
        candidate_tips = [
            n for n in dag.graph.nodes
            if not dag.is_confirmed(n)
        ]
        
        # Step 2: calculate similarity for each tip
        scored = []
        selected = []
        for tip in candidate_tips:
            score = self.cs_evaluate_similarity(tip, dag, new_parameters, logger)
            if score is not None:
                scored.append((tip, score))
        
        if corrupted:
            # If corrupted, the order is reversed to prioritize lower scores, i.e. malicious nodes
            scored.sort(key=lambda x: x[1], reverse=False)
            # Lower threshold for corrupted nodes
            # Step 3: select the worst unconfirmed nodes
            selected = [tip for tip, _ in scored if _ >= 0.0001][:2]
        else:
            # Sort by descending score
            scored.sort(key=lambda x: x[1], reverse=True)
            if logger:
                logger.info(f"Scored tips: {scored}")
            else:
                if logger:
                    logger.info(f"Scored tips: {scored}")
            
            # Check past cone for all scored nodes and remove those with violations
            filtered_scored = []
            for tip, score in scored:
                nodes_to_remove = self.check_past_cone_similarity([tip], dag, logger=logger)
                if len(nodes_to_remove) == 0:
                    # No violations found, keep this tip
                    filtered_scored.append((tip, score))
                else:
                    # Violations found, remove this tip
                    if logger:
                        logger.info(f"Removing tip {tip} due to {len(nodes_to_remove)} past cone violations")
                    else:
                        if logger:
                            logger.info(f"Removing tip {tip} due to {len(nodes_to_remove)} past cone violations")
            
            # Update scored with filtered results
            scored = filtered_scored
            if logger:
                logger.info(f"Scored tips after past cone filtering: {scored}")
            else:
                if logger:
                    logger.info(f"Scored tips after past cone filtering: {scored}")
            
            # Step 3: select the best nodes (those that passed past cone check)
            selected = [tip for tip, _ in scored[:2]]
            if logger:
                logger.info(f"Selected after past cone check: {selected}")
            else:
                if logger:
                    logger.info(f"Selected after past cone check: {selected}")
        # Step 5: if less than 2, add fallback considering also confirmed nodes
        if len(selected) < 2:
            if logger:
                logger.info(f"FALLBACK FROM CONFIRMED NODES {2 - len(selected)}")
            else:
                if logger:
                    logger.info(f"FALLBACK FROM CONFIRMED NODES {2 - len(selected)}")
            confirmed_nodes = [
                n for n in dag.graph.nodes
                if dag.is_confirmed(n) and n not in selected
            ]
            # Filter confirmed nodes for those that are similar
            confirmed_scores = []
            for node in confirmed_nodes:
                if logger:
                    logger.debug(f"Checking confirmed node: {node}")
                else:
                    if logger:
                        logger.info(f"Checking confirmed node: {node}")

                # Quick compatibility check before calling evaluate_similarity
                try:
                    current_parameters = dag.nodes[node].data
                    # Check if the data looks like neural network parameters
                    if not isinstance(current_parameters, (dict, type(new_parameters))):
                        if logger:
                            logger.warning(f"Skipping node {node} - incompatible data type: {type(current_parameters)}")
                        else:
                            if logger:
                                logger.warning(f"  Skipping node {node} - incompatible data type: {type(current_parameters)}")
                        continue
                except Exception as e:
                    if logger:
                        logger.error(f"Error accessing data for node {node}: {e}")
                    else:
                        if logger:
                            logger.error(f"  Error accessing data for node {node}: {e}")
                    continue
                
                score = self.cs_evaluate_similarity(node, dag, new_parameters, logger)
                if score is not None:
                    confirmed_scores.append((node, score))
            if corrupted:
                # If corrupted, we want to select the worst confirmed nodes
                confirmed_scores.sort(key=lambda x: x[1], reverse=False)
                selected += [node for node, _ in confirmed_scores[:2 - len(selected)]]
            else:
                # If not corrupted, we want to select the best confirmed nodes and check their past cone
                new_selected = []    
                confirmed_scores.sort(key=lambda x: x[1], reverse=True)
                if logger:
                    logger.info(f"Confirmed nodes: {confirmed_scores}")
                else:
                    if logger:
                        logger.info(f"Confirmed nodes: {confirmed_scores}")
                repeat = True
                while len(scored)> 2 and repeat:
                    # Step 3: select the best unconfirmed nodes
                    repeat = False
                    new_selected = [node for node, _ in confirmed_scores[:2 - len(selected)]]
                    # Step 4: Examine the past cone of the selected nodes
                    # Check if any node in the past cone has low similarity with its parents
                    nodes_to_remove = self.check_past_cone_similarity(new_selected, dag, logger=logger)
                    # Remove nodes that failed the similarity check (with corrupted behavior)
                    for node_id in nodes_to_remove:
                        if node_id in new_selected:
                            new_selected.remove(node_id)
                            repeat = True
                            # Remove the tuple (node_id, score) from confirmed_scores list
                            confirmed_scores = [(tip, score) for tip, score in confirmed_scores if tip != node_id]
                if logger:
                    logger.info(f"New selected: {new_selected}")
                else:
                    if logger:
                        logger.info(f"New selected: {new_selected}")
                selected += new_selected
        # Step 5: if still 0, pick any other node as final fallback
        if len(selected) < 1:
            backup_nodes = [n for n in dag.graph.nodes if n not in selected]
            random.shuffle(backup_nodes)
            selected += backup_nodes[:1 - len(selected)]
        if logger:
            logger.info(f"Selected tips: {selected}")
        else:
            if logger:
                logger.info(f"Selected tips: {selected}")
        return selected[:2]

class TrustInformedTipSelector_NoThreshold(TipSelectionAlgorithm):
    #TODO:
    # make the similarity threshold variable based on the epoch
    def __init__(self, max_depth=3, outlier_detection=0.10, cw_penalty_factor=0.02, similarity_method='cosine'):
        self.max_depth = max_depth # depth of the past cone to inspect
        self.outlier_detection = outlier_detection  # Optimized from extended FL testing
        self.cw_penalty_factor = cw_penalty_factor  # Factor to penalize high CW nodes
        self.similarity_method = similarity_method  # Method for similarity calculation

    def update_outlier_detection(self, new_threshold, logger=None):
        """
        Update the similarity threshold dynamically.
        
        :param new_threshold: New threshold value
        :type new_threshold: float
        """
        self.outlier_detection = new_threshold
        if logger:
            logger.info(f"Updated similarity threshold to {self.outlier_detection}")
        else:
            if logger:
                logger.info(f"Updated similarity threshold to {self.outlier_detection}")

    def get_threshold(self):
        """
        Get the current similarity threshold.
        
        :return: Current threshold value
        :rtype: float
        """
        return self.outlier_detection

    def cs_evaluate_similarity(self, node_id, dag: DAG, new_parameters, logger=None):
        current_parameters = dag.nodes[node_id].data
        if dag.nodes[node_id].peer.name == "CentralNode":
            return 0.8 - (self.cw_penalty_factor * dag.nodes[node_id].cw)
        else:
            sim = parameter_utils.cos_sim_simple(current_parameters, new_parameters)
            if dag.nodes[node_id].epoch < dag.current_epoch:
                return (sim - (self.cw_penalty_factor * (dag.current_epoch - dag.nodes[node_id].epoch + 1) * dag.nodes[node_id].cw))
            else:
                return (sim - (self.cw_penalty_factor * dag.nodes[node_id].cw))

    def evaluate_similarity(self, node_id, dag: DAG, new_parameters, logger=None):
        weight=1
        cw_penalty = 1
        # Calculate similarity between the new parameters and the current node's parameters
        current_parameters = dag.nodes[node_id].data
        
        # Debug: log information about the parameters
        if logger:
            logger.debug(f"Evaluating similarity for node {node_id}")
            logger.debug(f"  Node peer name: {dag.nodes[node_id].peer.name}")
        
        if type(new_parameters) != type(current_parameters):
            if logger:
                logger.warning(f"new_parameters and current_parameters are not of the same type: {type(new_parameters)} vs {type(current_parameters)}")
            else:
                if logger:
                    logger.warning(f"  WARNING: new_parameters and current_parameters are not of the same type: {type(new_parameters)} vs {type(current_parameters)}")
            # Return a neutral score for incompatible data types
            return 0.5
        
        # Check if both parameters are neural network parameters
        if hasattr(new_parameters, '__len__') and hasattr(current_parameters, '__len__'):
            if hasattr(new_parameters, 'shape') and logger:
                logger.debug(f"  New parameters shape: {new_parameters.shape}")
            if hasattr(current_parameters, 'shape') and logger:
                logger.debug(f"  Current parameters shape: {current_parameters.shape}")
        
        # Check if current_parameters looks like a neural network parameter set
        if not isinstance(current_parameters, (dict, type(new_parameters))):
            if logger:
                logger.warning(f"current_parameters is not a dict or same type as new_parameters")
                logger.debug(f"  Current parameters content preview: {str(current_parameters)[:100]}...")
            else:
                if logger:
                    logger.warning(f"  WARNING: current_parameters is not a dict or same type as new_parameters")
                    logger.debug(f"  Current parameters content preview: {str(current_parameters)[:100]}...")
            # Return a neutral score for incompatible data types
            return 0.5
        
        sim = parameter_utils.cos_sim_simple(new_parameters, current_parameters)
        weighted_similarity = weight * sim * cw_penalty
        return weighted_similarity if not np.isnan(sim) else -1
    
    def check_past_cone_similarity(self, selected_nodes, sim_comp, dag: DAG, debug=1, logger=None):
        """
        Check if any node in the past cone of selected nodes has low similarity with its parents (corrupted behavior).
        Returns a list of nodes that should be removed from selection.
        """
        nodes_to_remove = []
        for node_id in selected_nodes:
            visited = set()
            queue = [(node_id, 0)]
            should_remove_node = False
            if dag.nodes[node_id].peer.name != "CentralNode":
                while queue and not should_remove_node:
                    current, depth = queue.pop(0)
                    if current in visited or depth > self.max_depth:
                        continue
                    visited.add(current)
                    current_parameters = dag.nodes[current].data
                    # Get parent nodes of current node
                    parent_nodes = list(dag.graph.successors(current))
                    if len(parent_nodes) == 0:
                        logger.error(f"Node {current} has no parents, skipping similarity check")
                    if logger and debug == 1:
                        logger.debug(f"NAME: {dag.nodes[current].peer.name}")
                    if dag.nodes[current].peer.name != "CentralNode":
                        # Compare current node's parameters with each parent's parameters
                        for parent in parent_nodes:
                            if(dag.nodes[parent].peer.name == "CentralNode"):
                                # Skip the central node, as it is not a peer and does not have parameters
                                sim=1.0
                            else:
                                parent_parameters = dag.nodes[parent].data
                                sim = parameter_utils.cos_sim_simple(current_parameters,parent_parameters)
                                
                                # Check if current node is malicious or corrupted and log detailed info
                                current_node = dag.nodes[current]
                                if current_node.malicious:
                                    if logger:
                                        logger.warning(f"MALICIOUS NODE {current}: similarity with parent {parent} = {sim:.4f}, sim_comp = {sim_comp:.4f}, difference = {abs(sim - sim_comp):.4f}, outlier_threshold = {self.outlier_detection:.4f}")
                                elif current_node.corrupted:
                                    if logger:
                                        logger.warning(f"CORRUPTED NODE {current}: similarity with parent {parent} = {sim:.4f}, sim_comp = {sim_comp:.4f}, difference = {abs(sim - sim_comp):.4f}, outlier_threshold = {self.outlier_detection:.4f}")
                                
                                if logger and debug == 1:  
                                    logger.debug(f"Comparing node {current} with parent {parent}: similarity = {sim:.4f}, sim_comp = {sim_comp:.4f}, diff with sim_comp = {abs(sim - sim_comp):.4f} and outlier detection threshold = {self.outlier_detection:.4f}")
                                if not np.isnan(sim) and abs(sim - sim_comp) > self.outlier_detection:
                                    if debug == 1 and logger:
                                        logger.debug(f"Node {node_id} excluded due to similarity difference ({abs(sim - sim_comp):.4f}) between node {current} and its parent {parent} exceeding threshold")
                                    should_remove_node = True
                                    break
                
                # Continue traversing to parents for next iteration
                for parent in parent_nodes:
                    queue.append((parent, depth + 1))
            
            if should_remove_node:
                nodes_to_remove.append(node_id)
        
        return nodes_to_remove

    def select_tips(self, dag: DAG, new_parameters, corrupted=False, logger=None) -> list:
        # Step 1: find all tips (CW < Theta)
        dag.update_cumulative_weights()
        candidate_tips = [
            n for n in dag.graph.nodes
            if not dag.is_confirmed(n)
        ]
        
        # Step 2: calculate similarity for each tip
        scored = []
        selected = []
        for tip in candidate_tips:
            score = self.cs_evaluate_similarity(tip, dag, new_parameters, logger)
            if score is not None:
                scored.append((tip, score))
        
        if corrupted:
            # If corrupted, prioritize malicious updates
            for tip,score in scored:
                if dag.nodes[tip].malicious:
                    #Move to first in scored list
                    scored.remove((tip,score))
                    scored.insert(0, (tip,score))
            # Lower threshold for corrupted nodes
            # Step 3: select the worst unconfirmed nodes
            selected = [tip for tip, _ in scored if _ >= 0.0001][:2]
        else:
            # Sort by descending score
            scored.sort(key=lambda x: x[1], reverse=True)
            if logger:
                logger.info(f"Scored tips: {scored}")
            else:
                if logger:
                    logger.info(f"Scored tips: {scored}")
            
            # Check past cone for all scored nodes and remove those with violations
            filtered_scored = []
            for tip, score in scored:
                nodes_to_remove = self.check_past_cone_similarity([tip], (score + self.cw_penalty_factor * dag.nodes[tip].cw), dag, logger=logger)
                if len(nodes_to_remove) == 0:
                    # No violations found, keep this tip
                    filtered_scored.append((tip, score))
                else:
                    # Violations found, remove this tip
                    if logger:
                        logger.info(f"Removing tip {tip} due to {len(nodes_to_remove)} past cone violations")
                    else:
                        if logger:
                            logger.info(f"Removing tip {tip} due to {len(nodes_to_remove)} past cone violations")
            
            # Update scored with filtered results
            scored = filtered_scored
            if logger:
                logger.info(f"Scored tips after past cone filtering: {scored}")
            else:
                if logger:
                    logger.info(f"Scored tips after past cone filtering: {scored}")
            
            # Step 3: select the best nodes (those that passed past cone check)
            selected = [tip for tip, _ in scored[:2]]
        # Step 5: if less than 2, add fallback considering also confirmed nodes
        if len(selected) < 2:
            if logger:
                logger.info(f"FALLBACK FROM CONFIRMED NODES {2 - len(selected)}")
            else:
                if logger:
                    logger.info(f"FALLBACK FROM CONFIRMED NODES {2 - len(selected)}")
            confirmed_nodes = [
                n for n in dag.graph.nodes
                if dag.is_confirmed(n) and n not in selected
            ]
            # Filter confirmed nodes for those that are similar
            confirmed_scores = []
            for node in confirmed_nodes:
                if dag.current_epoch - dag.nodes[node].epoch < 3:
                    # Quick compatibility check before calling evaluate_similarity
                    try:
                        current_parameters = dag.nodes[node].data
                        # Check if the data looks like neural network parameters
                        if not isinstance(current_parameters, (dict, type(new_parameters))):
                            if logger:
                                logger.warning(f"Skipping node {node} - incompatible data type: {type(current_parameters)}")
                            else:
                                if logger:
                                    logger.warning(f"  Skipping node {node} - incompatible data type: {type(current_parameters)}")
                            continue
                    except Exception as e:
                        if logger:
                            logger.error(f"Error accessing data for node {node}: {e}")
                        else:
                            if logger:
                                logger.error(f"  Error accessing data for node {node}: {e}")
                        continue
                    
                    score = self.cs_evaluate_similarity(node, dag, new_parameters, logger)
                    if score is not None:
                        confirmed_scores.append((node, score))
            if corrupted:
                # If corrupted, prioritize malicious updates
                for tip,score in confirmed_scores:
                    if dag.nodes[tip].malicious:
                        #Move to first in scored list
                        confirmed_scores.remove((tip,score))
                        confirmed_scores.insert(0, (tip,score))
                selected += [node for node, _ in confirmed_scores[:2 - len(selected)]]
            else:
                # If not corrupted, we want to select the best confirmed nodes and check their past cone
                new_selected = []    
                confirmed_scores.sort(key=lambda x: x[1], reverse=True)
                #if logger:
                #    logger.info(f"Confirmed nodes: {confirmed_scores}")
                repeat = True
                while len(scored)> 2 and repeat:
                    # Step 3: select the best unconfirmed nodes
                    repeat = False
                    new_selected = [node for node, _ in confirmed_scores[:2 - len(selected)]]
                    # Step 4: Examine the past cone of the selected nodes
                    # Check if any node in the past cone has low similarity with its parents
                    nodes_to_remove = self.check_past_cone_similarity(new_selected, dag, logger=logger)
                    # Remove nodes that failed the similarity check (with corrupted behavior)
                    for node_id in nodes_to_remove:
                        if node_id in new_selected:
                            new_selected.remove(node_id)
                            repeat = True
                            # Remove the tuple (node_id, score) from confirmed_scores list
                            confirmed_scores = [(tip, score) for tip, score in confirmed_scores if tip != node_id]
                if logger:
                    logger.info(f"New selected: {new_selected}")
                else:
                    if logger:
                        logger.info(f"New selected: {new_selected}")
                selected += new_selected
        # Step 5: if still 0, pick any other node as final fallback
        if len(selected) < 1:
            backup_nodes = [n for n in dag.graph.nodes if n not in selected]
            random.shuffle(backup_nodes)
            selected += backup_nodes[:1 - len(selected)]
        if logger:
            logger.info(f"Selected tips: {selected}")
        else:
            if logger:
                logger.info(f"Selected tips: {selected}")
        return selected[:2]