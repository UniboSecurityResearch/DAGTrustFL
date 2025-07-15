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

class MyCustomTipSelector(TipSelectionAlgorithm):

    def __init__(self, max_depth=4, similarity_threshold=0.4, cw_penalty_factor=0.05):
        self.max_depth = max_depth #chi, depth of the past cone to inspect
        self.similarity_threshold = similarity_threshold
        self.cw_penalty_factor = cw_penalty_factor  # Factor to penalize high CW nodes

    # Cosine similarity between two parameter sets - also considers magnitude!
    # This method is used in Zhu et Al. (2024)
    def cosine_similarity(self, params1, params2):
        return parameter_utils.cosine_similarity_parameters(params1, params2)
    
    def euclidean_distance(self, params1, params2):
        #TODO
        v1 = parameter_utils.flatten_parameters(params1)
        v2 = parameter_utils.flatten_parameters(params2)
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
    def evaluate_similarity(self, node_id, dag: DAG, new_parameters):
        visited = set()
        queue = [(node_id, 0)]
        weighted_similarities = []
        total_weights = 0

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > self.max_depth:
                continue
            visited.add(current)
            current_parameters = dag.nodes[current].data

            # Calculate similarity between the new parameters and the current node's parameters
            sim = self.cosine_similarity(new_parameters, current_parameters)

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

    def select_tips(self, dag: DAG, new_parameters, corrupted=False) -> list:
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
            score = self.evaluate_similarity(tip, dag, new_parameters)
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
                score = self.evaluate_similarity(node, dag, new_parameters)
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

    def __init__(self, max_depth=4, similarity_threshold=0.2, cw_penalty_factor=0.05):
        self.max_depth = max_depth #chi, depth of the past cone to inspect
        self.similarity_threshold = similarity_threshold
        self.cw_penalty_factor = cw_penalty_factor  # Factor to penalize high CW nodes

    # Cosine similarity between two parameter sets - also considers magnitude!
    # This method is used in Zhu et Al. (2024)
    def cosine_similarity(self, params1, params2):
        return parameter_utils.cosine_similarity_parameters(params1, params2)
    
    def euclidean_distance(self, params1, params2):
        #TODO
        v1 = parameter_utils.flatten_parameters(params1)
        v2 = parameter_utils.flatten_parameters(params2)
        return np.linalg.norm(v1 - v2)

    # Evaluate similarity score recursively up to a maximum depth
    # It evaluates also the past cone of the transaction, to check for
    # corrupted behaviors
    def evaluate_similarity(self, node_id, dag: DAG, new_parameters):
        weight=1
        cw_penalty = 1
        # Calculate similarity between the new parameters and the current node's parameters
        current_parameters = dag.nodes[node_id].data
        sim = self.cosine_similarity(new_parameters, current_parameters)
        weighted_similarity = weight * sim * cw_penalty
        return weighted_similarity if not np.isnan(sim) else 0
    
    def check_past_cone_similarity(self, selected_nodes, dag: DAG, debug=0):
        """
        Check if any node in the past cone of selected nodes has low similarity with its parents (corrupted behavior).
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
                current_parameters = dag.nodes[current].data
                
                # Get parent nodes of current node
                parent_nodes = list(dag.graph.successors(current))
                
                # Compare current node's parameters with each parent's parameters
                for parent in parent_nodes:
                    parent_parameters = dag.nodes[parent].data
                    sim = self.cosine_similarity(current_parameters, parent_parameters)
                    
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

    def select_tips(self, dag: DAG, new_parameters, corrupted=False) -> list:
        # Step 1: find all tips (CW < Theta)
        dag.update_cumulative_weights()
        candidate_tips = [
            n for n in dag.graph.nodes
            if not dag.is_confirmed(n)
        ]
        print(f"Candidate tips: {candidate_tips}")
        # Step 2: calculate similarity for each tip
        scored = []
        selected = []
        for tip in candidate_tips:
            score = self.evaluate_similarity(tip, dag, new_parameters)
            #print(f"Tip: {tip}, Score: {score}")
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
            print(f"Scored tips: {scored}")
            repeat = True
            while len(scored)> len(selected) and repeat:
                # Step 3: select the best unconfirmed nodes, with score >= Epsilon
                repeat = False
                selected = [tip for tip, _ in scored if _ >= self.similarity_threshold][:2]
                #Step 4: Examine the past cone of the selected nodes
                # Check if any node in the past cone has low similarity with its parents
                nodes_to_remove = self.check_past_cone_similarity(selected, dag)
                # Remove nodes that failed the similarity check (with corrupted behavior)
                for node_id in nodes_to_remove:
                    if node_id in selected:
                        selected.remove(node_id)
                        repeat = True
                        print(f"Node {node_id} removed from selection due to low similarity in past cone")
                        # Remove the tuple (node_id, score) from scored list
                        scored = [(tip, score) for tip, score in scored if tip != node_id]
                print(f"Selected after past cone check: {selected}")
        # Step 5: if less than 2, add fallback considering also confirmed nodes
        if len(selected) < 2:
            print(f"FALLBACK FROM CONFIRMED NODES {len(selected)}")
            confirmed_nodes = [
                n for n in dag.graph.nodes
                if dag.is_confirmed(n) and n not in selected
            ]
            # Filter confirmed nodes for those that are similar
            confirmed_scores = []
            for node in confirmed_nodes:
                score = self.evaluate_similarity(node, dag, new_parameters)
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
                print(f"Confirmed nodes: {confirmed_scores}")
                repeat = True
                while len(scored)> 2 and repeat:
                    # Step 3: select the best unconfirmed nodes
                    repeat = False
                    new_selected = [node for node, _ in confirmed_scores[:2 - len(selected)]]
                    #Step 4: Examine the past cone of the selected nodes
                    # Check if any node in the past cone has low similarity with its parents
                    nodes_to_remove = self.check_past_cone_similarity(new_selected, dag)
                    # Remove nodes that failed the similarity check (with corrupted behavior)
                    for node_id in nodes_to_remove:
                        if node_id in new_selected:
                            new_selected.remove(node_id)
                            repeat = True
                            # Remove the tuple (node_id, score) from confirmed_scores list
                            confirmed_scores = [(tip, score) for tip, score in confirmed_scores if tip != node_id]
                print(f"New selected: {new_selected}")
                selected += new_selected
        # Step 5: if still 0, pick any other node as final fallback
        if len(selected) < 1:
            backup_nodes = [n for n in dag.graph.nodes if n not in selected]
            random.shuffle(backup_nodes)
            selected += backup_nodes[:1 - len(selected)]
        print(f"Selected tips: {selected}")
        return selected[:2]
    
    