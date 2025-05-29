import DAG
import numpy as np
import random

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
        self.max_depth = max_depth
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

    # Evaluate similarity score recursively up to a maximum depth
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

    def select_tips(self, dag: DAG, new_matrix) -> list:
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