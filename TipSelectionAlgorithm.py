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
    def select_tips(self, dag: DAG) -> list:
        dag.update_cumulative_weights()
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
                fallback = unconfirmed[0]  # unica scelta possibile
            return [unconfirmed[0], fallback]
        else:
            # se tutti i tip sono confermati o nessuno è unconfirmed
            if len(all_nodes) >= 2:
                return random.sample(all_nodes, 2)
            else:
                return [all_nodes[0], all_nodes[0]]  # fallback sicuro
            
            
    def select_tips_relaxed(self, dag: DAG) -> list:
        dag.update_cumulative_weights()
        
        # Considera tutti i nodi del DAG non ancora confermati (CW <= 3)
        unconfirmed = [n for n in dag.graph.nodes if not dag.is_confirmed(n)]

        if len(unconfirmed) >= 2:
            return random.sample(unconfirmed, 2)
        elif len(unconfirmed) == 1:
            fallback_candidates = [n for n in dag.graph.nodes if n != unconfirmed[0]]
            if fallback_candidates:
                fallback = random.choice(fallback_candidates)
            else:
                fallback = unconfirmed[0]  # unica scelta possibile
            return [unconfirmed[0], fallback]
        else:
            # tutti i nodi sono confermati → fallback su nodi qualsiasi
            all_nodes = list(dag.graph.nodes)
            if len(all_nodes) >= 2:
                return random.sample(all_nodes, 2)
            else:
                return [all_nodes[0], all_nodes[0]]

        
class MyCustomTipSelector(TipSelectionAlgorithm):
    def select_tips(self, dag: DAG) -> list:
            """TODO"""
            raise NotImplementedError