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
            return [tips[0]]
        else:
            return []
        
class IOTALikeTipSelector(TipSelectionAlgorithm):
    def select_tips(self, dag: DAG) -> list:
        tips = dag.get_tips()
        all_nodes = list(dag.graph.nodes())

        if len(tips) >= 2:
            return random.sample(tips, 2)
        elif len(tips) == 1:
            non_tip_nodes = [n for n in all_nodes if n != tips[0]]
            fallback = random.choice(non_tip_nodes) if non_tip_nodes else tips[0]
            return [tips[0], fallback]
        else:
            # Nessun tip, scegli due nodi qualsiasi
            return random.sample(all_nodes, 2) if len(all_nodes) >= 2 else [all_nodes[0], all_nodes[0]]