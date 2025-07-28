import DAG
import DAGNode
import numpy as np
import TipSelectionAlgorithm
class Peer:
    def __init__(self, dag: DAG, selector: TipSelectionAlgorithm, id, name="Peer", malicious=False, corrupted=False):
        self.dag = dag
        self.selector = selector
        self.id = id
        self.name = name
        self.malicious = malicious
        self.corrupted = corrupted


    def add_node(self,new_node,logger=None):
        print("Searching for the node:", new_node.id[:6])
        selected_parents = self.selector.select_tips(self.dag, new_node.data, corrupted=self.corrupted,logger=logger)

        # Logging info
        print(f"[{self.name}] Adding node {new_node.id[:6]}")
        for pid in selected_parents:
            cw = self.dag.nodes[pid].cw
            status = "(CONFIRMED)" if self.dag.is_confirmed(pid) else ""
            print(f"  ↳ parent {pid[:6]} | CW={cw} {status}")
        print()
        self.dag.add_node(new_node, selected_parents)
        return new_node.id
    
    def manual_add_node(self, new_node, selected_parents, logger=None):
        # Logging info
        if logger:
            logger.info(f"[{self.name}] Adding node {new_node.id[:6]}")
        for pid in selected_parents:
            cw = self.dag.nodes[pid].cw
            status = "(CONFIRMED)" if self.dag.is_confirmed(pid) else ""
            if logger:
                logger.info(f"  ↳ parent {pid[:6]} | CW={cw} {status}")
        if logger:
            logger.info("")
        self.dag.add_node(new_node, selected_parents)
        return new_node.id
