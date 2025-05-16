import DAG
import DAGNode
import numpy as np
import TipSelectionAlgorithm
class Peer:
    def __init__(self, dag: DAG, selector: TipSelectionAlgorithm, name="Peer"):
        self.dag = dag
        self.selector = selector
        self.name = name


    def add_node(self,new_node):
        selected_parents = self.selector.select_tips(self.dag, new_node.data)

        # Logging info
        print(f"[{self.name}] Adding node {new_node.id[:6]}")
        for pid in selected_parents:
            cw = self.dag.nodes[pid].cw
            status = "(CONFIRMED)" if self.dag.is_confirmed(pid) else ""
            print(f"  ↳ parent {pid[:6]} | CW={cw} {status}")
        print()
        self.dag.add_node(new_node, selected_parents)
        return new_node.id
