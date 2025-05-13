import DAG
import DAGNode
import numpy as np
import TipSelectionAlgorithm
class Peer:
    def __init__(self, dag: DAG, selector: TipSelectionAlgorithm, name="Peer"):
        self.dag = dag
        self.selector = selector
        self.name = name

    def generate_data_matrix(self, size=(3, 3)):
        return np.random.rand(*size)

    def add_node(self):
        data_matrix = self.generate_data_matrix()
        new_node = DAGNode.DAGNode(data_matrix)
        selected_parents = self.selector.select_tips(self.dag, data_matrix)

        # Logging info
        print(f"\n[{self.name}] Adding node {new_node.id[:6]}")
        for pid in selected_parents:
            cw = self.dag.nodes[pid].cw
            status = "(CONFIRMED)" if self.dag.is_confirmed(pid) else ""
            print(f"  ↳ parent {pid[:6]} | CW={cw} {status}")

        self.dag.add_node(new_node, selected_parents)
        return new_node.id
