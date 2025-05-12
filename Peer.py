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
        selected_tips = self.selector.select_tips(self.dag)
        self.dag.add_node(new_node, selected_tips)
        return new_node.id
