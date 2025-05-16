import networkx as nx
import uuid

class DAGNode:
    def __init__(self, data_matrix, malicious=False):
        self.id = str(uuid.uuid4())
        self.data = data_matrix
        self.approved_by = set()
        self.cw = 1  # peso di quel nodo
        self.malicious = malicious  # Indica se il nodo è malevolo

    def __repr__(self):
        return f"DAGNode({self.id[:8]}, malicious={self.malicious})"