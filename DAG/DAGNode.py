import networkx as nx
import uuid

class DAGNode:
    def __init__(self, data_matrix, peer, epoch, malicious=False, corrupted=False):
        self.id = str(uuid.uuid4())
        self.peer = peer
        self.epoch = epoch
        self.data = data_matrix
        self.approved_by = set()
        self.cw = 1  # weight of this node
        self.malicious = malicious  # Indicates if the node is malicious
        self.corrupted = corrupted  # Indicates if the node is corrupted

    def __repr__(self):
        return f"DAGNode({self.id[:8]}, malicious={self.malicious})"