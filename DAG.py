import networkx as nx

class DAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.tips = set()
        self.nodes = {}

    def add_node(self, node, parents):
        self.graph.add_node(node.id, data=node.data)
        self.nodes[node.id] = node

        for parent_id in parents:
            self.graph.add_edge(node.id, parent_id)
            self.nodes[parent_id].approved_by.add(node.id)
            self.tips.discard(parent_id)

        self.tips.add(node.id)

    def get_tips(self):
        return list(self.tips)

    def get_node_data(self, node_id):
        node = self.nodes.get(node_id)
        return node.data if node else None

