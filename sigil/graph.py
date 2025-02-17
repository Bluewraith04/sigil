from .node import Node
from .variable import Variable
from collections import deque


class Graph:
    _default_graph = None

    def __init__(self):
        self.nodes: list[Node] = []  # All nodes in the graph
        self.trainable_vars: list[Node] = []  # Track variables for optimization
        self.topological_order: list[Node] = []  # Nodes sorted in topological order

    @classmethod
    def get_default_graph(cls):
        return cls._default_graph

    def __enter__(self):
        # Set this graph as the default context
        self._prev_graph = Graph._default_graph
        Graph._default_graph = self
        return self

    def __exit__(self, *args):
        # Restore the previous default graph
        Graph._default_graph = self._default_graph
        self.topological_order = None

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
            if isinstance(node, Variable) and node.trainable:
                self.trainable_vars.append(node)

    def build_topological_order(self):
        # Use kahn's algorithm to sort nodes
        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for output in node.outputs:
                in_degree[output] += 1

        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        self.topological_order = []

        while queue:
            node = queue.popleft()
            self.topological_order.append(node)
            for output in node.outputs:
                in_degree[output] -= 1
                if in_degree[output] == 0:
                    queue.append(output)

        if len(self.topological_order) != len(self.nodes):
            raise RuntimeError("Graph contains cycles!")

    def __repr__(self):  # Added representation
        return f"<Graph with {len(self.nodes)} nodes>"
