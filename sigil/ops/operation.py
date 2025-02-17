from sigil.node import Node
from sigil.graph import Graph


class Operation(Node):
    def __init__(self, *inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        for node in self.inputs:
            if not isinstance(node, Node):
                raise TypeError(
                    f"{self.__class__.__name__} operation expected Node like input not {node.__class__.__name}")
            node.outputs.append(self)

        active_graph = Graph.get_default_graph()
        if active_graph:
            for node in self.inputs:
                if node not in active_graph.nodes:
                    active_graph.add_node(node)
            active_graph.add_node(self)

    def compute(self):
        raise NotImplementedError

    def backward(self, upstream_gradient=1):
        raise NotImplementedError
