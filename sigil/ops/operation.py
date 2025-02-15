from sigil.node import Node


class Operation(Node):
    def __init__(self, *inputs, name=None):
        super().__init__(name)
        self.inputs = inputs
        for node in self.inputs:
            if not isinstance(node, Node):
                raise TypeError("Add operation expected inputs to be a subclass of Node")
            node.outputs.append(self)

    def compute(self):
        raise NotImplementedError

    def backward(self, upstream_gradient=1):
        raise NotImplementedError
