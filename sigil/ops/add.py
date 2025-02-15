import numpy as np

from sigil.ops.operation import Operation


class Add(Operation):
    def __init__(self, *inputs):
        super().__init__(*inputs)

    def compute(self):
        if not all(isinstance(node.value, (int, float, np.ndarray)) for node in self.inputs):
            raise TypeError("Inputs to Add operation must be numeric (int, float, or numpy.ndarray)")

        self.value = sum([node.value for node in self.inputs])

        if isinstance(self.value, np.ndarray) and not all(node.value.shape == self.inputs[0].value.shape for node in self.inputs):
            raise ValueError("Input arrays must have the same shape")
        return self.value

    def backward(self, upstream_gradient=1):
        for node in self.inputs:
            if isinstance(node.value, np.ndarray):
                if isinstance(upstream_gradient, np.ndarray) and node.value.shape != upstream_gradient.shape:
                    raise ValueError("Gradient shape must match node value shape for array inputs")
                node.backward(upstream_gradient * np.ones_like(node.value))
            else:
                node.backward(upstream_gradient)
