from sigil.ops.operation import Operation
import numpy as np


class ReLU(Operation):
    def compute(self):
        self.value = np.maximum(self.inputs[0].value, 0)
        return self.value

    def backward(self, upstream_gradient=1):
        gradient = np.mean(upstream_gradient * (self.inputs[0].value > 0), axis=0)
        self.inputs[0].backward(gradient)
