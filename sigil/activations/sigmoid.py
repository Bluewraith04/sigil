from sigil.ops.operation import Operation
import numpy as np


class Sigmoid(Operation):
    def compute(self):
        self.value = 1 / (1 + np.exp(-self.inputs[0].value))
        return self.value

    def backward(self, upstream_gradient=1):
        sigmoid = self.value
        gradient = np.mean(upstream_gradient * sigmoid * (1 - sigmoid), axis=0)
        self.inputs[0].backward(gradient)