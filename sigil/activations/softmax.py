from sigil.ops.operation import Operation
import numpy as np


class Softmax(Operation):
    def compute(self):
        exp_x = np.exp(self.inputs[0].value - np.max(self.inputs[0].value, axis=1, keepdims=True))
        self.value = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.value

    def backward(self, upstream_gradient=1):
        softmax = self.value
        gradient = np.mean(upstream_gradient * softmax * (1 - softmax), axis=0)
        self.inputs[0].backward(gradient)
