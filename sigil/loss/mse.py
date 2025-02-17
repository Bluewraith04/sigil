from sigil.ops.operation import Operation
import numpy as np


class MeanSquaredError(Operation):
    def compute(self):
        y_true, y_pred = self.inputs[0].value, self.inputs[1].value
        self.value = np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, upstream_gradient=1):
        y_true, y_pred = self.inputs[0].value, self.inputs[1].value
        n = y_true.shape[-1]
        gradient = np.mean(2 * (y_pred - y_true) / n, axis=0)
        self.inputs[1].backward(upstream_gradient * gradient)
        return gradient
