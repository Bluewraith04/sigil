from sigil.ops.operation import Operation
import numpy as np


class CrossEntropyLoss(Operation):
    def compute(self):
        y_true, y_pred = self.inputs[0].value, self.inputs[1].value
        epsilon = 1e-10  # For numerical stability
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -np.sum(y_true * np.log(y_pred), axis=-1)
        self.value = np.mean(cross_entropy)  # Average over batch

    def backward(self, upstream_gradient=1):
        y_true, y_pred = self.inputs[0].value, self.inputs[1].value
        n = y_true.shape[0]
        gradient = np.mean(-y_true / (y_pred + 1e-10), axis=0)  # Added epsilon for numerical stability
        self.inputs[1].backward(upstream_gradient * gradient) # Average gradient
        return gradient