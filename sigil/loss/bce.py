from sigil.ops.operation import Operation
import numpy as np


class BinaryCrossEntropyLoss(Operation):
    def compute(self):
        y_true, y_pred = self.inputs[0].value, self.inputs[1].value  # y_true: true labels (0 or 1), y_pred: predicted probabilities
        epsilon = 1e-10  # For numerical stability
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon) # Clip y_pred for numerical stability
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # Averaged loss
        self.value = loss
        return self.value

    def backward(self, upstream_gradient=1):
        y_true, y_pred = self.inputs[0].value, self.inputs[1].value
        n = y_true.shape[0] # batch size
        gradient = np.mean(-(y_true / (y_pred + 1e-10) - (1 - y_true) / (1 - y_pred + 1e-10)), axis=0) # Averaged gradient
        self.inputs[1].backward(upstream_gradient * gradient)
        return gradient
