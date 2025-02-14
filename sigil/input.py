from .node import Node
import numpy as np


class Input(Node):
    def __init__(self, expected_shape, name=None):
        super().__init__(name)
        self.expected_shape = expected_shape

    def set_value(self, value):
        self.value = np.array(value)
        if self.value.shape[1:] != self.expected_shape:
            raise ValueError(f"Expected input values with shape {("batch_size",) + self.expected_shape}")
        if len(self.value.shape) == 1:
            self.value = self.value.reshape((1, -1))
        
    def compute(self):
        pass

    def backward(self, upstream_gradient=1):
        pass
