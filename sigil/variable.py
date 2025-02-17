from .node import Node
from .init import Zeros
import numpy as np


class Variable(Node):
    def __init__(self, name=None, shape=(), initializer=Zeros(), trainable=True, dtype=np.float64):
        super().__init__(name)
        self.shape = shape
        self.initializer = initializer
        self.value = initializer(self.shape)
        self.trainable = trainable
        self.dtype = dtype
        self.grad = np.zeros_like(self.value, dtype=self.dtype)

    def compute(self):
        pass

    def backward(self, upstream_gradient=1):
        self.grad += upstream_gradient

    def zero_grad(self):
        self.gradient = np.zeros_like(self.value, dtype=self.dtype)
