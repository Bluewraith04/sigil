import numpy as np
from abc import ABC, abstractmethod


class Node(ABC):
    def __init__(self, name=None):
        self.name = name or id(self)
        self.value = None
        self.outputs = []

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def backward(self, upstream_gradient=1):
        pass
    