from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    @abstractmethod
    def __call__(self, shape):
        pass


class Zeros(Initializer):
    def __call__(self, shape):
        return np.zeros(shape)


class Ones(Initializer):
    def __call__(self, shape):
        return np.zeros(shape)


class RandomUniform(Initializer):
    def __call__(self, shape):
        return np.random.uniform(low=0.0, high=1.0, size=shape)


class RandomNormal(Initializer):
    def __call__(self, shape):
        return np.random.normal(loc=0.0, scale=1.0, size=shape)


class GlorotUniform(Initializer):
    def __call__(self, shape):
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(low=-limit, high=limit, size=shape)


class GlorotNormal(Initializer):
    def __call__(self, shape):
        limit = np.sqrt(6 / sum(shape))
        return np.random.normal(loc=0, scale=limit, size=shape)


class HeUniform(Initializer):
    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0]))
        return np.random.uniform(low=-limit, high=limit, size=shape)


class HeNormal(Initializer):
    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0]))
        return np.random.normal(loc=0, scale=limit, size=shape)
