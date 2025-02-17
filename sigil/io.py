from .node import Node
import numpy as np


class Input(Node):
    def __init__(self, expected_shape, name=None, dtype=np.float64):
        super().__init__(name)
        self.expected_shape = expected_shape
        self.dtype = dtype

    def set_value(self, value):
        self.value = np.array(value, dtype=self.dtype)
        if len(self.value.shape) == 1:
            self.value = self.value.reshape((1, -1))
        if self.value.shape[1:] != self.expected_shape:
            raise ValueError(f"Expected input values with shape {("batch_size",) + self.expected_shape}")
        
    def compute(self):
        pass

    def backward(self, upstream_gradient=1):
        pass


class DataLoader:
    def __init__(self, x, y, batch_size=1, shuffle_data=True):
        x = np.asarray(x)
        y = np.asarray(y)
        if not (x.shape[0] == y.shape[0]):
            raise ValueError("X and y must have the same number of samples")

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle_data
        self.n_samples = x.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size  # Calculate number of batches correctly
        self.current_batch = 0

    def __iter__(self):
        if self.shuffle:
            self.x, self.y = self._shuffle_data()  # Shuffle if requested
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start = self.current_batch * self.batch_size
            end = min((self.current_batch + 1) * self.batch_size, self.n_samples)  # Handle last batch correctly
            x_batch = self.x[start:end]
            y_batch = self.y[start:end]
            self.current_batch += 1
            return x_batch, y_batch
        else:
            raise StopIteration

    def _shuffle_data(self):  # Helper function to shuffle data
        permutation = np.random.permutation(self.n_samples)
        return self.x[permutation], self.y[permutation]

    def __len__(self):  # Add length method
        return self.n_batches

