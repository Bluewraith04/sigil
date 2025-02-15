import numpy as np
from sigil.ops.operation import Operation


class MatMul(Operation):
    def __init__(self, input1, input2):
        super().__init__(input1, input2)

    def compute(self):
        input1 = self.inputs[0].value
        input2 = self.inputs[1].value

        if not isinstance(input1, np.ndarray) or not isinstance(input2, np.ndarray):
            raise TypeError("Ïnputs to MatMul operation must be Numpy arrays.")

        if input1.ndim != 2 or input2.ndim != 2:
            raise ValueError("Inputs to MatMul operation must be 2-dimensional matrices.")

        if input1.shape[1] != input2.shape[0]:
            raise ValueError(f"Cannot perform MatMul operation on arrays with shapes {input1.shape} and {input2.shape}")

        self.value = np.matmul(input1, input2)
        return self.value

    def backward(self, upstream_gradient=1):
        if not isinstance(upstream_gradient, (int, float, np.ndarray)):
            raise TypeError("Upstream gradient must be numeric (int, float, or numpy.ndarray)")

        input1 = self.inputs[0].value
        input2 = self.inputs[1].value

        if isinstance(upstream_gradient, np.ndarray) and self.value is not None \
                and upstream_gradient.shape != self.value.shape:
            raise ValueError("Upstream gradient shape must match operation output shape")

        # Calculate gradients for each input
        grad1 = np.matmul(upstream_gradient, input2.T)
        grad2 = np.matmul(input1.T, upstream_gradient)

        self.inputs[0].backward(grad1)
        self.inputs[1].backward(grad2)
