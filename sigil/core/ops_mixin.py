from base import OperationRegistry


class TensorOpsMixin:
    """Mixin to provide arithmetic operations via the OperationRegistry."""

    def __add__(self, other):
        return OperationRegistry.get_op("add").forward(self, other)

    def __sub__(self, other):
        return OperationRegistry.get_op("sub").forward(self, other)

    def __mul__(self, other):
        return OperationRegistry.get_op("mul").forward(self, other)

    def __truediv__(self, other):
        return OperationRegistry.get_op("div").forward(self, other)

    def __matmul__(self, other):
        return OperationRegistry.get_op("matmul").forward(self, other)
