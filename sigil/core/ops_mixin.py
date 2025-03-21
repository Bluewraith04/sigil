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

    def __pow__(self, power, modulo=None):
        return OperationRegistry.get_op("pow").forward(self, power)

    def __radd__(self, other):
        return OperationRegistry.get_op("add").forward(other, self)

    def __rsub__(self, other):
        return OperationRegistry.get_op("sub").forward(other, self)

    def __rmul__(self, other):
        return OperationRegistry.get_op("mul").forward(other, self)

    def __rtruediv__(self, other):
        return OperationRegistry.get_op("div").forward(other, self)

    def __rpow__(self, other):
        return OperationRegistry.get_op("pow").forward(other, self)

    def __pos__(self):
        return OperationRegistry.get_op("add").forward(0, self)

    def __neg__(self):
        return OperationRegistry.get_op("sub").forward(0, self)

