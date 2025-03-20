from base import TensorLike, IOperation
from ops_mixin import TensorOpsMixin
from typing import Any, List, Optional
import numpy as np


class Tensor(TensorOpsMixin, TensorLike):
    def __init__(self,
                 data: Any,
                 requires_grad: bool = False,
                 op: Optional[IOperation] = None,
                 inputs: Optional[List] = None
                 ) -> None:
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.op = op
        self.inputs = inputs if inputs else []
        self.grad = None if not requires_grad else Tensor(0)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def backward(self, grad: Optional["TensorLike"] = None) -> None:
        if grad is None:
            grad = Tensor(1)
        self.grad = grad
        if self.op:
            grads = self.op.backward(grad, self)
            for inp, g in zip(self.inputs, grads):
                if inp.requires_grad:
                    inp.backwards(g)
