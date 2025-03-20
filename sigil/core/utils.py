from base import TensorLike, OperationRegistry, IOperation
from functools import wraps
from tensor import Tensor
from typing import Callable, TypeVar, Any, Type, List, Optional

# Generic type for the base operation function
F = TypeVar("F", bound=Callable[..., Any])


def register(name: str):
    def wrapper(op: Type[IOperation]) -> Type[IOperation]:
        try:
            OperationRegistry.register_op(name=name, op=op())
        except ValueError as e:
            raise RuntimeError(f"Failed to register operation '{name}': {e}")
        return op
    return wrapper


def forward_op(base_op: F) -> F:
    @wraps(base_op)
    def op(self, *inputs: Any) -> TensorLike:
        # Ensure all inputs are TensorLike
        processed_inputs = [
            inp if isinstance(inp, TensorLike) else Tensor(inp, requires_grad=False)
            for inp in inputs
        ]

        # Extract data from TensorLike inputs
        input_data = [inp.data for inp in inputs]

        # Apply the base operation
        result_data = base_op(*input_data)

        # Wrap the result in a Tensor
        result = Tensor(result_data)

        # Track the operation and inputs for autograd
        if any(inp.requires_grad for inp in processed_inputs):
            result.requires_grad = True
            result.op = self
            result.inputs = list(inputs)

        return result

    return op


def backward_op(base_op: F) -> F:
    @wraps(base_op)
    def op(self, grad: TensorLike, node: TensorLike) -> List[Optional[TensorLike]]:
        # Ensure gradient is TensorLike
        if not isinstance(grad, TensorLike):
            raise TypeError("Gradient must be a TensorLike object.")

        # Compute backward gradients
        gradients = base_op(self, grad, node)

        # Ensure output is a list of optional Tensors
        if not isinstance(gradients, list):
            raise TypeError("Backward operation must return a list of TensorLike or None.")

        return gradients

    return op
