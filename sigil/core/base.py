from abc import abstractmethod
from functools import wraps
from typing import Protocol, Any, List, Optional, Type, runtime_checkable


# -------------------
# Protocols (Interfaces)
# -------------------
@runtime_checkable
class TensorLike(Protocol):
    data: Any
    grad: Optional["TensorLike"]
    requires_grad: bool
    op: Optional["IOperation"]
    inputs: List["TensorLike"]

    @abstractmethod
    def backward(self, grad: Optional["TensorLike"] = None) -> None:
        pass


class IOperation(Protocol):
    @abstractmethod
    def forward(self, *inputs: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def backward(self, grad: TensorLike, node: TensorLike) -> List[Optional["TensorLike"]]:
        pass


# -------------------
# Operation Registry
# -------------------

class OperationRegistry:
    _operations: dict[str, IOperation] = {}

    @classmethod
    def register_op(cls, name: str, op: IOperation):
        if name in cls._operations:
            raise ValueError(f"Operation {name} already registered")
        cls._operations[name] = op

    @classmethod
    def get_op(cls, name: str):
        if name not in cls._operations:
            raise KeyError(f"Operation {name} not found. ")
        return cls._operations[name]

