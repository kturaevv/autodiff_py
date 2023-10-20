from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol


class Scalar(Protocol):
    @property
    def id(self) -> int:
        ...

    @property
    def parents(self) -> Iterable[Scalar]:
        ...

    def accumulate_grad(self, d_x) -> None:
        ...

    def is_leaf(self) -> bool:
        ...

    def chain_rule(self, deriv) -> Iterable[Tuple[Scalar, Any]]:
        ...

    def backward(self):
        ...


def topological_sort(v: Scalar) -> Iterable[Scalar]:
    order = []
    visited = set()

    def build_topological_order(v: Scalar):
        if v in visited or v.is_leaf():
            return

        visited.add(v)
        for p in v.parents:
            build_topological_order(p)
        order.append(v)

    build_topological_order(v)
    return reversed(order)


def backpropagate(variable: Scalar, deriv) -> None:
    grad_dict = {}
    grad_dict[variable.id] = deriv
    order = topological_sort(variable)

    for node in order:
        d_out = grad_dict[node.id]
        chain = node.chain_rule(d_out)
        for var, grad in chain:
            if var.is_leaf():
                var.accumulate_grad(grad)
            elif var.id in grad_dict:
                grad_dict[var.id] += grad
            else:
                grad_dict[var.id] = grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
