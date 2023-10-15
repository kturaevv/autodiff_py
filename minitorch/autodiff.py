from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

# from minitorch import Tensor

def topological_sort(v):
    # topological order all of the children in the graph
    order = []
    visited = set()
    def build_topological_order(v):
        if v not in visited:
            visited.add(v)
            for child in v.history.inputs:
                build_topological_order(child)
            order.append(v)
    build_topological_order(v)
    return order


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
