from __future__ import annotations 

from typing import Union, Optional, TypeAlias, Sequence, List, Type
from typing_extensions import TypeAlias
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import array, float64

from minitorch import functions
from minitorch import Function, Context


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


@dataclass
class History:

    fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Optional[Sequence[Tensor]] = () 


class Tensor:

    def __init__(self, data, history: History = History()) -> None:
        self.data = np.array(data)
        self.history = history
        self.grad = 0

    def __add__(self, other) -> Tensor:
        return functions.Add.apply(self, other)

    def __matmul__(self, other):
        return functions.MatMul.apply(self, other)

    def __mul__(self, other):
        return functions.Mul.apply(self, other)

    def __pow__(self, power):
        return functions.Pow.apply(self, power)

    """ Handle other use cases using base operations. """
   
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}))"

    def backward(self):
        from minitorch.autodiff import topological_sort
        order = topological_sort(self)

        def backpropagate(variable: Tensor, deriv) -> None:
            """
            Runs backpropagation on the computation graph in order to
            compute derivatives for the leave nodes.

            Args:
                variable: The right-most variable
                deriv  : Its derivative that we want to propagate backward to the leaves.

            No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
            """
            fn = variable.history.fn
            ctx = variable.history.ctx
            deriv = fn._backward(ctx, deriv)

        for v in order.pop(-1).history.inputs:
            self.grad = np.ones(v.data.shape)
            backpropagate(v, self.grad)
        
        for v in reversed(order):
            backpropagate(v, self.grad)

