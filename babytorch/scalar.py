from __future__ import annotations 

from typing import Optional, Sequence, Type, Iterable
from dataclasses import dataclass

from minitorch import functions
from minitorch.functions import Function, Context


@dataclass
class History:
    fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Optional[Sequence[Scalar]] = () 


class Scalar:
   
    def __init__(self, data, history: History = History()) -> None:
        self.data = data
        self.history = history
        self.grad = 0

    def __add__(self, other) -> Scalar:
        return functions.Add.apply(self, other)
    
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

    def accumulate_grad(self, d_x) -> None:
        """ Accrue value to the the gradient. """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = 0.0
        self.grad += d_x

    def is_leaf(self) -> bool:
        """ Ensure that the Scalar was not created by the user, i.e., is not a starting Scalar."""
        return self.history is not None and self.history.fn is None

    def chain_rule(self, deriv):
        """ A proxy to compute and return derivatives and its corresponding inputs."""
        h = self.history
        assert h.ctx
        assert h.fn
        assert h.inputs
        # grads is the tuple of derivatives
        # h_inputs = (a, b)
        # grads    = (d_b, d_a)
        grads = h.fn._backward(h.ctx, deriv)
        return zip(h.inputs, grads)

    def backward(self):
        from minitorch.autodiff import topological_sort, backpropagate
        self.grad = 1 
        backpropagate(self, self.grad)

    @property
    def parents(self) -> Iterable[Scalar]:
        assert self.history.inputs is not None
        return self.history.inputs
 

    @property
    def id(self) -> int:
        return id(self)