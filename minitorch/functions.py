from typing import Tuple, Union

from .tensor import Tensor
from .history import Context


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:

    @classmethod
    def apply(cls,):
        pass


class Neg(Function):
    
    def forward(ctx, a, b):
        pass

    def backward(ctx, b):
        pass

class Inv(Function):
    pass

class Add(Function):
    pass

class Mul(Function):
    pass

class Sigmoid(Function):
    pass

class ReLU(Function):
    pass

class Log(Function):
    pass

class Exp(Function):
    pass

class Sum(Function):
    pass

class MatMul(Function):
    pass
