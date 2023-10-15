from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import minitorch
from minitorch.autodiff import Context


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class FunctionInterface(ABC):

    @abstractmethod
    def forward(*args): ...

    @abstractmethod
    def backward(*args): ...


# Constructors
class Function(FunctionInterface):

    @abstractmethod
    def forward(cls, ctx: Context, d_grad): ...

    @classmethod
    def _backward(cls, ctx: Context, d_grad):
        return wrap_tuple(cls.backward(ctx, d_grad))
    
    @classmethod
    def apply(cls, *values):
        # Ensure the other one is a Tensor
        for i in range(len(values)):
            if not isinstance(values[i], minitorch.Tensor):
                values[i] = minitorch.Tensor(values[i])      

        ctx = Context()
        forward_data = cls.forward(ctx, *[v.data for v in values])
        history = minitorch.History(cls, ctx, values)
        return minitorch.Tensor(data = forward_data, history = history)


class Add(Function):
    
    def forward(ctx: Context, a, b) -> np.ndarray:
        ctx.save_for_backward(a,b)
        return a + b
    
    def backward(ctx: Context, d_grad):
        a, b = ctx.saved_tensors
        return d_grad,d_grad 


class Mul(Function):

    def forward(ctx: Context, a, b) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a * b

    def backward(ctx: Context, d_grad):
        a, b = ctx.saved_tensors
        return b * d_grad, a * d_grad 


class Pow(Function):

    def forward(ctx: Context, a, b) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a ** b

    def backward(ctx: Context, d_grad):
        data, power = ctx.saved_tensors 
        return ( power * data ** (power - 1) ) * d_grad 


class MatMul(Function):

    def forward(ctx: Context, a, b) -> np.ndarray:
        ctx.save_for_backward(a,b)
        return a @ b

    def backward(ctx: Context, d_grad):
        a, b = ctx.saved_tensors
        a: np.ndarray
        b: np.ndarray
        d_a = d_grad @ b.T
        d_b = a.T, d_grad
        return d_a, d_b


class Sigmoid(Function):
    def forward(ctx, a, b):
        pass

    def backward(ctx, b):
        pass


class ReLU(Function):
    def forward(ctx, a, b):
        pass

    def backward(ctx, b):
        pass


class Log(Function):
    def forward(ctx, a, b):
        pass

    def backward(ctx, b):
        pass


class Exp(Function):
    def forward(ctx, a, b):
        pass

    def backward(ctx, b):
        pass


class Sum(Function):
    def forward(ctx, a, b):
        pass

    def backward(ctx, b):
        pass
