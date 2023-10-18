from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import babytorch
from babytorch.autodiff import Context


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

    @classmethod
    def _forward(cls, ctx: Context, *vals):
        return cls.forward(ctx, *vals)

    @classmethod
    def _backward(cls, ctx: Context, d_grad):
        return wrap_tuple(cls.backward(ctx, d_grad))
    
    @classmethod
    def apply(cls, *values):
        # Ensure the other one is a Tensor
        raw_vals = []
        scalars = []
        for v in values:
            if isinstance(v, babytorch.Scalar):
                raw_vals.append(v.data)
                scalars.append(v)
            else:
                raw_vals.append(v)
                scalars.append(babytorch.Scalar(v))

        ctx = Context()
        
        forward_data = cls._forward(ctx, *raw_vals)
        history = babytorch.History(cls, ctx, scalars)
        return babytorch.Scalar(data = forward_data, history = history)


class Add(Function):
    
    def forward(ctx: Context, a, b) -> np.ndarray:
        ctx.save_for_backward(a,b)
        return a + b
    
    def backward(ctx: Context, d_grad):
        a, b = ctx.saved_tensors
        return d_grad, d_grad 


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


class Sigmoid(Function):
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)

    def backward(ctx: Context, d_grad):
        pass


class ReLU(Function):
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)

    def backward(ctx: Context, d_grad):
        pass


class Log(Function):
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)

    def backward(ctx: Context, d_grad):
        pass


class Exp(Function):
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)

    def backward(ctx: Context, d_grad):
        pass


