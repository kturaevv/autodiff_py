from __future__ import annotations 

from typing import Union, Optional, TypeAlias, Sequence, List
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
from numpy import array, float64

from minitorch import functions
from storage import TensorData


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


class Tensor:
    def __init__(self, data: TensorData) -> None:
        self._tensor = data

    def _ensure_tensor(self, b) -> Tensor:
        "Turns a python number into a tensor with the same backend."

        if isinstance(b, Tensor):
            b = Tensor.make([b], (1,), backend=self.backend)
        return b

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        # backend: TensorBackend
    ) -> Tensor:
        "Create a new tensor from data"
        return Tensor(TensorData(storage, shape, strides))

    
    def __add__(self, b) -> Tensor:
        return functions.Add.apply(self, self._ensure_tensor(b))


    def __sub__(self, b) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        "Not used until Module 3"
        return MatMul.apply(self, b)

    def __lt__(self, b) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b) -> Tensor:
        return self + b

    def __rmul__(self, b) -> Tensor:
        return self * b
