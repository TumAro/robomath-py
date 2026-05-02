from dataclasses import dataclass
from numpy import asarray

from robomath.groups._base import LieGroup
from robomath._types import Matrix_2x2
from robomath._core.so2 import (
    SO2_test,
    rot_inv
)

@dataclass (frozen=True)
class SO2(LieGroup):
    matrix: Matrix_2x2

    def __post_init__(self) -> None:
        if not SO2_test(self.matrix):
            raise ValueError("Not a SO2 matrix!")
        object.__setattr__(self, 'matrix', asarray(self.matrix))
        
    def __repr__(self) -> str:
        return f"SO2({self.matrix})"
    
    def __matmul__(self, mat2: Matrix_2x2) -> Matrix_2x2:
        return self.matrix @ mat2
    
    def __array__(self, dtype=None):
        if dtype:
            return self.matrix.astype(dtype)
        return self.matrix
    
    def inv(self) -> Matrix_2x2:
        return rot_inv(self.matrix)
    
    def log(self):
        # TODO: no logarithm for SO2
        raise NotImplementedError
    
    def from_tangent(self, omega: Matrix_2x2, theta: float):
        # TODO: no algebra fro SO2
        return NotImplementedError
