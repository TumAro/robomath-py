from __future__ import annotations
from dataclasses import dataclass
from robomath._types import Matrix_4x4, Vector6
from numpy import asarray

from robomath._core.se3 import (
    se3_test,
    se3_to_vec,
    mat_exp6,
    vec_to_se3
)

@dataclass (frozen=True)
class se3:
    matrix: Matrix_4x4

    def __post_init__(self) -> None:
        if not se3_test(self.matrix):
            raise ValueError("Not a se3 matrix!")
        object.__setattr__(self, 'matrix', asarray(self.matrix))
        
    @classmethod
    def from_vec(cls, vec: Vector6) -> se3:
        return cls(vec_to_se3(vec))
        
    def __repr__(self) -> str:
        return f"se3({self.matrix})"
    
    def to_vec(self) -> Vector6:
        return se3_to_vec(self.matrix)
    
    def exp(self, theta: float) -> Matrix_4x4:
        return mat_exp6(self.matrix, theta)