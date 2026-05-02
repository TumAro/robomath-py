from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple

from robomath.groups._base import LieGroup
from robomath._types import Matrix_6x6, Matrix_4x4, Vector3
from robomath.groups.SO3 import SO3
from robomath.tangent import so3, se3
from robomath._core.se3 import (
    SE3_test,
    get_rotation,
    get_translation,
    trans_inverse,
    logarithm6,
    adjoint,
    compose
)

@dataclass (frozen=True)
class SE3(LieGroup):
    matrix: Matrix_4x4
    rotation: SO3 = field(repr=False, init=False)
    translation: Vector3 = field(repr=False, init=False)

    def __post_init__(self) -> None:
        if not SE3_test(self.matrix):
            raise ValueError("Not a SE3 matrix!")
        
        object.__setattr__(self, 'rotation', SO3(get_rotation(self.matrix)))
        object.__setattr__(self, 'translation', get_translation(self.matrix))
        
    def __repr__(self) -> str:
        return f"SE3(\n{self.matrix}\n)"
    
    def __matmul__(self, other: SE3):
        return SE3(compose(self.matrix, other.matrix))
    
    def inv(self) -> SE3:
        return SE3(trans_inverse(self.matrix))
    
    def log(self) -> Tuple[so3.so3, Vector3, float]:
        mat, vec, theta = logarithm6(self.matrix)
        return (so3.so3(mat), vec, theta)
    
    def adjoint(self) -> Matrix_6x6:
        return adjoint(self.matrix)
    
    @staticmethod
    def from_tangent(twist: se3.se3, theta: float) -> SE3:
        return SE3(twist.exp(theta))