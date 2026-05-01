from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal

from robomath.groups._base import LieGroup
from robomath._types import Matrix_3x3, Vector3
from robomath.tangent.so3 import so3
from robomath._core.so3 import (
    _identity,
    SO3_test,
    rot_inv,
    logarithm,
    rodrigues,
    rot_x, rot_y, rot_z
)

@dataclass (frozen=True)
class SO3(LieGroup):
    matrix: Matrix_3x3

    def __post_init__(self) -> None:
        if not SO3_test(self.matrix):
            raise ValueError("Not a  matrix!")
        
    def __repr__(self) -> str:
        return f"SO3(\n{self.matrix}\n)"
    
    def __matmul__(self, other: SO3) -> SO3:
        return SO3(self.matrix @ other.matrix)
    
    def __array__(self, dtype = None):
        if dtype:
            return self.matrix.astype(dtype)
        else:
            return self.matrix
        
    def inv(self) -> SO3:
        return SO3(rot_inv(self.matrix))
    
    def log(self) -> Tuple[so3, float]:
        matrix, theta = logarithm(self.matrix)
        return so3(matrix), theta
    
    @staticmethod
    def rodrigues(axis_vec: Vector3, theta: float) -> SO3:
        return SO3(rodrigues(axis_vec, theta))
    
    @staticmethod
    def from_tangent(skew_mat: so3, theta: float) -> SO3:
        from numpy.linalg import norm
        vec = skew_mat.to_vec()
        vec = vec / norm(vec)
        return SO3(rodrigues(vec, theta))
    
    @staticmethod
    def principal_rot(axis: Literal['x', 'y', 'z'], theta: float) -> SO3:
        if axis == 'x':
            return SO3(rot_x(theta))
        elif axis == 'y':
            return SO3(rot_y(theta))
        elif axis == 'z':
            return SO3(rot_z(theta))
        
        return _identity
