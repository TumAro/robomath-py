from dataclasses import dataclass
from numpy import asarray

from robomath._types import Matrix_3x3, Vector3
from robomath._core.so3 import (
    check_skew_symmetry,
    skew_to_vec
)

@dataclass (frozen=True)
class so3:
    matrix: Matrix_3x3

    def __post_init__(self) -> None:
        if not check_skew_symmetry(self.matrix):
            raise ValueError("Not a skew symmetric matrix!")
        object.__setattr__(self, 'matrix', asarray(self.matrix))
        
    def __repr__(self) -> str:
        return f"so3({self.matrix})"
        
    def to_vec(self) -> Vector3:
        return skew_to_vec(self.matrix)
    
    def exp(self, theta: float):
        # TODO: implement after SO3 class exists
        raise NotImplementedError