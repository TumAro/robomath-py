from __future__ import annotations
from dataclasses import dataclass

from robomath.groups._base import LieGroup
from robomath._types import Matrix_4x4
from robomath._core.se3 import (
    SE3_test
)

@dataclass (frozen=True)
class SE3(LieGroup):
    matrix: Matrix_4x4

    def __post_init__(self) -> None:
        if not SE3_test(self.matrix):
            raise ValueError("Not a SE3 matrix!")
        
    def __repr__(self) -> str:
        return f"SE3(\n{self.matrix}\n)"