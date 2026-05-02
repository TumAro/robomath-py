from dataclasses import dataclass

from robomath.groups._base import LieGroup
from robomath._types import Matrix_3x3


@dataclass (frozen=True)
class SE2(LieGroup):
    matrix: Matrix_3x3
    # TODO: se2 tangent space is not built yet

    def __repr__(self) -> str:
        return f"SO2({self.matrix})"
    
    def __array__(self, dtype = None):
        if dtype:
            return self.matrix.astype(dtype)
        return self.matrix