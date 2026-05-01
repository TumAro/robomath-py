from dataclasses import dataclass

from robomath.groups._base import LieGroup
from robomath._types import Matrix_3x3


@dataclass (frozen=True)
class SO2(LieGroup):
    matrix: Matrix_3x3
    # TODO: se2 tangent space is not built yet