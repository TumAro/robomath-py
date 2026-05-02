from __future__ import annotations
from dataclasses import dataclass
from numpy import asarray

from robomath._types import Vector6, Vector3

@dataclass (frozen=True)
class Wrench:
    _vec: Vector6

    def __post_init__(self):
        if self._vec.shape != (6,):
            raise ValueError(f"Wrench requires R^6, got shape: {self._vec.shape}")
        object.__setattr__(self, '_vec', asarray(self._vec))

    def __repr__(self) -> str:
        return f"Wrench(moment: {self.moment}, force: {self.force})"
    
    def __array__(self, dtype=None):
        if dtype:
            return self._vec.astype(dtype)
        return self._vec
    
    @property
    def moment(self) -> Vector3:
        return self._vec[:3]
    
    @property
    def force(self) -> Vector3:
        return self._vec[3:]