from numpy.typing import NDArray
from typing import List

def transpose(R: List[List[float]]):
    '''
    Computes the transpose of a matrix represented as nested Python lists.

    INPUT:
    R : List[List[float]] — matrix as nested lists

    OUTPUT:
    R^T : List[List[float]] — transpose of R
    '''
    return [list(row) for row in zip(*R)]


def _check_square(R: NDArray) -> bool:
    r, c = R.shape
    if r != c:
        raise ValueError("NOT a square Matrix")
    return r==c