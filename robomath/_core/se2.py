import numpy as np
from numpy.typing import NDArray
from robomath._core.so2 import SO2_test, rot_matrix

# * IMPLEMENTATION OF ==> SE GROUP

# * IMPLEMENTATION OF ==> se algebra

def SE2_test(T: NDArray) -> bool:
    '''
    Tests whether a matrix belongs to SE(2).

    INPUT:
    T : (3x3 NDArray) — candidate transformation matrix

    OUTPUT:
    bool — True if T ∈ SE(2), i.e. top-left 2x2 ∈ SO(2) and bottom row = [0, 0, 1]
    '''
    if T.shape != (3, 3):
        return False
    if not SO2_test(T[:2, :2]):
        return False
    if not np.allclose(T[2], [0, 0, 1]):
        return False
    return True

def transform(theta: float, p: NDArray) -> NDArray:
    '''
    Constructs a 2D homogeneous transformation matrix.

    INPUT:
    theta : float        — angle of rotation in radians, θ ∈ R
    p     : (2,) NDArray — translation vector, p ∈ R^2

    OUTPUT:
    T : (3x3 NDArray) — homogeneous transformation matrix T ∈ SE(2)
    '''
    T = np.eye(3)
    T[:2, :2] = rot_matrix(theta)
    T[:2, 2] = p
    return T

def trans_inverse(T: NDArray) -> NDArray:
        '''
        Computes the inverse of a transformation matrix in SE(2).

        INPUT:
        T : (3x3 NDArray) — transformation matrix T ∈ SE(2)

        OUTPUT:
        T^-1 : (3x3 NDArray) — inverse transformation T^-1 ∈ SE(2)
               computed as [R^T, -R^T p; 0, 1]
        '''
        if not SE2_test(T):
            raise ValueError("Not a valid transformation matrix in SE2")
        R = T[:2, :2]
        p = T[:2, 2]
        invT = np.eye(3)
        invT[:2, :2] = R.T
        invT[:2, 2] = -(R.T) @ p
        return invT
