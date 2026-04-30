import numpy as np
from math import sin, cos
from numpy.typing import NDArray

_identity = np.identity(2, dtype=np.float32)

# * IMPLEMENTATION OF ==> SO GROUP

def SO2_test(R: NDArray) -> bool:
    '''
    Tests whether a matrix belongs to SO(2).

    INPUT:
    R : (2x2 NDArray) — candidate rotation matrix

    OUTPUT:
    bool — True if R ∈ SO(2), i.e. R^T R = I and det(R) = 1
    '''
    if not np.allclose(R.T @ R, _identity):
        return False
    if abs(np.linalg.det(R) - 1) > 1e-6:
        return False
    
    return True

def rot_inv(R: NDArray) -> NDArray[np.float32]:
    '''
    Computes the inverse of a rotation matrix in SO(2).

    INPUT:
    R : (2x2 NDArray) — rotation matrix, R ∈ SO(2)

    OUTPUT:
    R^-1 : (2x2 NDArray) — inverse rotation, R^-1 = R^T ∈ SO(2)
            returns identity if R ∉ SO(2)
    '''
    if SO2_test(R):
        return R.T

    return _identity

def rot_matrix(theta: float) -> NDArray[np.float32]:
    '''
    Constructs a 2D rotation matrix from an angle.

    INPUT:
    theta : float — angle of rotation in radians, θ ∈ R

    OUTPUT:
    R : (2x2 NDArray) — rotation matrix R ∈ SO(2)
    '''
    return np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])


# * IMPLEMENTATION OF ==> so ALGEBRA

