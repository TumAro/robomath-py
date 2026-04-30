import numpy as np
from math import sin, cos
from numpy.typing import NDArray
from typing import List, Tuple

_identity = np.identity(3, dtype=np.float32)

# * IMPLEMENTATION OF ==> SO GROUP
def _check_square(R: NDArray) -> bool:
    r, c = R.shape
    if r != c:
        raise ValueError("NOT a square Matrix")
    return r==c

def SO3_test(R: NDArray) -> bool:
    '''
    Tests whether a matrix belongs to SO(3).

    INPUT:
    R : (3x3 NDArray) — candidate rotation matrix

    OUTPUT:
    bool — True if R ∈ SO(3), i.e. R^T R = I and det(R) = 1
    '''
    if not np.allclose(R.T @ R, _identity):
        return False
    if abs(np.linalg.det(R) - 1) > 1e-6:
        return False
    
    return True

def transpose(R: List[List[float]]):
    '''
    Computes the transpose of a matrix represented as nested Python lists.

    INPUT:
    R : List[List[float]] — matrix as nested lists

    OUTPUT:
    R^T : List[List[float]] — transpose of R
    '''
    return [list(row) for row in zip(*R)]

def rot_inv(R: NDArray) -> NDArray[np.float32]:
    '''
    Computes the inverse of a rotation matrix in SO(3).

    INPUT:
    R : (3x3 NDArray) — rotation matrix, R ∈ SO(3)

    OUTPUT:
    R^-1 : (3x3 NDArray) — inverse rotation, R^-1 = R^T ∈ SO(3)
            returns identity if R ∉ SO(3)
    '''
    if SO3_test(R):
        return R.T

    return _identity

def rot_x(theta: float) -> NDArray[np.float32]:
    '''
    Constructs a rotation matrix about the X-axis.

    INPUT:
    theta : float — angle of rotation in radians, θ ∈ R

    OUTPUT:
    R : (3x3 NDArray) — rotation matrix R ∈ SO(3)
    '''
    return np.array([
        [1, 0, 0],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta), cos(theta)]
    ])

def rot_y(theta: float) -> NDArray[np.float32]:
    '''
    Constructs a rotation matrix about the Y-axis.

    INPUT:
    theta : float — angle of rotation in radians, θ ∈ R

    OUTPUT:
    R : (3x3 NDArray) — rotation matrix R ∈ SO(3)
    '''
    return np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])

def rot_z(theta: float) -> NDArray[np.float32]:
    '''
    Constructs a rotation matrix about the Z-axis.

    INPUT:
    theta : float — angle of rotation in radians, θ ∈ R

    OUTPUT:
    R : (3x3 NDArray) — rotation matrix R ∈ SO(3)
    '''
    return np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0,0,1]
    ])

def rodrigues(w: List, theta: float) -> NDArray:
    '''
    Computes the rotation matrix from an axis-angle representation (Rodrigues' formula).

    INPUT:
    w     : List[float] — unit rotation axis, ω ∈ R^3, ‖ω‖ = 1
    theta : float       — angle of rotation in radians, θ ∈ R

    OUTPUT:
    R : (3x3 NDArray) — rotation matrix R ∈ SO(3)
        R = I + sin(θ)[ω] + (1 - cos(θ))[ω]²
    '''

    if len(w) != 3:
        raise ValueError("Rotation axis is not in 3D")
    
    if abs(sum(i*i for i in w) - 1.0) > 1e-9:
        raise ValueError("Rotation axis is not unit vector.")
    
    skew_omega = skew_symmetric(*w)

    rot = np.identity(3) + sin(theta)*skew_omega + (1-cos(theta))*(skew_omega @ skew_omega)
    return rot

def skew_mat_to_SO3(w_theta: NDArray) -> NDArray:
        '''
        Matrix exponential: maps a scaled skew-symmetric matrix to SO(3).

        INPUT:
        w_theta : (3x3 NDArray) — scaled skew-symmetric matrix [ω]θ ∈ so(3)

        OUTPUT:
        R : (3x3 NDArray) — rotation matrix R = e^([ω]θ) ∈ SO(3)
        '''
        w_vec = skew_to_vec(w_theta)
        theta = float(np.linalg.norm(w_vec))
        w_hat = [float(x / theta) for x in w_vec]

        return rodrigues(w_hat, theta)

# * IMPLEMENTATION OF ==> so ALGEBRA

def skew_symmetric(x1, x2, x3) -> NDArray:
    '''
    Constructs the skew-symmetric matrix of a 3D vector.

    INPUT:
    x1, x2, x3 : float — components of vector x ∈ R^3

    OUTPUT:
    [x] : (3x3 NDArray) — skew-symmetric matrix [x] ∈ so(3)
    '''
    return np.array([
        [0, -x3, x2],
        [x3 , 0, -x1],
        [-x2, x1, 0]
    ])

def check_skew_symmetry(matrix: NDArray) -> bool:
    '''
    Tests whether a matrix belongs to so(3).

    INPUT:
    matrix : (3x3 NDArray) — candidate skew-symmetric matrix

    OUTPUT:
    bool — True if matrix ∈ so(3), i.e. matrix = -matrix^T
    '''
    return np.allclose(matrix.T, -matrix)

def skew_to_vec(matrix: NDArray) -> List[float]:
    '''
    Extracts the 3D vector from a skew-symmetric matrix.

    INPUT:
    matrix : (3x3 NDArray) — skew-symmetric matrix [x] ∈ so(3)

    OUTPUT:
    x : List[float] — vector x ∈ R^3 such that [x] = matrix
    '''
    if not check_skew_symmetry(matrix):
        raise ValueError("Not a skew symmetrix matrix!")
    
    x1 = matrix[2][1]
    x2 = matrix[0][2]
    x3 = matrix[1][0]

    return [x1, x2, x3]

def logarithm(R: NDArray) -> Tuple[NDArray, float]:
        '''
        Matrix logarithm: maps a rotation matrix to so(3).

        INPUT:
        R : (3x3 NDArray) — rotation matrix R ∈ SO(3)

        OUTPUT:
        [ω] : (3x3 NDArray) — skew-symmetric matrix [ω] ∈ so(3), rotation axis
        θ   : float         — angle of rotation in radians, θ ∈ [0, π]
        '''

        if R.shape != (3,3):
            raise ValueError("Not a rotation Matrix in SO3")

        # case 1: R = I
        if np.allclose(R, np.identity(3)):
            return np.zeros((3,3)), 0
        
        # case 2: tr(R) = -1
        if abs(np.trace(R) + 1) < 1e-9:
            theta = np.pi
            if abs(1 + R[2, 2]) > 1e-9:
                w = (1 / np.sqrt(2 * (1 + R[2, 2]))) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            elif abs(1 + R[1, 1]) > 1e-9:
                w = (1 / np.sqrt(2 * (1 + R[1, 1]))) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            else:
                w = (1 / np.sqrt(2 * (1 + R[0, 0]))) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
            return skew_symmetric(*w), theta
        
        # otherwise
        theta = np.arccos(0.5 * (np.trace(R) - 1))
        w = (R - R.T)/(2*sin(theta))
        return w, theta
