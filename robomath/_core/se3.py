import numpy as np
from math import sin, cos
from numpy.typing import NDArray
from typing import Tuple

from robomath._core.so3 import (
    check_skew_symmetry,
    logarithm,
    rodrigues,
    skew_symmetric,
    skew_to_vec,
    SO3_test,
)

# * IMPLEMENTATION OF ==> SE GROUP

def SE3_test(T: NDArray) -> bool:
    '''
    Tests whether a matrix belongs to SE(3).

    INPUT:
    T : (4x4 NDArray) — candidate transformation matrix

    OUTPUT:
    bool — True if T ∈ SE(3), i.e. top-left 3x3 ∈ SO(3) and bottom row = [0, 0, 0, 1]
    '''
    if T.shape != (4, 4):
        return False
    if not SO3_test(T[:3, :3]):
        return False
    if not np.allclose(T[3], [0, 0, 0, 1]):
        return False
    return True

def transform(R: NDArray, p: NDArray) -> NDArray:
    '''
    Constructs a 3D homogeneous transformation matrix from a rotation and translation.

    INPUT:
    R : (3x3 NDArray) — rotation matrix R ∈ SO(3)
    p : (3,) NDArray  — translation vector p ∈ R^3

    OUTPUT:
    T : (4x4 NDArray) — homogeneous transformation matrix T ∈ SE(3)
    '''
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T

def get_rotation(T: NDArray) -> NDArray:
    '''
    Extracts the rotation matrix from a transformation matrix.

    INPUT:
    T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

    OUTPUT:
    R : (3x3 NDArray) — rotation component R ∈ SO(3)
    '''
    if not SE3_test(T):
        raise ValueError("Not a valid transformation matrix in SE3")
    return T[:3, :3]

def get_translation(T: NDArray) -> NDArray:
    '''
    Extracts the translation vector from a transformation matrix.

    INPUT:
    T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

    OUTPUT:
    p : (3,) NDArray — translation component p ∈ R^3
    '''
    if not SE3_test(T):
        raise ValueError("Not a valid transformation matrix in SE3")
    return T[:3, 3]

def rotation(w: NDArray, theta: float) -> NDArray:
    '''
    Constructs a pure rotation transformation matrix (no translation).

    INPUT:
    w     : List[float] — unit rotation axis, ω ∈ R^3, ‖ω‖ = 1
    theta : float       — angle of rotation in radians, θ ∈ R

    OUTPUT:
    T : (4x4 NDArray) — transformation matrix T ∈ SE(3) with p = 0
    '''
    T = np.eye(4)
    T[:3, :3] = rodrigues(w, theta)
    return T

def translation(p: NDArray) -> NDArray:
    '''
    Constructs a pure translation transformation matrix (no rotation).

    INPUT:
    p : (3,) NDArray — displacement vector p ∈ R^3

    OUTPUT:
    T : (4x4 NDArray) — transformation matrix T ∈ SE(3) with R = I
    '''
    T = np.eye(4)
    T[:3, 3] = p
    return T

def trans_inverse(T: NDArray) -> NDArray:
    '''
    Computes the inverse of a transformation matrix in SE(3).

    INPUT:
    T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

    OUTPUT:
    T^-1 : (4x4 NDArray) — inverse transformation T^-1 ∈ SE(3)
            computed as [R^T, -R^T p; 0, 1]
    '''
    if not SE3_test(T):
        raise ValueError("Not a valid transformation matrix in SE3")
    R = T[:3, :3]
    p = T[:3, 3]

    invT = np.eye(4)
    invT[:3,:3] = R.T
    invT[:3, 3] = -(R.T)@p

    return invT

def adjoint(T: NDArray) -> NDArray:
    '''
    Computes the adjoint representation of a transformation matrix.

    INPUT:
    T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

    OUTPUT:
    [Ad_T] : (6x6 NDArray) — adjoint matrix, [Ad_T] ∈ R^(6x6)
                used to change the reference frame of a twist: V_a = [Ad_T] V_b
    '''
    if not SE3_test(T):
        raise ValueError("Input is not a valid SE3 transformation matrix")

    R = T[:3,:3]
    p = T[:3, 3]

    adj = np.zeros((6,6))

    adj[:3,:3] = R
    adj[3:,3:] = R
    adj[3:, :3] = skew_symmetric(*p) @ R

    return adj

def wrench_transform(F: NDArray, T: NDArray) -> NDArray:
    '''
    Changes the reference frame of a wrench using the adjoint transpose.

    INPUT:
    F : (6,) NDArray  — wrench F_a ∈ R^6 expressed in frame {a}
    T : (4x4 NDArray) — transformation matrix T_{ab} ∈ SE(3) from {a} to {b}

    OUTPUT:
    F_b : (6,) NDArray — wrench expressed in frame {b} ∈ R^6
            F_b = [Ad_{T_{ab}}]^T F_a
    '''
    return adjoint(T).T @ F

def compose(A: NDArray, B: NDArray) -> NDArray:
        '''
        INPUT
        A, B in SE3

        OUTPUT
        R = AB in SE3
        '''
        if not SE3_test(A) or not SE3_test(B):
            raise ValueError("Not a valid Transformation Matrix")
        
        R_A = get_rotation(A)
        p_A = get_translation(A)

        R_B = get_rotation(B)
        p_B = get_translation(B)

        R = np.eye(4)

        R[:3,:3] = R_A @ R_B
        R[:3, 3] = R_A @ p_B + p_A

        return R
        

# * IMPLEMENTATION OF ==> se algebra

def se3_test(mat: NDArray) -> bool:
    '''
    Tests whether a matrix belongs to se(3).

    INPUT:
    mat : (4x4 NDArray) — candidate Lie algebra matrix

    OUTPUT:
    bool — True if mat ∈ se(3), i.e. top-left 3x3 ∈ so(3) and bottom row = [0, 0, 0, 0]
    '''
    if mat.shape != (4, 4):
        return False
    if not np.allclose(mat[3, :], 0):
        return False
    if not check_skew_symmetry(mat[:3, :3]):
        return False
    return True

def vec_to_se3(V: NDArray) -> NDArray:
    '''
    Constructs an se(3) matrix from a twist vector.

    INPUT:
    V : (6,) NDArray — twist vector V = (ω, v) ∈ R^6

    OUTPUT:
    [V] : (4x4 NDArray) — twist matrix [V] ∈ se(3)
    '''
    if V.shape != (6,):
        raise ValueError("Input must be a vector in R^6")

    w, v = V[:3], V[3:]

    w_so3 = skew_symmetric(*w)

    result = np.zeros((4,4))
    result[:3,:3] = w_so3
    result[:3,3] = v

    return result

def se3_to_vec(mat: NDArray) -> NDArray:
    '''
    Extracts the twist vector from an se(3) matrix.

    INPUT:
    [V] : (4x4 NDArray) — twist matrix [V] ∈ se(3)

    OUTPUT:
    V : (6,) NDArray — twist vector V = (ω, v) ∈ R^6
    '''
    if not se3_test(mat):
        raise ValueError("Input is not a valid se3 matrix")

    w = skew_to_vec(mat[:3,:3])
    v = mat[:3, 3]

    return np.concatenate([w, v])

def screw_exp6(w: NDArray, v: NDArray, theta: float) -> NDArray:
    '''
    Matrix exponential: maps a screw axis and angle to SE(3).

    INPUT:
    w     : (3x3 NDArray) — skew-symmetric matrix [ω] ∈ so(3)
    v     : (3,) NDArray  — linear velocity component v ∈ R^3
    theta : float         — angle/distance of motion along the screw, θ ∈ R

    OUTPUT:
    T : (4x4 NDArray) — transformation matrix T = e^([S]θ) ∈ SE(3)
    '''
    if not check_skew_symmetry(w):
        raise ValueError("Matrix provided is not a skew symmetric matrix")

    exp = np.eye(4)
    if np.abs(np.linalg.norm(skew_to_vec(w)) - 1) < 1e-9:
        
        exp[:3, :3] = rodrigues(skew_to_vec(w), theta)
        exp[:3, 3] = (np.eye(3)*theta + (1-cos(theta))*w + (theta - sin(theta)) * (w @ w)  ) @ v

        return exp
    
    elif np.linalg.norm(skew_to_vec(w)) < 1e-9 and np.abs(np.linalg.norm(v) - 1) < 1e-9:

        exp[:3, 3] = v * theta

        return exp
    
    else:
        raise ValueError("Invalid screw axis: must have ‖ω‖=1, or ω=0 and ‖v‖=1")

def mat_exp6(se3mat: NDArray, theta: float) -> NDArray:
    '''
    Matrix exponential from an se(3) matrix and angle to SE(3).

    INPUT:
    se3mat : (4x4 NDArray) — twist matrix [S] ∈ se(3)
    theta  : float         — angle/distance of motion, θ ∈ R

    OUTPUT:
    T : (4x4 NDArray) — transformation matrix T = e^([S]θ) ∈ SE(3)
    '''
    if not se3_test(se3mat):
        raise ValueError("Not a se3 matrix")
    
    return screw_exp6(se3mat[:3, :3], se3mat[:3, 3], theta)

def logarithm6(T: NDArray) -> Tuple[NDArray, NDArray, float]:
    '''
    Matrix logarithm: maps a transformation matrix to se(3).

    INPUT:
    T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

    OUTPUT:
    [ω] : (3x3 NDArray) — skew-symmetric matrix [ω] ∈ so(3), rotation axis
    v   : (3,) NDArray  — linear component of the screw axis, v ∈ R^3
    θ   : float         — angle/distance of motion, θ ∈ [0, π]

    such that e^([S]θ) = T, where S = (ω, v)
    '''

    if not SE3_test(T):
        raise ValueError("Not a valid SE3 matrix.")
    
    R = T[:3, :3]
    p = T[:3, 3]

    if np.allclose(R, np.eye(3)):
        w = np.zeros((3,3))
        if np.allclose(p, np.zeros(3)):
            return (w, np.zeros(3), 0.0)
        theta = float(np.linalg.norm(p))
        v = p / theta
        return (w, v, theta)

    else:
        w, theta = logarithm(R)
        v = ( np.eye(3)/theta - w/2 + (1/theta - cos(theta/2)/(2*sin(theta/2))) * (w @ w) ) @ p

        return (w, v, theta)
        
def axis_to_screw(S: NDArray) -> Tuple[NDArray, NDArray, float]:
    '''
    Inverse of screw_to_axis: recovers {q, s_hat, h} from a normalized screw axis.

    INPUT:
    S : (6,) NDArray — normalized screw axis S = (ω, v) ∈ R^6
                        either ‖ω‖ = 1 (rotation) or ω = 0 and ‖v‖ = 1 (translation)

    OUTPUT:
    q     : (3,) NDArray — a point on the screw axis (closest point to origin)
    s_hat : (3,) NDArray — unit direction of the screw axis
    h     : float        — pitch of the screw (np.inf for pure translation)

    From the book (Def 3.24):
        Case ‖ω‖ = 1:  s_hat = ω,  h = ωᵀv,  q = ω × (v − h·ω)
        Case ω = 0:    s_hat = v,  h = ∞,     q = undefined (set to zeros)
    '''
    if S.shape != (6,):
        raise ValueError("S must be a vector in R^6")

    w, v = S[:3], S[3:]

    if abs(np.linalg.norm(w) - 1) < 1e-9:
        s_hat = w
        h = float(np.dot(w, v))
        q = np.cross(w, v - h * w)
        return q, s_hat, h

    elif np.linalg.norm(w) < 1e-9 and abs(np.linalg.norm(v) - 1) < 1e-9:
        s_hat = v
        h = np.inf
        q = np.zeros(3)
        return q, s_hat, h

    else:
        raise ValueError("S is not a normalized screw axis: need ‖ω‖=1 or ω=0 and ‖v‖=1")

def screw_to_axis(q: NDArray, s_hat: NDArray, h: float) -> NDArray:
        '''
        INPUT:
        q     : a point on the screw axis (R^3)
        s_hat : unit vector in the direction of the screw axis (R^3)
        h     : pitch of the screw (h=0 for pure rotation, h=inf for pure translation)

        OUTPUT:
        S = (w, v) in R^6 — normalized screw axis
        '''

        if q.shape != (3,):
            raise ValueError("q must be a point in R^3")
        if s_hat.shape != (3,):
            raise ValueError("s_hat must be a vector in R^3")
        if abs(np.linalg.norm(s_hat) - 1) > 1e-9:
            raise ValueError("s_hat must be a unit vector")

        if np.isinf(h):
            w = np.zeros(3)
            v = s_hat
        else:
            w = s_hat
            v = np.cross(-s_hat, q) + h * s_hat

        return np.concatenate([w, v])
