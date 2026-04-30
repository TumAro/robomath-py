import numpy as np
from math import sin, cos
from numpy.typing import NDArray
from typing import List, Tuple


# * SPECIAL ORTHOGONAL GROUP
class SO2:
    identity = np.identity(2, dtype=np.float32)

    @staticmethod
    def SO2_test(R: NDArray) -> bool:
        '''
        Tests whether a matrix belongs to SO(2).

        INPUT:
        R : (2x2 NDArray) — candidate rotation matrix

        OUTPUT:
        bool — True if R ∈ SO(2), i.e. R^T R = I and det(R) = 1
        '''
        if not np.allclose(R.T @ R, SO2.identity):
            return False
        if abs(np.linalg.det(R) - 1) > 1e-6:
            return False
        
        return True

    @staticmethod
    def rot_inv(R: NDArray) -> NDArray[np.float32]:
        '''
        Computes the inverse of a rotation matrix in SO(2).

        INPUT:
        R : (2x2 NDArray) — rotation matrix, R ∈ SO(2)

        OUTPUT:
        R^-1 : (2x2 NDArray) — inverse rotation, R^-1 = R^T ∈ SO(2)
               returns identity if R ∉ SO(2)
        '''
        if SO2.SO2_test(R):
            return R.T

        return SO2.identity
    
    @staticmethod
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

class SO3:
    identity = np.identity(3, dtype=np.float32)

    def _check_square(self, R: NDArray) -> bool:
        r, c = R.shape
        if r != c:
            raise ValueError("NOT a square Matrix")
        return r==c
    
    @staticmethod
    def SO3_test(R: NDArray) -> bool:
        '''
        Tests whether a matrix belongs to SO(3).

        INPUT:
        R : (3x3 NDArray) — candidate rotation matrix

        OUTPUT:
        bool — True if R ∈ SO(3), i.e. R^T R = I and det(R) = 1
        '''
        if not np.allclose(R.T @ R, SO3.identity):
            return False
        if abs(np.linalg.det(R) - 1) > 1e-6:
            return False
        
        return True
  
    @staticmethod
    def transpose(R: List[List[float]]):
        '''
        Computes the transpose of a matrix represented as nested Python lists.

        INPUT:
        R : List[List[float]] — matrix as nested lists

        OUTPUT:
        R^T : List[List[float]] — transpose of R
        '''
        return [list(row) for row in zip(*R)]
    
    @staticmethod
    def rot_inv(R: NDArray) -> NDArray[np.float32]:
        '''
        Computes the inverse of a rotation matrix in SO(3).

        INPUT:
        R : (3x3 NDArray) — rotation matrix, R ∈ SO(3)

        OUTPUT:
        R^-1 : (3x3 NDArray) — inverse rotation, R^-1 = R^T ∈ SO(3)
               returns identity if R ∉ SO(3)
        '''
        if SO3.SO3_test(R):
            return R.T

        return SO3.identity
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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

    @staticmethod
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
        
        skew_omega = so3.skew_symmetric(*w)

        rot = np.identity(3) + sin(theta)*skew_omega + (1-cos(theta))*(skew_omega @ skew_omega)
        return rot

    @staticmethod
    def skew_mat_to_SO3(w_theta: NDArray) -> NDArray:
        '''
        Matrix exponential: maps a scaled skew-symmetric matrix to SO(3).

        INPUT:
        w_theta : (3x3 NDArray) — scaled skew-symmetric matrix [ω]θ ∈ so(3)

        OUTPUT:
        R : (3x3 NDArray) — rotation matrix R = e^([ω]θ) ∈ SO(3)
        '''
        w_vec = so3.skew_to_vec(w_theta)
        theta = float(np.linalg.norm(w_vec))
        w_hat = [float(x / theta) for x in w_vec]

        return SO3.rodrigues(w_hat, theta)

class so3:
    @staticmethod
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
    
    @staticmethod
    def check_skew_symmetry(matrix: NDArray) -> bool:
        '''
        Tests whether a matrix belongs to so(3).

        INPUT:
        matrix : (3x3 NDArray) — candidate skew-symmetric matrix

        OUTPUT:
        bool — True if matrix ∈ so(3), i.e. matrix = -matrix^T
        '''
        return np.allclose(matrix.T, -matrix)
    
    @staticmethod
    def skew_to_vec(matrix: NDArray) -> List[float]:
        '''
        Extracts the 3D vector from a skew-symmetric matrix.

        INPUT:
        matrix : (3x3 NDArray) — skew-symmetric matrix [x] ∈ so(3)

        OUTPUT:
        x : List[float] — vector x ∈ R^3 such that [x] = matrix
        '''
        if not so3.check_skew_symmetry(matrix):
            raise ValueError("Not a skew symmetrix matrix!")
        
        x1 = matrix[2][1]
        x2 = matrix[0][2]
        x3 = matrix[1][0]

        return [x1, x2, x3]
    
    @staticmethod
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
            return so3.skew_symmetric(*w), theta
        
        # otherwise
        theta = np.arccos(0.5 * (np.trace(R) - 1))
        w = (R - R.T)/(2*sin(theta))
        return w, theta


# * SPECIAL EUCLIDEAN GROUP
class SE2:
    @staticmethod
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
        if not SO2.SO2_test(T[:2, :2]):
            return False
        if not np.allclose(T[2], [0, 0, 1]):
            return False
        return True

    @staticmethod
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
        T[:2, :2] = SO2.rot_matrix(theta)
        T[:2, 2] = p
        return T

    @staticmethod
    def trans_inverse(T: NDArray) -> NDArray:
        '''
        Computes the inverse of a transformation matrix in SE(2).

        INPUT:
        T : (3x3 NDArray) — transformation matrix T ∈ SE(2)

        OUTPUT:
        T^-1 : (3x3 NDArray) — inverse transformation T^-1 ∈ SE(2)
               computed as [R^T, -R^T p; 0, 1]
        '''
        if not SE2.SE2_test(T):
            raise ValueError("Not a valid transformation matrix in SE2")
        R = T[:2, :2]
        p = T[:2, 2]
        invT = np.eye(3)
        invT[:2, :2] = R.T
        invT[:2, 2] = -(R.T) @ p
        return invT

class SE3:
    @staticmethod
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
        if not SO3.SO3_test(T[:3, :3]):
            return False
        if not np.allclose(T[3], [0, 0, 0, 1]):
            return False
        return True

    @staticmethod
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
    
    @staticmethod
    def get_rotation(T: NDArray) -> NDArray:
        '''
        Extracts the rotation matrix from a transformation matrix.

        INPUT:
        T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

        OUTPUT:
        R : (3x3 NDArray) — rotation component R ∈ SO(3)
        '''
        if not SE3.SE3_test(T):
            raise ValueError("Not a valid transformation matrix in SE3")
        return T[:3, :3]
    
    @staticmethod
    def get_translation(T: NDArray) -> NDArray:
        '''
        Extracts the translation vector from a transformation matrix.

        INPUT:
        T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

        OUTPUT:
        p : (3,) NDArray — translation component p ∈ R^3
        '''
        if not SE3.SE3_test(T):
            raise ValueError("Not a valid transformation matrix in SE3")
        return T[:3, 3]
    
    @staticmethod
    def rotation(w: list, theta: float) -> NDArray:
        '''
        Constructs a pure rotation transformation matrix (no translation).

        INPUT:
        w     : List[float] — unit rotation axis, ω ∈ R^3, ‖ω‖ = 1
        theta : float       — angle of rotation in radians, θ ∈ R

        OUTPUT:
        T : (4x4 NDArray) — transformation matrix T ∈ SE(3) with p = 0
        '''
        T = np.eye(4)
        T[:3, :3] = SO3.rodrigues(w, theta)
        return T
    
    @staticmethod
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
    
    @staticmethod
    def trans_inverse(T: NDArray) -> NDArray:
        '''
        Computes the inverse of a transformation matrix in SE(3).

        INPUT:
        T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

        OUTPUT:
        T^-1 : (4x4 NDArray) — inverse transformation T^-1 ∈ SE(3)
               computed as [R^T, -R^T p; 0, 1]
        '''
        if not SE3.SE3_test(T):
            raise ValueError("Not a valid transformation matrix in SE3")
        R = T[:3, :3]
        p = T[:3, 3]

        invT = np.eye(4)
        invT[:3,:3] = R.T
        invT[:3, 3] = -(R.T)@p

        return invT

    @staticmethod
    def adjoint(T: NDArray) -> NDArray:
        '''
        Computes the adjoint representation of a transformation matrix.

        INPUT:
        T : (4x4 NDArray) — transformation matrix T ∈ SE(3)

        OUTPUT:
        [Ad_T] : (6x6 NDArray) — adjoint matrix, [Ad_T] ∈ R^(6x6)
                 used to change the reference frame of a twist: V_a = [Ad_T] V_b
        '''
        if not SE3.SE3_test(T):
            raise ValueError("Input is not a valid SE3 transformation matrix")

        R = T[:3,:3]
        p = T[:3, 3]

        adj = np.zeros((6,6))

        adj[:3,:3] = R
        adj[3:,3:] = R
        adj[3:, :3] = so3.skew_symmetric(*p) @ R

        return adj

    @staticmethod
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
        return SE3.adjoint(T).T @ F

    @staticmethod
    def compose(A: NDArray, B: NDArray) -> NDArray:
        '''
        INPUT
        A, B in SE3

        OUTPUT
        R = AB in SE3
        '''
        if not SE3.SE3_test(A) or not SE3.SE3_test(B):
            raise ValueError("Not a valid Transformation Matrix")
        
        R_A = SE3.get_rotation(A)
        p_A = SE3.get_translation(A)

        R_B = SE3.get_rotation(B)
        p_B = SE3.get_translation(B)

        R = np.eye(4)

        R[:3,:3] = R_A @ R_B
        R[:3, 3] = R_A @ p_B + p_A

        return R
        


class se3:

    @staticmethod
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
        if not so3.check_skew_symmetry(mat[:3, :3]):
            return False
        return True
    
    @staticmethod
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

        w_so3 = so3.skew_symmetric(*w)

        result = np.zeros((4,4))
        result[:3,:3] = w_so3
        result[:3,3] = v

        return result

    @staticmethod
    def se3_to_vec(mat: NDArray) -> NDArray:
        '''
        Extracts the twist vector from an se(3) matrix.

        INPUT:
        [V] : (4x4 NDArray) — twist matrix [V] ∈ se(3)

        OUTPUT:
        V : (6,) NDArray — twist vector V = (ω, v) ∈ R^6
        '''
        if not se3.se3_test(mat):
            raise ValueError("Input is not a valid se3 matrix")

        w = so3.skew_to_vec(mat[:3,:3])
        v = mat[:3, 3]

        return np.concatenate([w, v])
    
    @staticmethod
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
        if not so3.check_skew_symmetry(w):
            raise ValueError("Matrix provided is not a skew symmetric matrix")

        exp = np.eye(4)
        if np.abs(np.linalg.norm(so3.skew_to_vec(w)) - 1) < 1e-9:
            
            exp[:3, :3] = SO3.rodrigues(list(so3.skew_to_vec(w)), theta)
            exp[:3, 3] = (np.eye(3)*theta + (1-cos(theta))*w + (theta - sin(theta)) * (w @ w)  ) @ v

            return exp
        
        elif np.linalg.norm(so3.skew_to_vec(w)) < 1e-9 and np.abs(np.linalg.norm(v) - 1) < 1e-9:

            exp[:3, 3] = v * theta

            return exp
        
        else:
            raise ValueError("Invalid screw axis: must have ‖ω‖=1, or ω=0 and ‖v‖=1")

    @staticmethod
    def mat_exp6(se3mat: NDArray, theta: float) -> NDArray:
        '''
        Matrix exponential from an se(3) matrix and angle to SE(3).

        INPUT:
        se3mat : (4x4 NDArray) — twist matrix [S] ∈ se(3)
        theta  : float         — angle/distance of motion, θ ∈ R

        OUTPUT:
        T : (4x4 NDArray) — transformation matrix T = e^([S]θ) ∈ SE(3)
        '''
        if not se3.se3_test(se3mat):
            raise ValueError("Not a se3 matrix")
        
        return se3.screw_exp6(se3mat[:3, :3], se3mat[:3, 3], theta)

    @staticmethod
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

        if not SE3.SE3_test(T):
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
            w, theta = so3.logarithm(R)
            v = ( np.eye(3)/theta - w/2 + (1/theta - cos(theta/2)/(2*sin(theta/2))) * (w @ w) ) @ p

            return (w, v, theta)
            
    @staticmethod
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

    @staticmethod
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
