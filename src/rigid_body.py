import numpy as np
from math import sin, cos
from numpy.typing import NDArray
from typing import List, Tuple


# * SPECIAL ORTHOGONAL GROUP
class SO2:
    identity = np.identity(2, dtype=np.float32)

    @staticmethod
    def so2_test(R: NDArray) -> bool:
        '''
        (R^T)R = I
        det(R) = 1
        '''
        if not np.allclose(R.T @ R, SO2.identity):
            return False
        if abs(np.linalg.det(R) - 1) > 1e-6:
            return False
        
        return True

    @staticmethod
    def rot_inv(R: NDArray) -> NDArray[np.float32]:
        '''
        if the given matrix is a rotation matrix ->
        R^-1 = R^T
        else -> return identity
        '''
        if SO2.so2_test(R):
            return R.T

        return SO2.identity
    
    @staticmethod
    def rot_matrix(theta: float) -> NDArray[np.float32]:
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
        (R^T)R = I
        det(R) = 1
        '''
        if not np.allclose(R.T @ R, SO3.identity):
            return False
        if abs(np.linalg.det(R) - 1) > 1e-6:
            return False
        
        return True
  
    @staticmethod
    def transpose(R: List[List[float]]):
        '''
        OUTPUT: Python Lists
        outputs the transpose of the matrix
        '''
        return [list(row) for row in zip(*R)]
    
    @staticmethod
    def rot_inv(R: NDArray) -> NDArray[np.float32]:
        '''
        if the given matrix is a rotation matrix ->
        R^-1 = R^T
        '''
        if SO3.SO3_test(R):
            return R.T

        return SO3.identity
    
    @staticmethod
    def rot_x(theta: float) -> NDArray[np.float32]:
        '''
        Rotation matrix about X-axes
        '''
        return np.array([
            [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)]
        ])
    
    @staticmethod
    def rot_y(theta: float) -> NDArray[np.float32]:
        '''
        Rotation matrix about Y-axes
        '''
        return np.array([
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)]
        ])
    
    @staticmethod
    def rot_z(theta: float) -> NDArray[np.float32]:
        '''
        Rotation matrix about Z-axes
        '''
        return np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0,0,1]
        ])

    @staticmethod
    def rodrigues(w: List, theta: float) -> NDArray:
        '''
        INPUT:
        w -> unit axis of rotation in 3D
        theta -> rotated how long

        OUTPUT:
        Rotation matrix in SO3
        '''

        if len(w) != 3:
            raise ValueError("Rotation axis is not in 3D")
        
        if abs(sum(i*i for i in w) - 1.0) > 1e-9:
            raise ValueError("Rotation axis is not unit vector.")
        
        skew_omega = so3.skew_symmetric(*w)

        rot = np.identity(3) + sin(theta)*skew_omega + (1-cos(theta))*(skew_omega @ skew_omega)
        return rot

    @staticmethod
    def skewmat2SO3(w_theta: NDArray) -> NDArray:
        w_vec = so3.skew2vec(w_theta)
        theta = float(np.linalg.norm(w_vec))
        w_hat = [float(x / theta) for x in w_vec]

        return SO3.rodrigues(w_hat, theta)

class so3:
    @staticmethod
    def skew_symmetric(x1, x2, x3) -> NDArray:
        '''
        INPUT
        vector x = [x1 x2 x3]^T in R^3
        
        OUTPUT
        skew symmetric matrix [x]
        '''
        return np.array([
            [0, -x3, x2],
            [x3 , 0, -x1],
            [-x2, x1, 0]
        ])
    
    @staticmethod
    def check_skew_symmetry(matrix: NDArray) -> bool:
        '''
        [x] = -[x]^T
        '''
        return np.allclose(matrix.T, -matrix)
    
    @staticmethod
    def skew2vec(matrix: NDArray) -> List[float]:
        if not so3.check_skew_symmetry(matrix):
            raise ValueError("Not a skew symmetrix matrix!")
        
        x1 = matrix[2][1]
        x2 = matrix[0][2]
        x3 = matrix[1][0]

        return [x1, x2, x3]
    
    @staticmethod
    def logarithm(R: NDArray) -> Tuple[NDArray, float]:
        '''
        INPUT:
        R -> rotation matrix in SO3 (3x3 NDArray)

        OUTPUT:
        skew_omega -> skew symmetric matrix of the rotation axis (3x3 NDArray)
        theta      -> angle of rotation in radians (float)
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
        T is in SE2 if:
        - shape is (3x3)
        - top-left 2x2 is a valid rotation matrix in SO2 (R^T R = I, det(R) = 1)
        - bottom row is [0, 0, 1]
        '''
        if T.shape != (3, 3):
            return False
        if not SO2.so2_test(T[:2, :2]):
            return False
        if not np.allclose(T[2], [0, 0, 1]):
            return False
        return True

    @staticmethod
    def transform(theta: float, p: NDArray) -> NDArray:
        '''
        INPUT:
        theta -> angle of rotation in radians
        p     -> position of the new frame origin expressed in the original frame (2, NDArray)

        OUTPUT:
        T -> homogeneous transformation matrix in SE2 (3x3 NDArray)
        '''
        T = np.eye(3)
        T[:2, :2] = SO2.rot_matrix(theta)
        T[:2, 2] = p
        return T

    @staticmethod
    def trans_inverse(T: NDArray) -> NDArray:
        '''
        Computes the inverse of a 2D transformation matrix analytically
        using T^-1 = [R^T, -R^T p; 0, 1] instead of np.linalg.inv

        INPUT:
        T -> homogeneous transformation matrix in SE2 (3x3 NDArray)

        OUTPUT:
        invT -> inverse transformation matrix in SE2 (3x3 NDArray)
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
        T is in SE3 if:
        - shape is (4x4)
        - top-left 3x3 is a valid rotation matrix (R^T R = I, det(R) = 1)
        - bottom row is [0, 0, 0, 1]
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
        INPUT:
        R -> rotation matrix in SO3 (3x3 NDArray)
        p -> position of the new frame origin expressed in the original frame (3, NDArray)

        OUTPUT:
        T -> homogeneous transformation matrix in SE3 (4x4 NDArray)
        '''
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p

        return T
    
    @staticmethod
    def getRotation(T: NDArray) -> NDArray:
        '''
        INPUT:
        T -> homogeneous transformation matrix in SE3 (4x4 NDArray)

        OUTPUT:
        R -> rotation part of the transformation (3x3 NDArray)
        '''
        if not SE3.SE3_test(T):
            raise ValueError("Not a valid transformation matrix in SE3")
        return T[:3, :3]
    
    @staticmethod
    def getTranslation(T: NDArray) -> NDArray:
        '''
        INPUT:
        T -> homogeneous transformation matrix in SE3 (4x4 NDArray)

        OUTPUT:
        p -> translation vector, origin of the new frame in the original frame (3, NDArray)
        '''
        if not SE3.SE3_test(T):
            raise ValueError("Not a valid transformation matrix in SE3")
        return T[:3, 3]
    
    @staticmethod
    def rotation(w: list, theta: float) -> NDArray:
        '''
        Pure rotation transformation (no translation)

        INPUT:
        w     -> unit axis of rotation in 3D
        theta -> angle of rotation in radians

        OUTPUT:
        T -> transformation matrix with rotation only, translation is zero (4x4 NDArray)
        '''
        T = np.eye(4)
        T[:3, :3] = SO3.rodrigues(w, theta)
        return T
    
    @staticmethod
    def translation(p: NDArray) -> NDArray:
        '''
        Pure translation transformation (no rotation)

        INPUT:
        p -> displacement vector in R^3 (3, NDArray)

        OUTPUT:
        T -> transformation matrix with translation only, rotation is identity (4x4 NDArray)
        '''
        T = np.eye(4)
        T[:3, 3] = p
        return T
    
    @staticmethod
    def trans_inverse(T: NDArray) -> NDArray:
        '''
        Computes the inverse of a transformation matrix analytically
        using T^-1 = [R^T, -R^T p; 0, 1] instead of np.linalg.inv

        INPUT:
        T -> homogeneous transformation matrix in SE3 (4x4 NDArray)

        OUTPUT:
        invT -> inverse transformation matrix in SE3 (4x4 NDArray)
        '''
        if not SE3.SE3_test(T):
            raise ValueError("Not a valid transformation matrix in SE3")
        R = T[:3, :3]
        p = T[:3, 3]

        invT = np.eye(4)
        invT[:3,:3] = R.T
        invT[:3, 3] = -(R.T)@p

        return invT
 
class se3:

    @staticmethod
    def check_se3(mat: NDArray) -> bool:
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
        INPUT:
        Vector from R^6 where V = (w, v)

        OUTPUT:
        Twist in se3
        '''
        w, v = V[:3], V[3:]

        w_so3 = so3.skew_symmetric(*w)

        result = np.zeros((4,4))
        result[:3,:3] = w_so3
        result[:3,3] = v

        return result

    @staticmethod
    def se3_to_vec(mat: NDArray) -> NDArray:
        '''
        INPUT:
        4x4 se3 matrix [V]

        OUTPUT:
        V = (w,v) in R^6
        '''

        w = so3.skew2vec(mat[:3,:3])
        v = mat[:3, 3]

        return np.concatenate(w,v)
    
    @staticmethod
    def adjoint(T: NDArray) -> NDArray:
        '''
        INPUT:
        4x4 Tranformation Matrix

        OUTPUT:
        adjoint matrix [Ad_T]
        '''

        R = T[:3,:3]
        p = T[:3, 3]

        adj = np.zeros((6,6))

        adj[:3,:3] = R
        adj[3:,3:] = R
        adj[3:, :3] = so3.skew_symmetric(*p) @ R

        return adj

    @staticmethod
    def screw_exp6(w: NDArray, v: NDArray, theta: float) -> NDArray:
        '''
        INPUT:
        let S = (w,v) screw axis

        OUTPUT:
        exponential coordinate
        '''
        if not so3.check_skew_symmetry(w):
            raise ValueError("Matrix provided is not a skew symmetric matrix")

        exp = np.eye(4)
        if np.abs(np.linalg.norm(so3.skew2vec(w)) - 1) < 1e-9:
            
            exp[:3, :3] = SO3.rodrigues(list(so3.skew2vec(w)), theta)
            exp[:3, 3] = (np.eye(3)*theta + (1-cos(theta))*w + (theta - sin(theta)) * (w @ w)  ) @ v

            return exp
        
        elif np.linalg.norm(so3.skew2vec(w)) < 1e-9 and np.abs(np.linalg.norm(v) - 1) < 1e-9:

            exp[:3, 3] = v * theta

            return exp
        
        else:
            raise ValueError("Invalid screw axis: must have ‖ω‖=1, or ω=0 and ‖v‖=1")

    @staticmethod
    def mat_exp6(se3mat: NDArray, theta: float) -> NDArray:
        if not se3.check_se3(se3mat):
            raise ValueError("Not a se3 matrix")
        
        return se3.screw_exp6(se3mat[:3, :3], se3mat[:3, 3], theta)

    @staticmethod
    def logarithm6(T: NDArray) -> Tuple[NDArray, NDArray, float]:
        '''
        INPUT:
        T in SE3

        OUPUT:
        [S]theta - ([w], v, theta) in se3
        '''

        if not SE3.SE3_test(T):
            raise ValueError("Not a valid SE3 matrix.")
        
        R = T[:3, :3]
        p = T[:3, 3]

        if np.allclose(R, np.eye(3)):
            w = np.zeros((3,3))
            theta = float(np.linalg.norm(p))
            v = p / theta
            return (w, v, theta)

        else:
            w, theta = so3.logarithm(R)
            v = ( np.eye(3)/theta - w/2 + (1/theta - cos(theta/2)/(2*sin(theta/2))) * (w @ w) ) @ p

            return (w, v, theta)
            


'''
- [ ] **`screw_to_axis(q, s_hat, h)`** —Definition 3.24 - §3.3.2.2, Def 3.24 — builds S from {q, ŝ, h}. (Ex 3.26, 3.27)
'''