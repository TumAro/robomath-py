import numpy as np
from math import sin, cos
from numpy.typing import NDArray
from typing import List, Tuple

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
    
class SE3:
    @staticmethod
    def transform(R: NDArray, p: NDArray) -> NDArray:

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p

        return T
    
    @staticmethod
    def getRotation(T: NDArray) -> NDArray:

        return T[:3, :3]
    
    @staticmethod
    def getTranslation(T: NDArray) -> NDArray:

        return T[:3, 3]
    
    @staticmethod
    def rotation(w: list, theta: float) -> NDArray:
        T = np.eye(4)
        T[:3, :3] = SO3.rodrigues(w, theta)
        return T
    
    @staticmethod
    def translation(p: NDArray) -> NDArray:

        T = np.eye(4)
        T[:3, 3] = p
        return T
    
    @staticmethod
    def trans_inverse(T: NDArray) -> NDArray:
        R = T[:3, :3]
        p = T[:3, 3]

        invT = np.eye(4)
        invT[:3,:3] = R.T
        invT[:3, 3] = -(R.T)@p

        return invT


'''
- [ ] **`rp_to_trans(R, p)`** — §3.3.1, Def 3.13 — builds 4×4 T from R, p. (Ex 3.16b)
- [ ] **`trans_to_rp(T)`** — extracts R, p from T.
- [ ] **`trans_inv(T)`** — §3.3.1.1, Prop 3.15 — T⁻¹ without numpy.linalg.inv. (Ex 3.16c)
'''


