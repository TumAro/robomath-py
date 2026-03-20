import numpy as np
from math import sin, cos
from numpy.typing import NDArray
from typing import List

class SO2:
    identity = np.identity(2, dtype=np.float32)

    @staticmethod
    def so2_test(R: NDArray) -> bool:
        '''
        (R^T)R = I
        det(R) = 1
        '''
        if R.T*R != SO2.identity:
            return False
        if np.linalg.det(R) != 1:
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
    def so3_test(R: NDArray) -> bool:
        '''
        (R^T)R = I
        det(R) = 1
        '''
        if R.T*R != SO3.identity:
            return False
        if np.linalg.det(R) != 1:
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
        if SO3.so3_test(R):
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