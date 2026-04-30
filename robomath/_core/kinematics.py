import math
from typing import List
from numpy.typing import NDArray
import numpy as np

from lie import SE3, se3

class ForwardKin:

    @staticmethod
    def fk_space(M: NDArray, S_list: NDArray, theta: NDArray):
        '''
        INPUT:
        M: home state when all angles are 0 in SE3
        S_list: all the screw axis in se3 of individual joints of the system
        theta: angle vector of all the joints

        above indexed at i: 0 to n-1

        OUTPUT:
        space form in SE3
        '''

        if not SE3.SE3_test(M):
            raise ValueError("The provided HOME STATE is not defined in SE3")
        if len(S_list) != len(theta):
            raise ValueError("Size mismatch for given Screw Axis list and theta vector")
        

        T = M
        if np.allclose(theta, 0, atol=1e-9):
            return T

        for i in range(len(theta)-1,-1,-1):

            if abs(theta[i]) <= 1e-9:
                continue
            
            if not se3.se3_test(S_list[i]):
                raise ValueError(f"The {{i}}-th Screw Axis is not in se3")
            
            T = se3.mat_exp6(S_list[i], theta[i]) @ T

        return T
    
    @staticmethod
    def fk_body(M: NDArray, B_list: NDArray, theta: NDArray):
        '''
        INPUT:
        M: home state when all angles are 0 in SE3
        B_list: all the screw axis in se3 of individual joints of the system
        theta: angle vector of all the joints

        above indexed at i: 0 to n-1

        OUTPUT:
        space form in SE3
        '''

        if not SE3.SE3_test(M):
            raise ValueError("The provided HOME STATE is not defined in SE3")
        if len(B_list) != len(theta):
            raise ValueError("Size mismatch for given Screw Axis list and theta vector")
        

        T = M
        if np.allclose(theta, 0, atol=1e-9):
            return T
        
        for i in range(len(theta)):
            if abs(theta[i]) <= 1e-9:
                continue
            if not se3.se3_test(B_list[i]):
                raise ValueError(f"The {{i}}-th Screw Axis is not in se3")
            T = T @ se3.mat_exp6(B_list[i], theta[i])
            
        return T
    
    @staticmethod
    def fk_consistency(M: NDArray, S_list: NDArray, theta: NDArray) -> bool:
        """
        INPUT:
        M: home state when all angles are 0 in SE3
        S_list: all the screw axis in se3 of individual joints of the system
        theta: angle vector of all the joints

        OUTPUT:
        bool
        """

        M_inv = SE3.trans_inverse(M)
        Ad_Minv = SE3.adjoint(M_inv)           # 6x6

        B_list = np.array([
            se3.vec_to_se3(Ad_Minv @ se3.se3_to_vec(S))
            for S in S_list
        ])

        T_space = ForwardKin.fk_space(M, S_list, theta)
        T_body  = ForwardKin.fk_body(M, B_list, theta)

        if np.allclose(T_space, T_body, atol=1e-6):
            return True
        
        return False