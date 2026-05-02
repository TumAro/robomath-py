from typing import List
from numpy.typing import NDArray
import numpy as np

from robomath.groups.SE3 import SE3
from robomath.tangent.se3 import se3


def fwd_space(M: SE3, S_list: List[se3], theta: NDArray) -> SE3:
    '''
    INPUT:
    M: home state when all angles are 0 in SE3
    S_list: all the screw axis in se3 of individual joints of the system
    theta: angle vector of all the joints

    above indexed at i: 0 to n-1

    OUTPUT:
    space form in SE3
    '''

    if len(S_list) != theta.shape[0]:
        raise ValueError("Size mismatch for given Screw Axis list and theta vector")
    
    T = M
    if np.allclose(theta, 0, atol=1e-9):
        return T

    for i in range(len(theta)-1,-1,-1):
        if abs(theta[i]) <= 1e-9:
            continue
        T = SE3(S_list[i].exp(theta[i])) @ T

    return T

def fwd_body(M: SE3, B_list: List[se3], theta: NDArray) -> SE3:
    '''
    INPUT:
    M: home state when all angles are 0 in SE3
    B_list: all the screw axis in se3 of individual joints of the system
    theta: angle vector of all the joints

    above indexed at i: 0 to n-1

    OUTPUT:
    space form in SE3
    '''

    if len(B_list) != len(theta):
        raise ValueError("Size mismatch for given Screw Axis list and theta vector")
    
    T = M
    if np.allclose(theta, 0, atol=1e-9):
        return T
    
    for i in range(len(theta)):
        if abs(theta[i]) <= 1e-9:
            continue
        T = T @ SE3(B_list[i].exp(theta[i])) 
        
    return T