from robomath import *
import numpy as np
from math import sqrt, pi
from robomath.kinematics import fwd_body, fwd_space

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("ex 1========================")
    M = SE3(np.array([
        [1,0,0,(2+sqrt(3))],
        [0,1,0,0],
        [0,0,1,(sqrt(3)+1)],
        [0,0,0,1]
    ]))
    print(M)

    print("ex 2========================")
    S_list_axis = [
        np.array([0,0,1,0,-1,0]),
        np.array([0,1,0,0,0,1]),
        np.array([0,1,0,1,0,sqrt(3)+1]),
        np.array([0,1,0,-sqrt(3)+1,0,2+sqrt(3)]),
        np.array([0,0,0,0,0,1]),
        np.array([0,0,1,0,-2-sqrt(3),0]),
    ]
    print(np.array(S_list_axis).T)

    print("ex 3=========================")
    B_list_axis = [
        np.array([0,0,1,0,sqrt(3)+1,0]),
        np.array([0,1,0,sqrt(3)+1,0,-(sqrt(3)+1)]),
        np.array([0,1,0,sqrt(3)+2,0,-1]),
        np.array([0,1,0,2,0,0]),
        np.array([0,0,0,0,0,1]),
        np.array([0,0,1,0,0,0])
    ]
    print(np.array(B_list_axis).T)

    print("ex 4========================")
    S_list = [se3.from_vec(vec) for vec in S_list_axis]
    theta = np.array([-pi/2, pi/2, pi/3, -pi/4,1,pi/6])
    ans4 = fwd_space(M, S_list, theta)
    print(ans4)

    print("ex 5========================")

    B_list = [se3.from_vec(vec) for vec in B_list_axis]
    ans5 = fwd_body(M,B_list,theta)
    print(ans5)