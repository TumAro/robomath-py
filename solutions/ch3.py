from src.rigid_body import so3, SO3
import numpy as np
from modern_robotics import VecToso3, MatrixExp3, MatrixLog3

if __name__ == "__main__":
    # assignments ------------------------

    # q9
    # w_theta = np.array([1,2,0])
    # theta = float(np.linalg.norm(w_theta))
    # w_hat = list(w_theta / theta)
    # print(SO3.rodrigues(w_hat, theta=theta))
    mat1 = VecToso3([1,2,0])
    matR = MatrixExp3(mat1)
    print(matR)

    # q10
    print(so3.skew_symmetric(*[1,2,0.5]))

    # q11
    # print(SO3.skew_mat_to_SO3(np.array([
    #     [0, 0.5, -1],
    #     [-0.5, 0, 2],
    #     [1, -2, 0]
    # ])))

    print(MatrixExp3(np.array([
        [0, 0.5, -1],
        [-0.5, 0, 2],
        [1, -2, 0]
    ])))

    # q12
    M, th = so3.logarithm(np.array([
        [0,0,1],
        [-1,0,0],
        [0,-1,0]
    ]))

    print(M*th)
    
    print(MatrixLog3(np.array([
        [0,0,1],
        [-1,0,0],
        [0,-1,0]
    ])))

