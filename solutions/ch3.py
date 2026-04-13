from src.rigid_body import so3, SO3
import numpy as np
from modern_robotics import VecToso3, MatrixExp3, MatrixLog3

if __name__ == "__main__":
    # assignments ------------------------

    # # q9
    # # w_theta = np.array([1,2,0])
    # # theta = float(np.linalg.norm(w_theta))
    # # w_hat = list(w_theta / theta)
    # # print(SO3.rodrigues(w_hat, theta=theta))
    # mat1 = VecToso3([1,2,0])
    # matR = MatrixExp3(mat1)
    # print(matR)

    # # q10
    # print(so3.skew_symmetric(*[1,2,0.5]))

    # # q11
    # # print(SO3.skew_mat_to_SO3(np.array([
    # #     [0, 0.5, -1],
    # #     [-0.5, 0, 2],
    # #     [1, -2, 0]
    # # ])))

    # print(MatrixExp3(np.array([
    #     [0, 0.5, -1],
    #     [-0.5, 0, 2],
    #     [1, -2, 0]
    # ])))

    # # q12
    # M, th = so3.logarithm(np.array([
    #     [0,0,1],
    #     [-1,0,0],
    #     [0,-1,0]
    # ]))

    # print(M*th)
    
    # print(MatrixLog3(np.array([
    #     [0,0,1],
    #     [-1,0,0],
    #     [0,-1,0]
    # ])))

    # ---- book exercises ----

    print("3.1 i>")
    print(so3.logarithm(np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])))

    print("3.3> P' = RP, but P is non invertable, we find the third independent vector and remember R is from SO3 so we find independent p3' as well then, P'P^-1 = R")

    print("3.6>")
    R_3_6 = SO3.rot_x(np.pi/2) @ SO3.rot_z(np.pi)
    print(R_3_6)
    print(so3.logarithm(R_3_6))

    print("3.7 a>")
    print(so3.logarithm(np.array([
        [0,0,1],
        [0,-1,0],
        [1,0,0]
    ])))
    