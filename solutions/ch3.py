from src.rigid_body import so3, SO3, SE3, se3
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
    
    print("3.16 i>")
    T_16 = SE3.transform(np.array([[0,-1,0],[0,0,-1],[1,0,0]]), np.array([3,0,0]))
    w_mat, v, theta = se3.logarithm6(T_16)
    w = so3.skew_to_vec(w_mat)
    print(f"omega={np.round(w,4)}, v={np.round(v,4)}, theta={round(theta,4)} rad")
    S = np.concatenate([w, v])
    print(f"Screw axis S = {np.round(S, 4)}")
    q, s_hat, h = se3.axis_to_screw(S)
    print(f"q={np.round(q,4)}, s_hat={np.round(s_hat,4)}, h={round(h,4)}")

    print("3.28 >")