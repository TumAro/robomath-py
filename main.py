from src.c_space import nDimensionalBody, Mechanism


if __name__ == "__main__":
    for i in range(1,5):
        print(f"dimension: {i}, dof: {nDimensionalBody.n_dim_dof(i)}")

    print(Mechanism.grubler(
        N = 10,
        J = 12,
        m = 6,
        joints = ['R']*4 + ['S']*8
    ))