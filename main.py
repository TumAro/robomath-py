from src.c_space import nDimensionalRigidBody, Mechanism


if __name__ == "__main__":
    rigidBody = nDimensionalRigidBody()
    for i in range(1,5):
        print(f"dimension: {i}, dof: {rigidBody.dim2dof(i)}, topology: {rigidBody.space_topology(i)}")

    # print(Mechanism.grubler(
    #     N = 10,
    #     J = 12,
    #     m = 6,
    #     joints = ['R']*4 + ['S']*8
    # ))