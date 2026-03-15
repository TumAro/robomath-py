from src.c_space import nDimensionalRigidBody, Mechanism


if __name__ == "__main__":
    rigidBody = nDimensionalRigidBody()
    for i in range(1,5):
        print(f"dimension: {i}, dof: {rigidBody.dim2dof(i)}, topology: {rigidBody.space_topology(i)}")

    # exercise 2.3 -> human arm
    '''
    shoulder Spherical -> elbow revolute -> wrist spherical
    '''
    print("2.3: ", Mechanism.grubler(
        N = 4,           # ground(torso) + upper arm + forearm + hand
        J = 3,           # shoulder(S) + elbow(R) + wrist(S)
        m = 6,           # spatial (3D)
        joints = ['S', 'R', 'S']
    ))


    # ex 2.4 -> 2 arm + steering
    print("2.4: ", Mechanism.grubler(
        N = 8+1,           # ground(torso + car) + upper arm + forearm + hand x 2 + palm-wheel-link
        J = 6+1,           # shoulder(S) + elbow(R) + wrist(S) x2 + steering
        m = 6,           # spatial (3D)
        joints = ['S', 'R', 'S']*2 + ['R']
    ))

    # ex 2.5
    print("2.5: ", Mechanism.grubler(
        N = 7,
        J = 7,
        m = 6,
        joints = ['R']*5 + ['S']*2
    ))

    # ex 2.6
    print("2.6 a: ", "R2 x T2 x T6") # the wheel (x,y, theta direction and phi wheel roll) + 6R independent angles
    print("2.6 b: ", "[a,b]2 x T2 x T4") # the wheel can be nearby the fridge, holding the door open, 2 teeths are constrained
    print("2.6 c: ", "[a,b]2 x T2 x T4 x [a,b]2 x T2 x T4")


    # ex 2.9
    ex2_9 = [
        Mechanism.grubler(N= 9, J= 10, m=3, joints=['R']*9+['P']),
        Mechanism.grubler(N= 13, J= 18, m=3, joints=['R']*16 + ['P']*2),
        Mechanism.grubler(N= 8, J= 10, m=3, joints= ['R']*6 + ['P']*2 + ['R', 'S']),
        Mechanism.grubler(N= 6, J= 7, m=3, joints=['R']*6 + ['P']),
        # Mechanism.grubler(N= , J= , m=3, joints=),
        Mechanism.grubler(N= 7, J= 8, m=3, joints=['R']*8),
    ]
    print("2.9: ", ex2_9)

    # ex 2.10
    ex2_10 = [
        Mechanism.grubler(N= 6, J= 7, m=3, joints=['R']*7),
        Mechanism.grubler(N= 6, J= 6, m=3, joints=['R']*2 + ['P']*4),
        Mechanism.grubler(N= 14, J= 18, m=3, joints=['R']*16 + ['P']*2),
        Mechanism.grubler(N= 21, J= 27, m=3, joints=['R']*18+['P']*9)
    ]
    print("2.10: ", ex2_10)

    # ex 2.11 -> ill do later

    # ex 2.12 -> ill do later

    # ex 2.13
    print("2.13: ", "6, xyx, roll pitc yaw")

    # ex 2.14
    print("2.14 a: ", Mechanism.grubler(N = 2+(2*3), J = 9, m = 6, joints=['U','P','U']*3))

    # ex 2.15
    # ex 2.16
    print("2.15 a: ", Mechanism.grubler(N = 4, J = 4, m = 3, joints=['R']*4))
    print("2.15 b: ", "config space is a latitude of the sphere S1")
    print("2.15 c: ", " the coupler draws a circle on the sphere surface ie a S1 line embedded on S2 surface")

    # ex 2.17 -> ill do later