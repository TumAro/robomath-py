import math
from typing import Optional, List
from collections import Counter

class nDimensionalRigidBody():
    @staticmethod
    def dim2dof(n: int) -> int:
        '''
        For n-dimensional body -> the dof
        '''
        return math.floor(n*(n+1)/2)

    @staticmethod
    def dof2rot_freedom(n: int, m: Optional[int]) -> int:
        '''
        For n-dimensional body -> possible ways of rotational degrees
        For given m-dof -> The possible ways of rotational degrees
        '''
        if m:
            return math.floor(m-n)
        
        return math.floor(n*(n-1)/2)
    
    @staticmethod
    def dof2trans_freedom(n: int, m: Optional[int]) -> int:
        '''
        For n-dimensional body -> it can translate to n direction in Euclidean space
        '''
        return n
    
    @staticmethod
    def space_topology(n: int) -> str:

        topology = ""
        for power in range(n, 0, -1):
            if power == n:
                topology += f"R{power}"
                continue

            topology += f" x S{power}"

        return topology

    

class Mechanism():
    f_values = {
        'R': 1,
        'P': 1,
        'H': 1,
        'C': 2,
        'U': 2,
        'S': 3
    }

    @staticmethod
    def grubler(N: int, J: int, m: int, joints: List[str]) -> int:
        '''
        N -> # of links including ground
        J -> # number of joints
        m -> dof of n-dimensional rigid body (3 for planner / 6 for spatial)
        joints -> List of types of joints
        '''
        if J != len(joints):
            raise ValueError("number of joints size mismatch")
        
        joints_freedom = Counter(joints)

        sum1 = m * (N - 1 -J)
        sum2 = sum([Mechanism.f_values[item]*count for  item, count in joints_freedom.items()])

        return sum1 + sum2
    
    @staticmethod
    def loop_closure():
        pass

    @staticmethod
    def pfaffian() -> bool:
        pass
    
    @staticmethod
    def planar_serial_fk_workspace() -> bool:
        pass