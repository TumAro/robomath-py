import math
from typing import Optional, List, Tuple
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
    def pfaffian():
        pass


class PlanarNBarClosedChain:

    def __init__(self, lengths: List[float]) -> None:
        '''
        INPUT:
        lengths -> the length of links
        '''
        self._link_lengths = lengths
        self._n = len(lengths)
        self._dof = Mechanism.grubler(
            N = self._n,
            J = self._n,
            m = 3,
            joints = ['R']*self._n
        )

        if self._n < 4:
            raise ValueError("Minimum closed chain needs 4 links, given: ", self._n)
        
    def _cumulative_angles(self, theta: List[float]):
        phi = []
        for i in range(self._n):
            phi.append(sum(theta[:i]))

        return phi
    
    def constraint_residual(self, theta: List[float]) -> Tuple:
        '''
        INPUT: theta array of length n
        OUTPUT: tuple of length 3
        '''

        if len(theta) != self._n:
            raise ValueError("Number of angles mismatch with number of links!")
        
        phi = self._cumulative_angles(theta)
        g1 = sum([self._link_lengths[i]*math.cos(phi[i]) for i in range(self._n)])
        g2 = sum([self._link_lengths[i]*math.sin(phi[i]) for i in range(self._n)])
        g3 = sum(theta) - 2*math.pi

        return (g1, g2, g3)

    def constarint_jacobian(self, theta: List[float]):
        '''
        INPUT: theta array of length n
        OUTPUT: 3 x n matrix  (this IS equation 2.7 from the chapter)
        '''
        n = len(theta)
        if n != self._n:
            raise ValueError("Number of angles mismatch with number of links!")

        phi = self._cumulative_angles(theta)
        s = [-self._link_lengths[i]*math.sin(phi[i]) for i in range(n)]
        c = [self._link_lengths[i]*math.cos(phi[i]) for i in range(n)]

        J = [[0.0]*n for _ in range(3)]
        J[0][n-1] = s[n-1]
        J[1][n-1] = c[n-1]

        for j in range(n-2, 0, -1):
            J[0][j] = s[j]+J[0][j+1]
            J[1][j] = c[j]+J[1][j+1]

        J[2][:] = [1.0]*n

        return J
    
    def is_valid_config(self, theta, tolerance=1e-8):
        if len(theta) != self._n:
            raise ValueError("Number of angles mismatch with number of links!")
        
        g1, g2, g3 = self.constraint_residual(theta)
        norm = math.sqrt(g1**2 + g2**2 + g3**2)
        return norm < tolerance

    def joint_positions(self, theta) -> List:
        if len(theta) != self._n:
            raise ValueError("Number of angles mismatch with number of links!")
        
        phi = self._cumulative_angles(theta)
        pos = [(0.0, 0.0)]  # origin

        for i in range(self._n):
            prev_x, prev_y = pos[i]
            length = self._link_lengths[i]

            new_x = prev_x + length*math.cos(phi[i])
            new_y = prev_y + length*math.sin(phi[i])
            pos.append((new_x, new_y))

        return pos

    def plot_linkage(self, theta, ax=None):
        import matplotlib.pyplot as plt
        show = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            show = True

        positions = self.joint_positions(theta)

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        ax.plot(xs, ys, '-o', color='steelblue', linewidth=2, markersize=6, zorder=2)

        ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]], '--', color='gray', linewidth=1.5, zorder=1)

        ax.plot(xs[0], ys[0], 's', color='black', markersize=10, zorder=3)
        ax.plot(xs[-1], ys[-1], 's', color='black', markersize=10, zorder=3)

        for i, (x, y) in enumerate(positions):
            ax.annotate(f'$\\theta_{{{i}}}$', (x, y),
                        textcoords="offset points",
                        xytext=(8, 8),
                        fontsize=10)

        residual = self.constraint_residual(theta)
        norm_r = math.sqrt(residual[0]**2 + residual[1]**2 + residual[2]**2)
        valid_str = "CLOSED" if norm_r < 1e-6 else "OPEN"
        ax.set_title(f'{valid_str} | Residual: {norm_r:.2e}')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    def plot_multi_configs(self, list_of_thetas):
        import matplotlib.pyplot as plt

        n_configs = len(list_of_thetas)
        cols = min(n_configs, 4)
        rows = math.ceil(n_configs / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

        # Flatten axes to 1D list regardless of grid shape
        if n_configs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, theta in enumerate(list_of_thetas):
            self.plot_linkage(theta, ax=axes[i])
            axes[i].set_title(f'Config {i + 1}\n' + axes[i].get_title())

        # Hide unused subplots
        for j in range(n_configs, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        plt.show()