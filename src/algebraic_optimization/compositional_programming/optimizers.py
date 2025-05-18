from typing import List, Callable, Union, TypeVar, Generic, Tuple
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
import typing

if typing.TYPE_CHECKING:
    from .open_flow_graphs import Open 
from .fin_functions import FinFunctionPy, pullback_matrix, pushforward_matrix

OptPyType = TypeVar('OptPyType', bound='OptimizerPy') 

@dataclass
class OptimizerPy:
    """An optimizer defined by state space dimension and dynamics function."""
    state_space_dim: int
    dynamics: Callable[[np.ndarray], np.ndarray]  # R^N -> R^N

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.state_space_dim,):
             raise ValueError(f"Input vector shape {x.shape} does not match state space dimension {self.state_space_dim}")
        return self.dynamics(x)

    @property
    def dom_dim(self) -> int:
        return self.state_space_dim

P_algebra = TypeVar('P_algebra', bound='OptimizerPy')

class AlgebraPy(Generic[P_algebra]):
    """
    Defines the algebraic structure for composing optimization problems.
    P_algebra is the type of the optimization problem (e.g., OptimizerPy).
    """
    def initial(self, n: int) -> P_algebra:
        """Returns an initial (identity-like) problem on n variables."""
        raise NotImplementedError

    def laxator(self, p1: P_algebra, p2: P_algebra, pullback_legs: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[P_algebra, Tuple[List[int], List[int]]]:
        """
        Composes two problems p1 and p2, identifying variables specified in pullback_legs.
        pullback_legs: ((indices_from_p1_domain,...), (indices_from_p2_domain,...))
                       These are the actual domain indices to be identified.
        Returns a new composed problem and a final_map.
        final_map: ([map_p1_orig_idx_to_new_idx], [map_p2_orig_idx_to_new_idx])
                   These lists map original domain indices of p1 and p2 to their new indices
                   in the composed problem's domain.
        """
        raise NotImplementedError
    
    def hom_map(self, phi: FinFunctionPy, s: P_algebra) -> P_algebra:
        """Applies a homomorphism phi to a problem s."""
        raise NotImplementedError


class ContinuousOptPy(AlgebraPy[OptimizerPy]):
    def initial(self, n: int) -> OptimizerPy:
        return OptimizerPy(state_space_dim=n, dynamics=lambda x: np.zeros_like(x))

    def hom_map(self, phi: FinFunctionPy, s: OptimizerPy) -> OptimizerPy:
        """phi_map: s -> phi_* o s o phi^*"""
        phi_star = pullback_matrix(phi) 
        phi_push = pushforward_matrix(phi)
        
        def new_dynamics(x: np.ndarray) -> np.ndarray:
            if phi.domain_size != s.dom_dim:
                raise ValueError(f"FinFunction domain {phi.domain_size} must match optimizer state space {s.dom_dim}")
            if x.shape != (phi.codomain_size,):
                 raise ValueError(f"Input vector shape {x.shape} for new_dynamics does not match FinFunction codomain size {phi.codomain_size}")

            pulled_back_x = phi_star @ x
            optimizer_output = s.dynamics(pulled_back_x)
            return phi_push @ optimizer_output
            
        return OptimizerPy(phi.codomain_size, new_dynamics)

    def laxator(self, p1: OptimizerPy, p2: OptimizerPy, pullback_legs: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[OptimizerPy, Tuple[List[int], List[int]]]:
        d1 = p1.state_space_dim
        d2 = p2.state_space_dim
        legs1, legs2 = pullback_legs

        if len(legs1) != len(legs2):
            raise ValueError("Pullback legs must have the same length.")
        if not all(0 <= leg < d1 for leg in legs1):
            raise ValueError(f"Pullback legs for p1 {legs1} out of bounds for domain size {d1}.")
        if not all(0 <= leg < d2 for leg in legs2):
            raise ValueError(f"Pullback legs for p2 {legs2} out of bounds for domain size {d2}.")

        final_map_p1 = [-1] * d1
        final_map_p2 = [-1] * d2
        current_new_idx = 0

        for leg_p1, leg_p2 in zip(legs1, legs2):
            if final_map_p1[leg_p1] == -1 and final_map_p2[leg_p2] == -1:
                final_map_p1[leg_p1] = current_new_idx
                final_map_p2[leg_p2] = current_new_idx
                current_new_idx += 1
            elif final_map_p1[leg_p1] != -1 and final_map_p2[leg_p2] == -1: 
                final_map_p2[leg_p2] = final_map_p1[leg_p1]
            elif final_map_p1[leg_p1] == -1 and final_map_p2[leg_p2] != -1: 
                final_map_p1[leg_p1] = final_map_p2[leg_p2]
            elif final_map_p1[leg_p1] != final_map_p2[leg_p2]: 
                raise ValueError(f"Inconsistent mapping in pullback_legs for p1 leg {leg_p1} and p2 leg {leg_p2}.")

        for i in range(d1):
            if final_map_p1[i] == -1:
                final_map_p1[i] = current_new_idx
                current_new_idx += 1
        
        for i in range(d2):
            if final_map_p2[i] == -1:
                final_map_p2[i] = current_new_idx
                current_new_idx += 1
        
        new_total_dim = current_new_idx

        def composed_dynamics(x_composed: np.ndarray) -> np.ndarray:
            if x_composed.shape != (new_total_dim,):
                raise ValueError(f"Input vector shape {x_composed.shape} does not match composed dimension {new_total_dim}")

            x_p1 = np.zeros(d1)
            for old_idx, new_idx in enumerate(final_map_p1):
                x_p1[old_idx] = x_composed[new_idx]
            
            x_p2 = np.zeros(d2)
            for old_idx, new_idx in enumerate(final_map_p2):
                x_p2[old_idx] = x_composed[new_idx]

            dx_p1 = p1.dynamics(x_p1)
            dx_p2 = p2.dynamics(x_p2)

            dx_composed = np.zeros(new_total_dim)
            for i in range(d1):
                dx_composed[final_map_p1[i]] += dx_p1[i]
            for j in range(d2):
                dx_composed[final_map_p2[j]] += dx_p2[j]
            
            return dx_composed

        composed_problem = OptimizerPy(state_space_dim=new_total_dim, dynamics=composed_dynamics)
        return composed_problem, (final_map_p1, final_map_p2)

class DiscreteOptPy(AlgebraPy[OptimizerPy]):
    def initial(self, n: int) -> OptimizerPy:
        return OptimizerPy(state_space_dim=n, dynamics=lambda x: x.copy())

    def hom_map(self, phi: FinFunctionPy, s: OptimizerPy) -> OptimizerPy:
        """phi_map: s -> id + phi_* o (s - id) o phi^*"""
        phi_star = pullback_matrix(phi)
        phi_push = pushforward_matrix(phi)
        
        if phi.domain_size != s.dom_dim:
            raise ValueError(f"FinFunction domain {phi.domain_size} must match optimizer state space {s.dom_dim}")
        if phi.codomain_size != s.dom_dim: 
             pass 

        def new_dynamics(x: np.ndarray) -> np.ndarray:
            if x.shape != (phi.codomain_size,):
                raise ValueError(f"Input x shape {x.shape} to DiscreteOptPy.hom_map.new_dynamics must match FinFunction codomain_size {phi.codomain_size}")
            y = phi_star @ x 
            s_y = s.dynamics(y)
            return x + phi_push @ (s_y - y)
            
        return OptimizerPy(phi.codomain_size, new_dynamics)

    def laxator(self, p1: OptimizerPy, p2: OptimizerPy, pullback_legs: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[OptimizerPy, Tuple[List[int], List[int]]]:
        temp_continuous_algebra = ContinuousOptPy()
        return temp_continuous_algebra.laxator(p1, p2, pullback_legs)

def euler_method(open_optimizer: 'Open[OptimizerPy]', gamma: float) -> 'Open[OptimizerPy]':
    """Converts a continuous optimizer to a discrete one using Euler's method."""
    original_optimizer = open_optimizer.problem 
    
    def discrete_dynamics(x: np.ndarray) -> np.ndarray:
        return x + gamma * original_optimizer.dynamics(x)
    
    discrete_opt = OptimizerPy(original_optimizer.dom_dim, discrete_dynamics)
    
    from .open_flow_graphs import Open 
    return Open(domain=original_optimizer.dom_dim, 
                  problem=discrete_opt, 
                  exposed=open_optimizer.exposed)

def simulate(open_optimizer: 'Open[OptimizerPy]', x0: np.ndarray, t_steps: int) -> np.ndarray:
    """Run a discrete optimizer for a number of time steps."""
    current_x = np.array(x0, dtype=float) 
    
    optimizer = open_optimizer.problem 
    
    if current_x.shape != (optimizer.dom_dim,):
        raise ValueError(f"Initial state x0 shape {current_x.shape} does not match optimizer domain {optimizer.dom_dim}")

    for _ in range(t_steps):
        current_x = optimizer(current_x) 
    return current_x