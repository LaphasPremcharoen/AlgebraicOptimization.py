from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from algebraic_optimization.compositional_programming.finset_algebras import FinSetAlgebra

@dataclass
class PrimalObjective:
    """
    An objective defining a minimization problem.
    Consists of a decision space and a cost function.
    """
    decision_space: int  # Dimension of the decision space
    objective: Callable[[np.ndarray], float]  # Cost function R^n -> R

    def __call__(self, x: np.ndarray) -> float:
        return self.objective(x)

class MinObj(FinSetAlgebra[PrimalObjective]):
    """
    Finset-algebra implementing composition of minimization problems
    by variable sharing.
    """

    def hom_map(self, phi: np.ndarray, p: PrimalObjective) -> PrimalObjective:
        """
        The morphism map is defined by ϕ ↦ (f ↦ f∘ϕ^*).
        phi should be a matrix representing the morphism.
        """
        def new_objective(x):
            return p(phi.T @ x)
        return PrimalObjective(phi.shape[1], new_objective)

    def laxator(self, Xs: List[PrimalObjective]) -> PrimalObjective:
        """
        Takes the "disjoint union" of a collection of primal objectives.
        """
        # Calculate total dimension
        total_dim = sum(X.decision_space for X in Xs)
        
        # Create combined objective function
        def combined_objective(x):
            start = 0
            total = 0
            for X in Xs:
                end = start + X.decision_space
                total += X(x[start:end])
                start = end
            return total
        
        return PrimalObjective(total_dim, combined_objective)

    def gradient_flow(self, p: PrimalObjective, x0: np.ndarray) -> np.ndarray:
        """
        Solve the optimization problem using gradient descent.
        """
        result = minimize(p, x0, method='BFGS')
        return result.x
