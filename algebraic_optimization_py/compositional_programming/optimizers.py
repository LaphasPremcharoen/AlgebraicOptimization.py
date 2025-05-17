from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import minimize
from algebraic_optimization_py.compositional_programming.objectives import PrimalObjective

def solve(problem: PrimalObjective, x0: List[float], step_size: float, n_iterations: int) -> np.ndarray:
    """
    Solve the optimization problem using distributed gradient descent.
    
    Args:
        problem: The optimization problem to solve
        x0: Initial guess for the solution
        step_size: Learning rate for gradient descent
        n_iterations: Number of iterations to run
    
    Returns:
        The optimized solution
    """
    # Convert initial guess to numpy array
    x = np.array(x0)
    
    # Get the objective function: use attribute if available, else the problem is callable
    objective = problem.objective if hasattr(problem, 'objective') else problem
    
    eps = 1e-6
    for _ in range(n_iterations):
        # Calculate gradient using forward finite differences
        grad = np.zeros_like(x)
        fx = objective(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            grad[i] = (objective(x_eps) - fx) / eps
        # Update solution
        x -= step_size * grad
    
    return x

def solve_scipy(problem: PrimalObjective, x0: List[float]) -> np.ndarray:
    """
    Solve the optimization problem using SciPy's minimize function.
    
    Args:
        problem: The optimization problem to solve
        x0: Initial guess for the solution
    
    Returns:
        The optimized solution
    """
    result = minimize(problem, x0, method='BFGS')
    return result.x
