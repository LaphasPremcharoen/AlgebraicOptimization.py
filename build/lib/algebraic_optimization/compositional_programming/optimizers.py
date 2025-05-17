from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import minimize
from .objectives import PrimalObjective

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
    
    # Get the objective function
    objective = problem.objective
    
    # Run gradient descent
    for _ in range(n_iterations):
        # Calculate gradient using finite differences
        grad = np.gradient(objective, x)
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
