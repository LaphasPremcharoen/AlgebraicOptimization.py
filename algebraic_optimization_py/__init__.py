"""AlgebraicOptimization.py - A Python package for compositional optimization.

This package provides tools for building large optimization problems out of simpler
subproblems and automatically compiling them to a distributed solver. It's particularly
useful for problems that can be expressed as compositions of smaller, more manageable
optimization tasks.

Example:
    >>> from algebraic_optimization_py import PrimalObjective, solve
    >>> import numpy as np
    >>>
    >>> # Define a simple quadratic objective
    >>> P = np.array([[2, 1], [1, 2]])
    >>> f = PrimalObjective(2, lambda x: x.T @ P @ x)
    >>>
    >>> # Solve the optimization problem
    >>> solution = solve(f)
    >>> print(f"Optimal value: {solution.optimal_value}")
"""

__version__ = "0.1.0"

from algebraic_optimization_py.compositional_programming.finset_algebras import FinSetAlgebra
from algebraic_optimization_py.compositional_programming.objectives import PrimalObjective, MinObj
from algebraic_optimization_py.compositional_programming.open_flow_graphs import Open
from algebraic_optimization_py.compositional_programming.optimizers import solve, solve_scipy

__all__ = [
    'FinSetAlgebra',
    'PrimalObjective',
    'MinObj',
    'Open',
    'solve',
    'solve_scipy',
    '__version__',
]
