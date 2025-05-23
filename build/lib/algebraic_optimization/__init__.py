"""AlgebraicOptimization.py - A Python package for compositional optimization.

This package provides tools for building large optimization problems out of simpler
subproblems and automatically compiling them to a distributed solver. It's particularly
useful for problems that can be expressed as compositions of smaller, more manageable
optimization tasks.

Python implementation of Algebraic Optimization based on the paper
"A Compositional Framework for First-Order Optimization" (2403.05711).

This package provides tools for algebraic optimization, including:
- Compositional programming for optimization problems
- Implementation of optimization algorithms from the paper
"""

__version__ = "0.1.0"
__author__ = "Algebraic Optimization Team"

# Import key components to make them easily accessible
from .compositional_programming import *

# Define what gets imported with 'from algebraic_optimization import *'
__all__ = [
    "compositional_programming",
    "__version__",
    "__author__",
]

from .compositional_programming.optimizers import solve, solve_scipy
from .compositional_programming.finset_algebras import FinSetAlgebra
from .compositional_programming.objectives import PrimalObjective, MinObj
from .compositional_programming.open_flow_graphs import Open

__all__ = [
    'FinSetAlgebra',
    'PrimalObjective',
    'MinObj',
    'Open',
    'solve',
    'solve_scipy',
    '__version__',
]
