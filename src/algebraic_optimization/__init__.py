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

# Import key components from the subpackage to make them available at the top level
# e.g., algebraic_optimization.Open, algebraic_optimization.OptimizerPy
from .compositional_programming import (
    FinSetAlgebra,
    Open,
    OptimizerPy,
    euler_method,
    simulate
)

# Define what gets imported with 'from algebraic_optimization import *'
# and what is generally considered the public API of the package.
__all__ = [
    'FinSetAlgebra',
    'Open',
    'OptimizerPy',
    'euler_method',
    'simulate',
    '__version__',
    '__author__',
]
