"""
AlgebraicOptimization.py - A Python package for building large optimization problems
out of simpler subproblems and automatically compiling them to a distributed solver.
"""

from algebraic_optimization_py.compositional_programming.finset_algebras import FinSetAlgebra
from algebraic_optimization_py.compositional_programming.objectives import PrimalObjective, MinObj
from algebraic_optimization_py.compositional_programming.open_flow_graphs import Open
from algebraic_optimization_py.compositional_programming.optimizers import solve, solve_scipy

__version__ = "0.1.0"
__all__ = [
    'FinSetAlgebra',
    'PrimalObjective',
    'MinObj',
    'Open',
    'solve',
    'solve_scipy'
]
