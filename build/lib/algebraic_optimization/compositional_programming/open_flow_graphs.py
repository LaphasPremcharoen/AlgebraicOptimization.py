from typing import List, Tuple, TypeVar, Generic, Dict, Callable
import numpy as np
from dataclasses import dataclass
from .finset_algebras import FinSetAlgebra

T = TypeVar('T')

@dataclass
class Open(Generic[T]):
    """
    An open problem that specifies which components of a problem's domain
    are open to composition with other problems.
    """
    domain: int  # Dimension of the domain
    problem: T  # The underlying problem
    exposed: List[int]  # Indices of exposed components

    def __post_init__(self):
        if not all(0 <= i < self.domain for i in self.exposed):
            raise ValueError("Exposed indices must be within domain dimension")

    def to_matrix(self) -> np.ndarray:
        """Convert exposed indices to a matrix representation."""
        n = self.domain
        m = len(self.exposed)
        matrix = np.zeros((m, n))
        for i, idx in enumerate(self.exposed):
            matrix[i, idx] = 1
        return matrix

    def compose(self, other: 'Open[T]', mapping: Dict[int, int]) -> 'Open[T]':
        """
        Compose this open problem with another open problem.
        
        Args:
            other: The other open problem to compose with
            mapping: A dictionary mapping exposed indices of this problem
                     to exposed indices of the other problem
        """
        # Validate mapping
        if not all(idx in self.exposed for idx in mapping.keys()):
            raise ValueError("Mapping contains indices not in exposed")
        if not all(idx in other.exposed for idx in mapping.values()):
            raise ValueError("Mapping contains indices not in other's exposed")

        # Create new domain dimension
        new_domain = self.domain + other.domain - len(mapping)
        
        # Create new exposed indices
        new_exposed = list(set(range(new_domain)) - set(mapping.keys()))
        
        # Create new problem
        def new_objective(x):
            # Split input into parts for each problem
            x_self = x[:self.domain]
            x_other = x[self.domain:new_domain]
            
            # Apply mapping to connect problems
            for src, dst in mapping.items():
                x_other[dst] = x_self[src]
            
            # Evaluate both problems
            return self.problem(x_self) + other.problem(x_other)
        
        return Open(new_domain, new_objective, new_exposed)
