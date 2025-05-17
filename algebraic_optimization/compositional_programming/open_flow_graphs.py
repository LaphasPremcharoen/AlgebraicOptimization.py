from typing import List, Tuple, TypeVar, Generic, Dict, Callable
import numpy as np
from dataclasses import dataclass
from algebraic_optimization.compositional_programming.finset_algebras import FinSetAlgebra

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

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the composed problem by calling the underlying objective function.
        """
        return self.problem(x)

    def compose(self, other: 'Open[T]', mapping: Dict[int, int]) -> 'Open[T]':
        """
        Compose this open problem with another open problem.
        
        Args:
            other: The other open problem to compose with
            mapping: A dictionary mapping variable indices of this problem
                     to variable indices of the other problem
        """
        # Determine variable groups: self-only, shared, other-only
        self_only = [i for i in range(self.domain) if i not in mapping]
        shared = list(mapping.keys())
        other_only = [j for j in range(other.domain) if j not in mapping.values()]

        # Build new domain ordering: self-only, shared, other-only
        new_domain = len(self_only) + len(shared) + len(other_only)

        def new_objective(x):
            # Assemble full vectors for each problem
            full_self = np.zeros(self.domain)
            full_other = np.zeros(other.domain)
            # Assign self-only
            for idx, var in enumerate(self_only):
                full_self[var] = x[idx]
            # Assign shared
            for idx, var in enumerate(shared):
                val = x[len(self_only) + idx]
                full_self[var] = val
                full_other[mapping[var]] = val
            # Assign other-only
            base = len(self_only) + len(shared)
            for idx, var in enumerate(other_only):
                full_other[var] = x[base + idx]
            return self.problem(full_self) + other.problem(full_other)

        # Compute new exposed indices based on original exposed variables
        new_exposed = []
        # Expose self's variables that were originally exposed and not mapped
        for var in self.exposed:
            if var not in mapping:
                new_exposed.append(self_only.index(var))
        # Expose other's variables that were originally exposed and not mapped
        base = len(self_only) + len(shared)
        for var in other.exposed:
            if var not in mapping.values():
                new_exposed.append(base + other_only.index(var))

        return Open(new_domain, new_objective, new_exposed)
