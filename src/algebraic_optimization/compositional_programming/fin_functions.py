import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass

@dataclass
class FinFunctionPy:
    """Represents a function between finite sets: f: domain_size -> codomain_size."""
    domain_size: int
    codomain_size: int
    mapping: np.ndarray # Array of integers, mapping[i] is f(i)

    def __post_init__(self):
        if not isinstance(self.mapping, np.ndarray) or self.mapping.ndim != 1 or not np.issubdtype(self.mapping.dtype, np.integer):
            raise ValueError("Mapping must be a 1D numpy array of integers.")
        if len(self.mapping) != self.domain_size:
            raise ValueError(f"Mapping length {len(self.mapping)} must match domain size {self.domain_size}.")
        
        # Only perform value checks if domain_size > 0, as empty mapping is valid for domain_size = 0
        if self.domain_size > 0:
            if not (np.all(self.mapping >= 0) and np.all(self.mapping < self.codomain_size)):
                # This check is problematic if codomain_size is 0. 
                # If codomain_size is 0, mapping values can't be < 0. This implies domain must be 0 too.
                if self.codomain_size == 0:
                     raise ValueError("Cannot map from non-empty domain to empty codomain.")
                raise ValueError(f"Mapping values must be within [0, {self.codomain_size-1}]. Found min {np.min(self.mapping) if self.mapping.size > 0 else 'N/A'}, max {np.max(self.mapping) if self.mapping.size > 0 else 'N/A'}.")
        elif self.codomain_size == 0 and self.domain_size > 0 : # Explicitly forbid map from non-empty to empty
             raise ValueError("Cannot map from non-empty domain to empty codomain unless domain is also empty.")


def pullback_matrix(f: FinFunctionPy) -> sp.csc_matrix:
    """Pullback f^*: R^codomain_size -> R^domain_size; f^*(y)[i] = y[f(i)]."""
    if f.domain_size == 0: # Handle empty domain case
        # Shape should be (domain_size, codomain_size)
        return sp.csc_matrix((f.domain_size, f.codomain_size), dtype=int)
    rows = np.arange(f.domain_size)
    cols = f.mapping
    data = np.ones(f.domain_size, dtype=int)
    return sp.csc_matrix((data, (rows, cols)), shape=(f.domain_size, f.codomain_size))

def pushforward_matrix(f: FinFunctionPy) -> sp.csc_matrix:
    """Pushforward (f_*): R^domain_size -> R^codomain_size. Dual of pullback."""
    return pullback_matrix(f).transpose().tocsc()
