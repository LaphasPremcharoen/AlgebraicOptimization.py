from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class FinSetAlgebra(ABC, Generic[T]):
    """
    A finset algebra is a lax symmetric monoidal functor (FinSet,+) → (Set,×).
    We use Python's type system to model the category of sets.
    """
    
    @abstractmethod
    def hom_map(self, phi, X: T) -> T:
        """Implement the action of a finset-algebra on morphisms."""
        raise NotImplementedError("Morphism map not implemented")

    @abstractmethod
    def laxator(self, Xs: List[T]) -> T:
        """Implement the product comparison (aka laxator) of a finset algebra."""
        raise NotImplementedError("Laxator not implemented")

    def oapply(self, phi, Xs: List[T]) -> T:
        """Implements operadic composition for a given finset-algebra implementation."""
        return self.hom_map(phi, self.laxator(Xs))
