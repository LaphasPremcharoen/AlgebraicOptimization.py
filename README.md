# Compositional Programming for Optimization Problems

This module provides tools for representing and composing optimization problems in a structured, algebraic manner. It draws inspiration from categorical approaches to system composition, allowing complex optimization problems to be built from simpler components.

The core idea is to represent optimization problems as "open" systems that can be wired together. The behavior of these systems is defined by their dynamics, and their composition is governed by an algebraic structure.

## Key Components

The main components exposed by this module are:

-   `OptimizerPy`: A data class representing a basic optimization problem, defined by its state space dimension and a dynamics function.
-   `AlgebraPy`: An abstract base class defining the algebraic operations (like composition) for `OptimizerPy` instances. Concrete implementations like `ContinuousOptPy` and `DiscreteOptPy` provide specific composition rules.
-   `Open`: A generic class that wraps an `OptimizerPy` instance, exposing certain variables (its "interface") for composition with other `Open` problems.
-   `euler_method`: A utility function to convert an `Open` problem with continuous dynamics into one with discrete dynamics using the forward Euler method.
-   `simulate`: A utility function to run the simulation of an `Open` problem with discrete dynamics for a specified number of steps.

## Classes and Functions

### 1. `OptimizerPy`

Defined in: `optimizers.py`

```python
@dataclass
class OptimizerPy:
    state_space_dim: int
    dynamics: Callable[[np.ndarray], np.ndarray]
```

-   **Purpose**: Represents the fundamental unit of an optimization problem.
-   **Attributes**:
    -   `state_space_dim (int)`: The dimension of the state space (N) for the problem.
    -   `dynamics (Callable[[np.ndarray], np.ndarray])`: A function `f: R^N -> R^N` that describes the evolution or update rule of the system. For continuous systems, this is typically `dx/dt = f(x)`. For discrete systems, this is `x_{k+1} = f(x_k)`.
-   **Methods**:
    -   `__call__(self, x: np.ndarray) -> np.ndarray`: Executes the `dynamics` function on a given state `x`.
    -   `dom_dim (property) -> int`: Returns `state_space_dim`.

### 2. `AlgebraPy` (Abstract Base Class)

Defined in: `optimizers.py`

```python
class AlgebraPy(Generic[P_algebra]):
    def initial(self, n: int) -> P_algebra: ...
    def laxator(self, p1: P_algebra, p2: P_algebra, pullback_legs: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Tuple[P_algebra, Tuple[List[int], List[int]]]: ...
    def hom_map(self, phi: FinFunctionPy, s: P_algebra) -> P_algebra: ...
```

-   **Purpose**: Defines the abstract algebraic interface for composing `OptimizerPy` problems (or any type `P_algebra` that adheres to the `OptimizerPy` structure).
-   **Methods (to be implemented by concrete subclasses)**:
    -   `initial(self, n: int) -> P_algebra`: Returns a base or identity-like problem on `n` variables.
    -   `laxator(self, p1, p2, pullback_legs)`: The core composition operation. It takes two problems (`p1`, `p2`) and `pullback_legs` (specifying which domain indices to identify between `p1` and `p2`). It returns a new composed problem and a `final_map` that tracks how original domain indices map to indices in the new composed problem's domain.
    -   `hom_map(self, phi: FinFunctionPy, s: P_algebra) -> P_algebra`: Applies a homomorphism (a structure-preserving map, represented by `FinFunctionPy`) to a problem `s`.

#### Concrete Implementations of `AlgebraPy`:

-   **`ContinuousOptPy(AlgebraPy[OptimizerPy])`**: Implements the algebraic operations for continuous-time optimization problems. Its `laxator` method combines dynamics by summing contributions to shared variables (effectively averaging their influences if normalized).
-   **`DiscreteOptPy(AlgebraPy[OptimizerPy])`**: Implements the algebraic operations for discrete-time optimization problems. Its `laxator` currently delegates to `ContinuousOptPy` for structural composition, assuming the dynamics functions themselves are discrete.

### 3. `Open`

Defined in: `open_flow_graphs.py`

```python
class Open(Generic[P]): # P is bound by OptimizerPy
    domain: int
    problem: P
    exposed: List[int]

    def __init__(self, domain: int, problem: P, exposed: List[int]): ...
    def compose(self, other: 'Open[P]', mapping: Dict[int, int], algebra: AlgebraPy[P], keep_mapped_vars_exposed: bool = False) -> 'Open[P]': ...
```

-   **Purpose**: Wraps an `OptimizerPy` instance (`problem`) to make it "open" for composition. It defines an interface through its `exposed` variables.
-   **Attributes**:
    -   `domain (int)`: The total number of variables in the problem's domain *as seen by this Open object*. This matches `problem.state_space_dim`.
    -   `problem (P)`: The underlying optimization problem (e.g., an `OptimizerPy` instance).
    -   `exposed (List[int])`: A sorted list of unique 0-indexed integers representing which variables in `self.domain` are exposed for connection with other `Open` problems.
-   **Methods**:
    -   `__init__(self, domain, problem, exposed)`: Constructor.
    -   `compose(self, other, mapping, algebra, keep_mapped_vars_exposed=False)`:
        -   Composes `self` with another `Open` problem (`other`).
        -   `mapping`: A dictionary where keys are indices into `self.exposed` and values are indices into `other.exposed`. This specifies which exposed variables to connect.
        -   `algebra`: An instance of `AlgebraPy` (e.g., `ContinuousOptPy`) that defines *how* the underlying `OptimizerPy` problems are composed via its `laxator` method.
        -   `keep_mapped_vars_exposed (bool)`: If `True`, the newly formed shared variables (resulting from the `mapping`) are also included in the `exposed` list of the composed `Open` problem. If `False` (default), these shared variables become internal.
        -   Returns a new `Open` object representing the composed system.

### 4. `euler_method`

Defined in: `optimizers.py`

```python
def euler_method(open_optimizer: 'Open[OptimizerPy]', gamma: float) -> 'Open[OptimizerPy]': ...
```

-   **Purpose**: Converts an `Open` problem with continuous dynamics into an `Open` problem with discrete dynamics using the forward Euler method.
-   **Arguments**:
    -   `open_optimizer (Open[OptimizerPy])`: The input open problem, assumed to contain an `OptimizerPy` with continuous dynamics.
    -   `gamma (float)`: The time step size for the Euler discretization.
-   **Returns**: A new `Open[OptimizerPy]` object where the `problem` attribute has its `dynamics` function replaced by `x_next = x_old + gamma * continuous_dynamics(x_old)`.

### 5. `simulate`

Defined in: `optimizers.py`

```python
def simulate(open_optimizer: 'Open[OptimizerPy]', x0: np.ndarray, t_steps: int) -> np.ndarray: ...
```

-   **Purpose**: Simulates the behavior of an `Open` problem with discrete dynamics over a number of time steps.
-   **Arguments**:
    -   `open_optimizer (Open[OptimizerPy])`: The input open problem, assumed to contain an `OptimizerPy` with discrete dynamics.
    -   `x0 (np.ndarray)`: The initial state vector.
    -   `t_steps (int)`: The number of discrete time steps to simulate.
-   **Returns**: The final state vector `np.ndarray` after `t_steps` of simulation.

## Example Usage

Here's an example of how to define, compose, discretize, and simulate optimization problems using the components from this module.

```python
import numpy as np
from algebraic_optimization.compositional_programming import (
    OptimizerPy,
    ContinuousOptPy,
    Open,
    euler_method,
    simulate
)

# 1. Define basic optimization problems (e.g., gradient descent on quadratic objectives)
# Problem 1: min 0.5*x0^2 + 0.5*x1^2
# Dynamics: dx/dt = -grad(objective) = [-x0, -x1]
def dynamics1(x): # x is [x0, x1]
    return -x
opt_problem1 = OptimizerPy(state_space_dim=2, dynamics=dynamics1)

# Problem 2: min 0.5*y0^2 + 0.5*y1^2
# Dynamics: dy/dt = [-y0, -y1]
def dynamics2(y): # y is [y0, y1]
    return -y
opt_problem2 = OptimizerPy(state_space_dim=2, dynamics=dynamics2)

print(f"Problem 1: {opt_problem1}")
print(f"Problem 2: {opt_problem2}")

# 2. Create Open problems, exposing some variables
# Open problem P1, exposing its second variable (index 1, which is x1)
# Domain of P1 is [x0, x1]
open_p1 = Open(domain=2, problem=opt_problem1, exposed=[1])

# Open problem P2, exposing its first variable (index 0, which is y0)
# Domain of P2 is [y0, y1]
open_p2 = Open(domain=2, problem=opt_problem2, exposed=[0])

print(f"\nOpen P1: {open_p1}")
print(f"Open P2: {open_p2}")

# 3. Compose the Open problems
# We need an algebra for composition
continuous_algebra = ContinuousOptPy()

# Map exposed variable of P1 (x1, which is open_p1.exposed[0])
# to exposed variable of P2 (y0, which is open_p2.exposed[0])
# The mapping is {p1_exposed_list_idx: p2_exposed_list_idx}
composition_mapping = {0: 0} 

composed_problem_continuous = open_p1.compose(
    open_p2, 
    mapping=composition_mapping, 
    algebra=continuous_algebra,
    keep_mapped_vars_exposed=False # The shared (x1,y0) variable becomes internal
)

print(f"\nComposed Continuous Problem: {composed_problem_continuous}")
# Expected domain of composed_problem_continuous: 3 variables
# (e.g., x0_from_p1, shared_x1_y0, y1_from_p2)
# Exposed variables: x0_from_p1 (original p1.domain[0]) and y1_from_p2 (original p2.domain[1])

# 4. Discretize the composed problem
dt = 0.1 # Time step for Euler method
composed_problem_discrete = euler_method(composed_problem_continuous, gamma=dt)
print(f"Discretized Composed Problem: {composed_problem_discrete}")

# 5. Simulate the discretized problem
# Initial state for the 3 variables of the composed system
# (e.g., [initial_x0, initial_shared_x1_y0, initial_y1])
initial_state_composed = np.array([10.0, 5.0, -8.0])
num_steps = 100

final_state = simulate(composed_problem_discrete, x0=initial_state_composed, t_steps=num_steps)

print(f"\nInitial state for composed system: {initial_state_composed}")
print(f"Simulating for {num_steps} steps with dt={dt}...")
print(f"Final state after simulation: {final_state}")

# The final_state should be close to [0,0,0] as the system minimizes the sum of quadratics.
# The exact mapping of final_state components to original variables (x0, y1, and the shared one)
# depends on the indexing chosen by the laxator in ContinuousOptPy.

## Supporting Modules and Classes

Beyond the core workflow of `OptimizerPy`, `AlgebraPy`, and `Open`, the `compositional_programming` module contains supporting classes and functions that provide foundational capabilities.

### 6. `FinFunctionPy` (from `fin_functions.py`)

Defines functions between finite sets and their representations as matrices.

```python
@dataclass
class FinFunctionPy:
    domain_size: int
    codomain_size: int
    mapping: np.ndarray # Array of integers, mapping[i] is f(i)
```

-   **Purpose**: Represents a function `f: {0, ..., domain_size-1} -> {0, ..., codomain_size-1}`.
-   **Attributes**:
    -   `domain_size (int)`: The size of the domain set.
    -   `codomain_size (int)`: The size of the codomain set.
    -   `mapping (np.ndarray)`: A 1D NumPy array of integers where `mapping[i]` is the image of `i` under the function `f`.
-   **Methods**:
    -   `__post_init__(self)`: Validates the integrity of the mapping (e.g., length, values within codomain bounds).

#### Associated Functions in `fin_functions.py`:

-   **`pullback_matrix(f: FinFunctionPy) -> sp.csc_matrix`**:
    -   Computes the matrix representation of the pullback operation `f^*` associated with `f`.
    -   If `f: X -> Y`, then `f^*` maps functions on `Y` (or vectors in `R^codomain_size`) to functions on `X` (or vectors in `R^domain_size`). Specifically, `(f^*v)[i] = v[f(i)]`.
    -   Returns a sparse CSC matrix of shape `(domain_size, codomain_size)`.

-   **`pushforward_matrix(f: FinFunctionPy) -> sp.csc_matrix`**:
    -   Computes the matrix representation of the pushforward operation `f_*`, which is the transpose of the pullback matrix.
    -   Returns a sparse CSC matrix of shape `(codomain_size, domain_size)`.

These matrices are used in `AlgebraPy.hom_map` implementations to transform the state spaces or dynamics of optimization problems.

### 7. `FinSetAlgebra` (Abstract Base Class from `finset_algebras.py`)

Defines an algebraic structure based on finite sets, acting as a blueprint for composing objects of a generic type `T`.

```python
class FinSetAlgebra(ABC, Generic[T]):
    @abstractmethod
    def hom_map(self, phi, X: T) -> T: ...
    @abstractmethod
    def laxator(self, Xs: List[T]) -> T: ...
    def oapply(self, phi, Xs: List[T]) -> T: ...
```

-   **Purpose**: Abstractly defines how a collection of objects (`Xs`) of type `T` can be combined (`laxator`) and how a mapping (`phi`, often a `FinFunctionPy` or its matrix form) can transform an object (`hom_map`). The `oapply` method provides a generic way to compose these operations, typically representing operadic composition.
-   **Type Parameter**: `T` (Generic TypeVar).
-   **Methods (abstract unless specified)**:
    -   `hom_map(self, phi, X: T) -> T`: Applies a morphism `phi` to an object `X`.
    -   `laxator(self, Xs: List[T]) -> T`: Combines a list of objects `Xs` into a single object of type `T`. This is akin to taking a product or sum in a categorical sense.
    -   `oapply(self, phi, Xs: List[T]) -> T`: A concrete method that implements operadic composition by first applying `laxator` to `Xs` and then `hom_map` with `phi` to the result. This is a common pattern in compositional structures.

### 8. `PrimalObjective` and `MinObj` (from `objectives.py`)

Provide structures for defining and composing minimization problems based on cost functions.

#### `PrimalObjective`

```python
@dataclass
class PrimalObjective:
    decision_space: int
    objective: Callable[[np.ndarray], float]
```

-   **Purpose**: Represents a standard minimization problem `min_x f(x)`.
-   **Attributes**:
    -   `decision_space (int)`: The dimension `n` of the decision variable vector `x` (i.e., `x` is in `R^n`).
    -   `objective (Callable[[np.ndarray], float])`: The cost function `f: R^n -> R` to be minimized.
-   **Methods**:
    -   `__call__(self, x: np.ndarray) -> float`: Evaluates the `objective` function at `x`.

#### `MinObj(FinSetAlgebra[PrimalObjective])`

-   **Purpose**: Implements the `FinSetAlgebra` interface for `PrimalObjective`s. It defines how to compose minimization problems, primarily by variable sharing and summing objectives.
-   **Inherits from**: `FinSetAlgebra[PrimalObjective]`.
-   **Methods**:
    -   `hom_map(self, phi: np.ndarray, p: PrimalObjective) -> PrimalObjective`:
        -   Transforms a `PrimalObjective` `p` using a matrix `phi` (representing `phi^*`).
        -   The new objective becomes `f'(x) = f(phi.T @ x)`.
        -   The decision space is adjusted according to the dimensions of `phi`.
    -   `laxator(self, Xs: List[PrimalObjective]) -> PrimalObjective`:
        -   Combines a list of `PrimalObjective`s `Xs` into a single one.
        -   The new decision space is the sum of the individual decision spaces (concatenation of decision vectors).
        -   The new objective function is the sum of the individual objective functions applied to their respective parts of the concatenated decision vector.
    -   `gradient_flow(self, p: PrimalObjective, x0: np.ndarray) -> np.ndarray`:
        -   A utility method (not part of the `FinSetAlgebra` interface directly) to solve the minimization problem defined by `p` starting from an initial guess `x0`.
        -   Uses `scipy.optimize.minimize` with the 'BFGS' method.

This provides a way to construct more complex cost functions from simpler ones by specifying how their variables are related or combined.

## Future Extensions

This module can be extended to include:

-   More sophisticated composition algebras.
-   Support for different types of optimization problems (e.g., constrained optimization, stochastic optimization).
-   Integration with automatic differentiation libraries for complex dynamics and objectives.
-   Visualization tools for composed systems.

---

This concludes the primary documentation for the `compositional_programming` module. Further details or examples can be added as needed.
