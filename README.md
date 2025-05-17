# AlgebraicOptimization.py

[![PyPI version](https://img.shields.io/pypi/v/algebraic-optimization-py.svg)](https://pypi.org/project/algebraic-optimization-py/)
[![Python Version](https://img.shields.io/pypi/pyversions/algebraic-optimization-py.svg)](https://pypi.org/project/algebraic-optimization-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This Python package is designed for building large optimization problems out of simpler subproblems and automatically compiling them to a distributed solver.

## Installation

```bash
pip install algebraic-optimization-py
```

## Basic Usage

Here's a simple example of how to use the package:

```python
from algebraic_optimization_py import FinSetAlgebra, PrimalObjective, solve

# Define your optimization problem here
# ...

# Solve the problem
solution = solve(problem)
```

## Features

- **Compositional Programming**: Build complex optimization problems from simpler components
- **Multiple Backends**: Supports various optimization backends
- **Efficient**: Optimized for performance with large-scale problems

## Documentation

For more detailed documentation, please refer to the [official documentation](https://github.com/yourusername/AlgebraicOptimization.py).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Example: Composite Optimization

Here's a more complete example showing how to build and solve a composite optimization problem:

```python
import numpy as np
from algebraic_optimization_py import FinSetAlgebra, PrimalObjective, solve

# Define variables and parameters
n1, n2, n3 = 5, 3, 5
P = np.random.randn(n1, n1)
Q = np.random.randn(n2, n2)
R = np.random.randn(n3, n3)
```python
# Create variables and constraints
x1 = Variable("x1", n1)
x2 = Variable("x2", n2)
x3 = Variable("x3", n3)

# Define optimization problems with constraints
prob1 = Problem(PrimalObjective(x1.T @ P @ x1), [x1[0] == 1.0])
prob2 = Problem(PrimalObjective(x2.T @ Q @ x2), [x2[0] == 1.0])
prob3 = Problem(PrimalObjective(x3.T @ R @ x3), [x3[0] == 1.0])

# Define composition pattern (chain x1 -> x2 -> x3)
A = FinSet(1)
B = FinSet(1)
C = FinSet(1)
AB = FinSet(1)
BC = FinSet(1)

# Define functions for composition
f = FinFunction([0], A, B)
g = FinFunction([0], B, C)
h1 = FinFunction([0], AB, A)
k1 = FinFunction([0], AB, B)
h2 = FinFunction([0], BC, B)
k2 = FinFunction([0], BC, C)

# Compose and solve the problem
prob = compose(
    [prob1, prob2, prob3],
    [[h1, k1], [h2, k2]],
    [f, g]
)

# Solve the composed problem
solution = solve(prob)
print(f"Optimal value: {solution.optimal_value}")
```

## Features

- **Compositional Programming**: Build complex optimization problems from simpler components
- **Multiple Backends**: Supports various optimization backends including SciPy
- **Efficient**: Optimized for performance with large-scale problems
- **Flexible**: Supports both constrained and unconstrained optimization

## Documentation

For more detailed documentation, please refer to the [official documentation](https://github.com/yourusername/AlgebraicOptimization.py).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/yourusername/AlgebraicOptimization.py/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please consider citing the original paper:

[A Compositional Framework for First-Order Optimization](https://arxiv.org/abs/2403.05711)
