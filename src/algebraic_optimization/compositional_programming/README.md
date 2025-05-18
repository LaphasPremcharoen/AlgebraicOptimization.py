# Algebraic Optimization

[![Python Version](https://img.shields.io/pypi/pyversions/algebraic-optimization.svg)](https://pypi.org/project/algebraic-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python implementation of the algebraic optimization framework described in the paper 
"A Compositional Framework for First-Order Optimization" (2403.05711). This package provides tools for 
building and solving optimization problems using compositional programming techniques.

## Features

- **Compositional Programming**: Build complex optimization problems by composing simpler subproblems
- **Efficient Implementations**: Optimized for performance with large-scale problems
- **Flexible Backends**: Supports multiple optimization backends

## Installation

```bash
pip install algebraic-optimization
```

## Quick Start

### Basic Usage

```python
import numpy as np
from algebraic_optimization import PrimalObjective, solve

# Define a simple quadratic objective
P = np.array([[2, 1], [1, 2]])
f = PrimalObjective(2, lambda x: x.T @ P @ x, grad=lambda x: 2 * P @ x)

# Solve the optimization problem
solution = solve(f)
print(f"Optimal value: {solution.optimal_value}")
print(f"Optimal point: {solution.optimal_point}")
```

### Compositional Programming

```python
from algebraic_optimization import FinSetAlgebra, Open, MinObj

# Define a simple flow network
with FinSetAlgebra() as fsa:
    # Define nodes and edges
    v1, v2, v3, v4 = fsa.add_nodes(4)
    e1 = fsa.add_edge(v1, v2)
    e2 = fsa.add_edge(v2, v3)
    e3 = fsa.add_edge(v3, v4)
    
    # Define optimization problem
    problem = MinObj(e1 + e2 + e3)
    
    # Add constraints
    problem.constrain(Open(v1), 1)  # Source
    problem.constrain(Open(v4), -1)  # Sink
    
    # Solve
    solution = solve(problem)
    print(f"Minimum cost flow: {solution.optimal_value}")
```

## Documentation

For detailed documentation, including API reference and examples, please see the [documentation](https://laphaspremcharoen.github.io/AlgebraicOptimization.py/).

## Examples

Check out the `examples/` directory for more comprehensive examples, including:
- Network flow optimization
- Distributed consensus problems
- Model predictive control

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite the original paper:

```
@article{algebraic_optimization_2024,
  title={A Compositional Framework for First-Order Optimization},
  author={Author(s)},
  journal={arXiv:2403.05711},
  year={2024},
  url={https://arxiv.org/abs/2403.05711}
}
