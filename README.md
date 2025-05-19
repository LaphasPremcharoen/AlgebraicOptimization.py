# Algebraic Optimization for Python

A Python implementation of the algebraic optimization framework described in the paper 
"A Compositional Framework for First-Order Optimization" (2403.05711). This package provides tools for 
building and solving optimization problems using compositional programming techniques.

## Example Usage

This example shows how to define, compose, and simulate optimization problems:

```python
import numpy as np
from algebraic_optimization.compositional_programming.open_flow_graphs import Open
from algebraic_optimization.compositional_programming.optimizers import (
    OptimizerPy, 
    ContinuousOptPy, 
    euler_method, 
    simulate
)

def quadratic_gradient_dynamics(P_mat, a_vec):
    """Negative gradient dynamics for quadratic cost."""
    P_plus_PT = P_mat + P_mat.T
    return lambda x: -(P_plus_PT @ x + a_vec)

# Problem parameters
P = np.array([
    [2.1154, -0.3038, 0.368, -1.5728, -1.203],
    [-0.3038, 1.5697, 1.0226, 0.159, -0.946],
    [0.368, 1.0226, 1.847, -0.4916, -1.2668],
    [-1.5728, 0.159, -0.4916, 2.2192, 1.5315],
    [-1.203, -0.946, -1.2668, 1.5315, 1.9281]
])
Q = np.array([
    [0.2456, 0.3564, -0.0088],
    [0.3564, 0.5912, -0.0914],
    [-0.0088, -0.0914, 0.8774]
])
R = np.array([
    [2.0546, -1.333, -0.5263, 0.3189],
    [-1.333, 1.0481, -0.0211, 0.2462],
    [-0.5263, -0.0211, 0.951, -0.7813],
    [0.3189, 0.2462, -0.7813, 1.5813]
])

a_vec = np.array([-0.26, 0.22, 0.09, 0.19, -0.96])
b_vec = np.array([-0.72, 0.12, 0.41])
c_vec = np.array([0.55, 0.51, 0.6, -0.61])

# Define optimization problems
f_dyn = quadratic_gradient_dynamics(P, a_vec)
f_opt = OptimizerPy(state_space_dim=P.shape[0], dynamics=f_dyn)
g_opt = OptimizerPy(state_space_dim=Q.shape[0], dynamics=quadratic_gradient_dynamics(Q, b_vec))
h_opt = OptimizerPy(state_space_dim=R.shape[0], dynamics=quadratic_gradient_dynamics(R, c_vec))

# Create Open problems with exposed variables
p1 = Open(domain=f_opt.state_space_dim, problem=f_opt, exposed=[1, 3])  # Expose u and x
p2 = Open(domain=g_opt.state_space_dim, problem=g_opt, exposed=[0, 1, 2])  # Expose u, w, y
p3 = Open(domain=h_opt.state_space_dim, problem=h_opt, exposed=[0, 2, 3])  # Expose u, w, z

# Compose problems
algebra = ContinuousOptPy()

# Step 1: Compose p1 and p2, connecting their 'u' variables
composite = p1.compose(
    p2, 
    mapping={0: 0},  # Connect p1's exposed[0] (u) to p2's exposed[0] (u)
    algebra=algebra,
    keep_mapped_vars_exposed=True
)

# Step 2: Compose with p3, connecting shared 'u' and 'w' variables
final_problem = composite.compose(
    p3,
    mapping={0: 0, 2: 1},  # Connect shared u and w
    algebra=algebra,
    keep_mapped_vars_exposed=False  # Only expose x, y, z in final problem
)

# Simulate the composed problem
x0 = np.random.rand(final_problem.domain)
final_state = simulate(
    euler_method(final_problem, gamma=0.1),  # Discretize
    x0,
    t_steps=200
)

print(f"Final state after 200 steps:")
print(f"x: {final_state[final_problem.exposed[0]]:.4f}")
print(f"y: {final_state[final_problem.exposed[1]]:.4f}")
print(f"z: {final_state[final_problem.exposed[2]]:.4f}")
