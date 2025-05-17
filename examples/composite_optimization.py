import numpy as np
from algebraic_optimization import (
    PrimalObjective,
    Open,
    solve,
    solve_scipy
)

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

a = np.array([-0.26, 0.22, 0.09, 0.19, -0.96])
b = np.array([-0.72, 0.12, 0.41])
c = np.array([0.55, 0.51, 0.6, -0.61])

def quadratic_cost(P, a):
    """Create a quadratic cost function."""
    def cost(x):
        return x.T @ P @ x + a.T @ x
    return cost

# Create subproblem objectives
f = PrimalObjective(5, quadratic_cost(P, a))
g = PrimalObjective(3, quadratic_cost(Q, b))
h = PrimalObjective(4, quadratic_cost(R, c))

# Create open problems
p1 = Open(5, f, [1, 3])  # Expose components 1 and 3
p2 = Open(3, g, [0, 1, 2])  # Expose all components
p3 = Open(4, h, [0, 2, 3])  # Expose components 0, 2, and 3

# Compose the problems according to the diagram:
# f(u,x)
# g(u,w,y)
# h(u,w,z)
# First compose p1 and p2 (f and g) through u
composite1 = p1.compose(p2, {1: 0, 3: 1})  # Connect u components

# Then compose the result with p3 through w
# The exposed indices from composite1 are [0, 2, 4, 5]
# We need to map these to the exposed indices of p3 [0, 2, 3]
# Map the first exposed index of composite1 (index 0) to the first exposed index of p3 (index 0)
# Map the third exposed index of composite1 (index 4) to the second exposed index of p3 (index 2)
composite_problem = composite1.compose(p3, {0: 0, 4: 2})  # Connect w components

# Print the dimensions to verify
print(f"Composite problem domain dimension: {composite_problem.domain}")
print(f"Composite problem exposed indices: {composite_problem.exposed}")

# Create initial guess based on the domain dimension
initial_guess = [100.0] * composite_problem.domain

# Solve using distributed gradient descent
solution = solve(composite_problem, initial_guess, step_size=0.1, n_iterations=100)
print("Solution using distributed gradient descent:", solution)

# Solve using SciPy's optimizer
scipy_solution = solve_scipy(composite_problem, initial_guess)
print("Solution using SciPy's optimizer:", scipy_solution)

# Print the dimensions to verify
print(f"Composite problem domain dimension: {composite_problem.domain}")
print(f"Composite problem exposed indices: {composite_problem.exposed}")

# Create initial guess based on the domain dimension
initial_guess = [100.0] * composite_problem.domain

# Solve using distributed gradient descent
solution = solve(composite_problem, initial_guess, step_size=0.1, n_iterations=100)
print("Solution using distributed gradient descent:", solution)

# Solve using SciPy's optimizer
scipy_solution = solve_scipy(composite_problem, initial_guess)
print("Solution using SciPy's optimizer:", scipy_solution)

# Print the dimensions to verify
print(f"Composite problem domain dimension: {composite_problem.domain}")
print(f"Composite problem exposed indices: {composite_problem.exposed}")

# Create initial guess based on the domain dimension
initial_guess = [100.0] * composite_problem.domain

# Solve using distributed gradient descent
solution = solve(composite_problem, initial_guess, step_size=0.1, n_iterations=100)
print("Solution using distributed gradient descent:", solution)

# Solve using SciPy's optimizer
scipy_solution = solve_scipy(composite_problem, initial_guess)
print("Solution using SciPy's optimizer:", scipy_solution)

# Print the dimensions to verify
print(f"Composite problem domain dimension: {composite_problem.domain}")
print(f"Composite problem exposed indices: {composite_problem.exposed}")

# Create initial guess based on the domain dimension
initial_guess = [100.0] * composite_problem.domain

# Solve using distributed gradient descent
solution = solve(composite_problem, initial_guess, step_size=0.1, n_iterations=100)
print("Solution using distributed gradient descent:", solution)

# Solve using SciPy's optimizer
scipy_solution = solve_scipy(composite_problem, initial_guess)
print("Solution using SciPy's optimizer:", scipy_solution)

# Print the dimensions to verify
print(f"Composite problem domain dimension: {composite_problem.domain}")
print(f"Composite problem exposed indices: {composite_problem.exposed}")

# Create initial guess based on the domain dimension
initial_guess = [100.0] * composite_problem.domain

# Solve using distributed gradient descent
solution = solve(composite_problem, initial_guess, step_size=0.1, n_iterations=100)
print("Solution using distributed gradient descent:", solution)

# Solve using SciPy's optimizer
scipy_solution = solve_scipy(composite_problem, initial_guess)
print("Solution using SciPy's optimizer:", scipy_solution)

# Print the dimensions to verify
print(f"Composite problem domain dimension: {composite_problem.domain}")
print(f"Composite problem exposed indices: {composite_problem.exposed}")

# Create initial guess based on the domain dimension
initial_guess = [100.0] * composite_problem.domain

# Solve using distributed gradient descent
solution = solve(composite_problem, initial_guess, step_size=0.1, n_iterations=100)
print("Solution using distributed gradient descent:", solution)

# Solve using SciPy's optimizer
scipy_solution = solve_scipy(composite_problem, initial_guess)
print("Solution using SciPy's optimizer:", scipy_solution)

# Print the dimensions to verify
print(f"Composite problem domain dimension: {composite_problem.domain}")
print(f"Composite problem exposed indices: {composite_problem.exposed}")

# Initial guess (dimension of composite_problem's domain)
initial_guess = [100.0] * composite_problem.domain

# Solve using distributed gradient descent
solution = solve(composite_problem, initial_guess, step_size=0.1, n_iterations=100)
print("Solution using distributed gradient descent:", solution)

# Solve using SciPy's optimizer
scipy_solution = solve_scipy(composite_problem, initial_guess)
print("Solution using SciPy's optimizer:", scipy_solution)
