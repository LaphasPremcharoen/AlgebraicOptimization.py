import numpy as np
from algebraic_optimization_py.compositional_programming.objectives import PrimalObjective
from algebraic_optimization_py.compositional_programming.open_flow_graphs import Open
from algebraic_optimization_py.compositional_programming.optimizers import solve, solve_scipy

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

def print_solution_info(method_name, solution, problem):
    """Print detailed information about the solution."""
    print(f"\n{'='*80}")
    print(f"{method_name} Solution:")
    print(f"  Solution vector: {np.array_str(solution, precision=4, suppress_small=True)}")
    # For Open objects, we can't directly access the cost function
    # So we'll just print the solution and problem dimensions
    print(f"  Domain dimension: {problem.domain}")
    print(f"  Exposed indices: {problem.exposed}")
    if hasattr(problem, 'objective') and hasattr(problem.objective, 'cost_function'):
        print(f"  Objective value: {problem.objective.cost_function(solution):.6f}")
    if hasattr(problem, 'gradient'):
        try:
            grad_norm = np.linalg.norm(problem.gradient(solution))
            print(f"  Norm of gradient: {grad_norm:.6f}")
        except Exception as e:
            print(f"  Could not compute gradient norm: {e}")

# Print problem information
print("\n" + "="*80)
print("PROBLEM INFORMATION:")
print(f"  Domain dimension: {composite_problem.domain}")
print(f"  Exposed indices: {composite_problem.exposed}")

# Set up initial guess
initial_guess = np.ones(composite_problem.domain)  # Using ones instead of 100s for better numerical stability
print(f"\nInitial guess: {initial_guess}")

# Solve using distributed gradient descent with more iterations
print("\nRunning distributed gradient descent...")
solution_gd = solve(
    composite_problem, 
    initial_guess.copy(),  # Make a copy to avoid modifying the original
    step_size=0.01,        # Smaller step size for better stability
    n_iterations=1000      # More iterations for better convergence
)
print_solution_info("Distributed Gradient Descent", solution_gd, composite_problem)

# Solve using SciPy's optimizer with more detailed output
print("\nRunning SciPy's optimizer...")
solution_scipy = solve_scipy(composite_problem, initial_guess.copy())
print_solution_info("SciPy Optimizer", solution_scipy, composite_problem)

# Compare the solutions
print("\n" + "="*80)
print("COMPARISON:")
solution_diff = np.linalg.norm(solution_gd - solution_scipy)
print(f"  Difference in solutions (L2 norm): {solution_diff:.6f}")

# Try to compare objective values if possible
obj_diff = None
if hasattr(composite_problem, 'objective') and hasattr(composite_problem.objective, 'cost_function'):
    try:
        obj_gd = composite_problem.objective.cost_function(solution_gd)
        obj_scipy = composite_problem.objective.cost_function(solution_scipy)
        obj_diff = abs(obj_gd - obj_scipy)
        print(f"  Difference in objective values: {obj_diff:.6f}")
    except Exception as e:
        print(f"  Could not compare objective values: {e}")

# Check if the solutions are close (within numerical tolerance)
tolerance = 1e-4
if solution_diff < tolerance:
    print("\nThe solutions are numerically equivalent (within tolerance).")
else:
    print("\nThe solutions are different. This could be due to:")
    print("  1. Different optimization algorithms (gradient descent vs BFGS/L-BFGS-B)")
    print("  2. Different convergence criteria")
    print("  3. Multiple local minima in the optimization landscape")
    print("  4. Numerical precision issues")
    
    if obj_diff is not None and obj_diff < tolerance:
        print("\nNote: While the solution vectors differ, the objective values are "
              "very close, suggesting multiple solutions with similar costs.")
    elif obj_diff is not None:
        better_solver = 'Gradient Descent' if obj_gd < obj_scipy else 'SciPy Optimizer'
        print(f"\nThe solution with the better (lower) objective value is: {better_solver}")

print("\nRecommendations:")
print("  1. Try different initial guesses to check for multiple local minima")
print("  2. For gradient descent, try adjusting the step size and number of iterations")
print("  3. For SciPy's optimizer, try different methods (e.g., 'BFGS', 'L-BFGS-B', 'SLSQP')")
print("  4. Check the gradient implementation if the solutions differ significantly")
