"""
Composite Optimization Example
==============================

This script demonstrates how to use the Algebraic Optimization package to solve
a composite optimization problem by breaking it down into smaller subproblems.

This version implements a linear chain: f(x,u) composed with g(u,v), 
then the result composed with h(v,w).
"""

import numpy as np
from algebraic_optimization.compositional_programming.open_flow_graphs import Open
from algebraic_optimization.compositional_programming.optimizers import OptimizerPy, euler_method, simulate, ContinuousOptPy

# For debugging the import issue with open_flow_graphs
import algebraic_optimization.compositional_programming.open_flow_graphs as ofg_module
print(f"DEBUG: open_flow_graphs.py loaded from: {ofg_module.__file__}")

# Global Problem Parameters are no longer used for this simplified example setup
# Users can define their own P, Q, R, a, b, c if they adapt the setup

# =============================================================================
# Helper Functions
# =============================================================================

def quadratic_gradient_dynamics(P_mat, a_vec):
    """Create a function for negative gradient dynamics of a quadratic cost.
    Cost: C(x) = 0.5 * x.T @ P_mat @ x + a_vec.T @ x (if P_mat is symmetric)
    More generally, if Cost: C(x) = x.T @ P_any @ x + a_vec.T @ x
    Gradient: grad C(x) = (P_any + P_any.T) @ x + a_vec
    Dynamics: dx/dt = -grad C(x)
    """
    P_plus_PT = P_mat + P_mat.T
    def dynamics(x: np.ndarray) -> np.ndarray:
        return -(P_plus_PT @ x + a_vec)
    return dynamics


# =============================================================================
# Problem Setup for f(x,u) - g(u,v) - h(v,w)
# =============================================================================

def setup_optimization_problem():
    """Set up the optimization problem for a linear chain f(x,u)-g(u,v)-h(v,w)."""
    
    # Define simple quadratic costs for 2D problems:
    # Cost_f(x0, x1) = (x0-c0)^2 + (x1-c1)^2. Dynamics params: P=2*I, a=-2*C
    # Let f(x,u): cost (x-1)^2 + (u-2)^2. Vars [x,u]. P_f = 2*np.eye(2), a_f = np.array([-2, -4])
    # Let g(u,v): cost (u-3)^2 + (v-4)^2. Vars [u,v]. P_g = 2*np.eye(2), a_g = np.array([-6, -8])
    # Let h(v,w): cost (v-5)^2 + (w-6)^2. Vars [v,w]. P_h = 2*np.eye(2), a_h = np.array([-10, -12])

    P_mat = 2 * np.eye(2) # P_mat for dynamics is P_orig + P_orig.T. If P_orig=I, then P_mat=2I
                          # If cost is (x-c)^2 = x^2 - 2cx + c^2, P_orig_ii = 1, a_orig_i = -2c_i
                          # So (P_orig + P_orig.T)_ii = 2. a_vec = [-2c_0, -2c_1]

    dyn_f = quadratic_gradient_dynamics(P_mat, np.array([-2*1, -2*2])) # for (x-1)^2, (u-2)^2
    dyn_g = quadratic_gradient_dynamics(P_mat, np.array([-2*3, -2*4])) # for (u-3)^2, (v-4)^2
    dyn_h = quadratic_gradient_dynamics(P_mat, np.array([-2*5, -2*6])) # for (v-5)^2, (w-6)^2

    opt_f = OptimizerPy(state_space_dim=2, dynamics=dyn_f)
    opt_g = OptimizerPy(state_space_dim=2, dynamics=dyn_g)
    opt_h = OptimizerPy(state_space_dim=2, dynamics=dyn_h)

    # For f(var0, var1), problem vars are [var0, var1]
    # p1 for f(x,u): domain vars [x,u]. Exposed list specifies order for mapping.
    #   Let domain be [x, u]. p1.exposed = [u_idx=1, x_idx=0]. So p1.exposed[0] is u.
    p1 = Open(domain=2, problem=opt_f, exposed=[1, 0]) 
    # p2 for g(u,v): domain vars [u,v].
    #   Let domain be [u, v]. p2.exposed = [u_idx=0, v_idx=1]. So p2.exposed[0] is u.
    p2 = Open(domain=2, problem=opt_g, exposed=[0, 1]) 
    # p3 for h(v,w): domain vars [v,w].
    #   Let domain be [v, w]. p3.exposed = [v_idx=0, w_idx=1]. So p3.exposed[0] is v.
    p3 = Open(domain=2, problem=opt_h, exposed=[0, 1]) 
    
    continuous_algebra = ContinuousOptPy()

    # Step 1: Compose p1 and p2, identifying p1's u with p2's u.
    # p1.exposed[0] (u from p1) maps to p2.exposed[0] (u from p2).
    # Mapping: {index_in_p1.exposed : index_in_p2.exposed}
    map_p1_p2 = {0: 0}
    print(f"Composing p1 and p2 with mapping: {map_p1_p2}")
    print(f"  p1.exposed: {p1.exposed}, p2.exposed: {p2.exposed}")
    composite_fg = p1.compose(p2, map_p1_p2, algebra=continuous_algebra)
    # composite_fg.exposed should be [x_from_p1, v_from_p2]. Domain size 3 (x, u_internal, v).
    print(f"  composite_fg created. Domain: {composite_fg.domain}, Exposed: {composite_fg.exposed}")

    # Step 2: Compose composite_fg with p3, identifying v from composite_fg with v from p3.
    # composite_fg.exposed[1] (v_from_p2) maps to p3.exposed[0] (v from p3).
    map_fg_p3 = {1: 0} 
    print(f"Composing composite_fg and p3 with mapping: {map_fg_p3}")
    print(f"  composite_fg.exposed: {composite_fg.exposed}, p3.exposed: {p3.exposed}")
    composite_problem = composite_fg.compose(p3, map_fg_p3, algebra=continuous_algebra)
    # composite_problem.exposed should be [x_from_p1, w_from_p3]. Domain size 4 (x, u_int, v_int, w).
    print(f"  Final composite_problem created. Domain: {composite_problem.domain}, Exposed: {composite_problem.exposed}")
    
    return composite_problem


# =============================================================================
# Solution Analysis
# =============================================================================

def print_solution_info(method_name, solution, open_problem: Open[OptimizerPy]):
    """Print detailed information about the solution."""
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} SOLUTION")
    print(f"{'='*80}")
    # Ensure solution is a numpy array for printing
    solution_array = np.array(solution)
    print(f"  Solution vector (final state): \n{np.array_str(solution_array, precision=4, suppress_small=True)}")
    print(f"  Domain dimension of final problem: {open_problem.domain}") 
    print(f"  Exposed indices of final problem: {open_problem.exposed}")
    # Note: open_problem.problem.state_space_dim should be same as open_problem.domain

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run the optimization example."""
    print("\n" + "="*80)
    print("COMPOSITE OPTIMIZATION EXAMPLE - REFACTORED FOR OPTIMIZERPY (Linear Chain)")
    print("="*80)
    
    print("\nSetting up the optimization problem f(x,u)-g(u,v)-h(v,w)...")
    continuous_open_problem = setup_optimization_problem()
    
    print("\n" + "="*80)
    print("CONTINUOUS COMPOSED PROBLEM INFORMATION")
    print("="*80)
    print(f"  Domain dimension: {continuous_open_problem.domain}")
    print(f"  Exposed indices: {continuous_open_problem.exposed}")
    if hasattr(continuous_open_problem.problem, 'state_space_dim'):
        print(f"  Underlying OptimizerPy state_space_dim: {continuous_open_problem.problem.state_space_dim}")
    
    # Initial guess for the domain of the final composed problem.
    # Expected domain is 4 for x, u_internal, v_internal, w.
    initial_guess = np.ones(continuous_open_problem.domain)
    print(f"\nInitial guess for simulation (domain {continuous_open_problem.domain}): {initial_guess}")
    
    gamma = 0.01
    discrete_open_problem = euler_method(continuous_open_problem, gamma)
    print(f"\nDiscretized problem using Euler method with gamma={gamma}")

    t_steps = 1000 
    print("\n" + "="*80)
    print(f"RUNNING SIMULATION FOR {t_steps} STEPS")
    print("="*80)
    
    solution_simulated = simulate(
        discrete_open_problem, 
        initial_guess.copy(),
        t_steps=t_steps
    )
    print_solution_info("Simulated Gradient Descent", solution_simulated, discrete_open_problem)


if __name__ == "__main__":
    main()
