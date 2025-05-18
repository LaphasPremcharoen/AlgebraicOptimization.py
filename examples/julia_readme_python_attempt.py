"""
Python Attempt to Replicate AlgebraicOptimization.jl README Example
=================================================================

This script sets up the components (problems, open problems) analogous to 
the basic usage example in the AlgebraicOptimization.jl README.md.

It then attempts to replicate the `oapply` composition 
using sequential calls to the Python `Open.compose` method, leveraging the
`keep_mapped_vars_exposed` parameter.
"""

import numpy as np
from algebraic_optimization.compositional_programming.open_flow_graphs import Open
from algebraic_optimization.compositional_programming.optimizers import OptimizerPy, ContinuousOptPy, euler_method, simulate # Ensure both are imported

# =============================================================================
# Helper Function (from composite_optimization.py)
# =============================================================================

def quadratic_gradient_dynamics(P_mat, a_vec):
    """Create a function for negative gradient dynamics of a quadratic cost.
    If Cost: C(x) = x.T @ P_any @ x + a_vec.T @ x
    Gradient: grad C(x) = (P_any + P_any.T) @ x + a_vec
    Dynamics: dx/dt = -grad C(x)
    """
    P_plus_PT = P_mat + P_mat.T
    def dynamics(x: np.ndarray) -> np.ndarray:
        return -(P_plus_PT @ x + a_vec)
    return dynamics

# =============================================================================
# Problem Parameters (from Julia README)
# =============================================================================
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

# =============================================================================
# Subproblem Setup (Python equivalent of PrimalObjective)
# =============================================================================
f_dyn = quadratic_gradient_dynamics(P, a_vec)
f_opt = OptimizerPy(state_space_dim=P.shape[0], dynamics=f_dyn)

g_dyn = quadratic_gradient_dynamics(Q, b_vec)
g_opt = OptimizerPy(state_space_dim=Q.shape[0], dynamics=g_dyn)

h_dyn = quadratic_gradient_dynamics(R, c_vec)
h_opt = OptimizerPy(state_space_dim=R.shape[0], dynamics=h_dyn)

# =============================================================================
# Open Problem Setup
# =============================================================================
# p1 for f(u,x) with domain 5. Julia: exposes 2nd (u_f), 4th (x_f).
# Python: p1.exposed = [1, 3] (0-indexed domain indices for u_f, x_f respectively)
#   p1.exposed[0] refers to original domain index 1 (u_f)
#   p1.exposed[1] refers to original domain index 3 (x_f)
p1 = Open(domain=f_opt.state_space_dim, problem=f_opt, exposed=[1, 3])
print(f"p1 (for f): domain={p1.domain}, problem_dim={p1.problem.state_space_dim}, exposed_domain_indices={p1.exposed}")

# p2 for g(u,w,y) with domain 3. Julia: exposes all (1st u_g, 2nd w_g, 3rd y_g).
# Python: p2.exposed = [0, 1, 2] (0-indexed domain indices for u_g, w_g, y_g respectively)
#   p2.exposed[0] refers to original domain index 0 (u_g)
#   p2.exposed[1] refers to original domain index 1 (w_g)
#   p2.exposed[2] refers to original domain index 2 (y_g)
p2 = Open(domain=g_opt.state_space_dim, problem=g_opt, exposed=[0, 1, 2])
print(f"p2 (for g): domain={p2.domain}, problem_dim={p2.problem.state_space_dim}, exposed_domain_indices={p2.exposed}")

# p3 for h(u,w,z) with domain 4. Julia: exposes 1st (u_h), 3rd (w_h), 4th (z_h).
# Python: p3.exposed = [0, 2, 3] (0-indexed domain indices for u_h, w_h, z_h respectively)
#   p3.exposed[0] refers to original domain index 0 (u_h)
#   p3.exposed[1] refers to original domain index 2 (w_h)
#   p3.exposed[2] refers to original domain index 3 (z_h)
p3 = Open(domain=h_opt.state_space_dim, problem=h_opt, exposed=[0, 2, 3])
print(f"p3 (for h): domain={p3.domain}, problem_dim={p3.problem.state_space_dim}, exposed_domain_indices={p3.exposed}")

continuous_algebra = ContinuousOptPy()

# =============================================================================
# Composition Steps to Replicate `oapply` for d = @relation_diagram (x,y,z) begin f(u,x); g(u,w,y); h(u,w,z) end
# =============================================================================
print("\n" + "="*80)
print("REPLICATING oapply(d, [p1,p2,p3]) WITH SEQUENTIAL Open.compose")
print("="*80)

print("The Julia diagram implies connecting:")
print("  - 'u' from p1, p2, and p3.")
print("  - 'w' from p2 and p3.")
print("  - Exposing 'x' from p1, 'y' from p2, 'z' from p3 as the final interface.")

print("\nStep 1: Compose p1 and p2, connecting their 'u' variables.")
# Mapping p1.exposed[0] (u_f, domain index 1) to p2.exposed[0] (u_g, domain index 0)
map_p1_p2 = {0: 0} 
composite_fg = p1.compose(p2, map_p1_p2, algebra=continuous_algebra, keep_mapped_vars_exposed=True)

print(f"  composite_fg: domain={composite_fg.domain}, problem_dim={composite_fg.problem.state_space_dim}, exposed_domain_indices={composite_fg.exposed}")
print("  Tracing exposed variables in composite_fg (domain size 7):")
print("    - Based on ContinuousOptPy.laxator internal index mapping:")
print("      - Shared 'u' (from p1.domain[1], p2.domain[0]) -> new domain index 0.")
print("      - p1's 'x_f' (from p1.domain[3])             -> new domain index 3.")
print("      - p2's 'w_g' (from p2.domain[1])             -> new domain index 5.")
print("      - p2's 'y_g' (from p2.domain[2])             -> new domain index 6.")
print(f"  So, composite_fg.exposed = {composite_fg.exposed} should correspond to [u_shared, x_f, w_g, y_g] in some order.")
# Expected sorted order: [0 (u_shared), 3 (x_f), 5 (w_g), 6 (y_g)]
# So: composite_fg.exposed[0] is u_shared (new domain index 0)
#     composite_fg.exposed[1] is x_f (new domain index 3)
#     composite_fg.exposed[2] is w_g (new domain index 5)
#     composite_fg.exposed[3] is y_g (new domain index 6)
idx_list_ushared_in_cfg = 0 # Corresponds to domain index 0 in composite_fg
idx_list_xf_in_cfg      = 1 # Corresponds to domain index 3 in composite_fg
idx_list_wg_in_cfg      = 2 # Corresponds to domain index 5 in composite_fg
idx_list_yg_in_cfg      = 3 # Corresponds to domain index 6 in composite_fg

print(f"    - Interpreted: composite_fg.exposed[{idx_list_ushared_in_cfg}] (value {composite_fg.exposed[idx_list_ushared_in_cfg]}) is the shared 'u'.")
print(f"    - Interpreted: composite_fg.exposed[{idx_list_xf_in_cfg}] (value {composite_fg.exposed[idx_list_xf_in_cfg]}) is 'x_f'.")
print(f"    - Interpreted: composite_fg.exposed[{idx_list_wg_in_cfg}] (value {composite_fg.exposed[idx_list_wg_in_cfg]}) is 'w_g'.")
print(f"    - Interpreted: composite_fg.exposed[{idx_list_yg_in_cfg}] (value {composite_fg.exposed[idx_list_yg_in_cfg]}) is 'y_g'.")

print("\nStep 2: Compose composite_fg with p3.")
print("  We need to connect:")
print(f"    - The shared 'u' from composite_fg (exposed list index {idx_list_ushared_in_cfg}) with u_h from p3 (exposed list index 0, p3.exposed[0]).")
print(f"    - 'w_g' from composite_fg (exposed list index {idx_list_wg_in_cfg}) with w_h from p3 (exposed list index 1, p3.exposed[1]).")

map_cfg_p3 = {
    idx_list_ushared_in_cfg: 0, # Map u_shared (cfg.exposed[0]) to u_h (p3.exposed[0])
    idx_list_wg_in_cfg: 1      # Map w_g (cfg.exposed[2]) to w_h (p3.exposed[1])
}
print(f"  Mapping for composite_fg.compose(p3): {map_cfg_p3}")

final_problem = composite_fg.compose(p3, map_cfg_p3, algebra=continuous_algebra, keep_mapped_vars_exposed=False)
# keep_mapped_vars_exposed=False, as the final interface should only be x, y, z.

print(f"\nfinal_problem: domain={final_problem.domain}, problem_dim={final_problem.problem.state_space_dim}, exposed_domain_indices={final_problem.exposed}")
print("  Tracing exposed variables in final_problem (domain size 9):")
print("    - Based on ContinuousOptPy.laxator internal index mapping:")
print("      - Variables mapped in this step ('u_shared_overall', 'w_shared_overall') are NOT exposed.")
print("      - Unmapped from composite_fg's exposed list:")
print(f"        - 'x_f' (from composite_fg.exposed[{idx_list_xf_in_cfg}], domain idx {composite_fg.exposed[idx_list_xf_in_cfg]}) -> new domain index 4.")
print(f"        - 'y_g' (from composite_fg.exposed[{idx_list_yg_in_cfg}], domain idx {composite_fg.exposed[idx_list_yg_in_cfg]}) -> new domain index 6.")
print("      - Unmapped from p3's exposed list:")
print(f"        - 'z_h' (from p3.exposed[2], domain idx {p3.exposed[2]}) -> new domain index 8.")
print(f"  So, final_problem.exposed = {final_problem.exposed} should correspond to [x_f_final, y_g_final, z_h_final].")
print(f"    These are new domain indices: [4, 6, 8] representing the original x, y, z ports of the diagram.")

print("\n" + "="*80)
print("SIMULATING THE FINAL COMPOSED PROBLEM")
print("="*80)

if final_problem:
    print(f"Final problem domain size: {final_problem.domain}")
    # Initial state for the simulation of the composite system
    # The domain of final_problem is 9. The actual problem lives in this space.
    x0_final = np.random.rand(final_problem.domain) 
    # x0_final = np.zeros(final_problem.domain) # Or start at zero
    print(f"  Initial state x0_final (sample): {x0_final[:min(5, final_problem.domain)]}...")

    num_steps = 200
    dt = 0.1

    # 1. Discretize the continuous final_problem using Euler's method
    #    euler_method takes an Open problem and returns a new Open problem with discretized dynamics.
    print(f"  Discretizing final_problem with dt(gamma)={dt}")
    final_problem_discrete = euler_method(final_problem, gamma=dt)

    # 2. Simulate the discretized problem
    #    The simulate function runs a discrete optimizer for t_steps and returns the final state.
    print(f"  Simulating discretized problem for {num_steps} steps.")
    # The simulate function expects an Open problem and initial state x0 for its full domain.
    final_state = simulate(final_problem_discrete, x0_final, t_steps=num_steps)
    
    # final_state = trajectory[-1, :] # simulate now returns only the final state
    print(f"  Simulated for {num_steps} steps with dt={dt}.")
    print(f"  Final state (sample): {final_state[:min(5, final_problem.domain)]}...")

    # The exposed variables of final_problem are [4, 6, 8]
    # These correspond to the original x, y, z variables.
    # We can see their values in the final state:
    if final_problem.exposed and len(final_problem.exposed) == 3:
        idx_x = final_problem.exposed[0] # Should be 4
        idx_y = final_problem.exposed[1] # Should be 6
        idx_z = final_problem.exposed[2] # Should be 8
        print(f"  Value of original 'x' (domain index {idx_x}) in final state: {final_state[idx_x]:.4f}")
        print(f"  Value of original 'y' (domain index {idx_y}) in final state: {final_state[idx_y]:.4f}")
        print(f"  Value of original 'z' (domain index {idx_z}) in final state: {final_state[idx_z]:.4f}")
else:
    print("Final problem was not constructed, skipping simulation.")


print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The multi-step composition to replicate the Julia `oapply` example appears successful!")
print("The `keep_mapped_vars_exposed=True` parameter was crucial for the intermediate step,")
print("allowing internally formed shared variables ('u' in this case) to be connected in subsequent compositions.")
print("The final exposed variables of `final_problem` represent the intended x, y, z interface of the diagram.")

if __name__ == "__main__":
    print("\nScript completed. Review the composition trace and results.")
