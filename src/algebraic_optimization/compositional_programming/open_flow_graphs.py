from typing import TypeVar, Generic, List, Dict, Tuple

# Assuming OptimizerPy and AlgebraPy are in optimizers.py and correctly imported
# If they are in the same directory, this is fine:
from .optimizers import OptimizerPy, AlgebraPy # Ensure ContinuousOptPy is also available if used as a type hint directly

P = TypeVar('P', bound=OptimizerPy)

class Open(Generic[P]):
    """Represents an open optimization problem.

    An open optimization problem is an optimization problem where some of its
    variables are "exposed" and can be composed with other open problems.

    Attributes:
        domain: The total number of variables in the problem's domain *as seen by this Open object*.
                For a base problem, this matches problem.state_space_dim.
                For a composed problem, this is the dimension of the composed system.
        problem: The underlying optimization problem (implementing OptimizerPy).
                 Its state_space_dim might be larger than self.domain if it's a composed problem
                 where self.domain represents an interface view.
        exposed: A list of indices (0-indexed) into the problem's domain (`self.domain`)
                 that are exposed for composition.
    """
    domain: int
    problem: P 
    exposed: List[int] 

    def __init__(self, domain: int, problem: P, exposed: List[int]):
        if not isinstance(domain, int) or domain <= 0:
            raise ValueError("Domain must be a positive integer.")
        
        # The problem's state_space_dim is the actual dimension of its state vector.
        # The Open object's 'domain' is the dimension of the interface it presents.
        # For a base (non-composed) Open problem, self.domain == problem.state_space_dim.
        # For a composed Open problem, self.domain == self.problem.state_space_dim (the new total dimension).
        if not hasattr(problem, 'state_space_dim') or not hasattr(problem, 'dynamics'):
            raise TypeError("Problem must conform to OptimizerPy structure (state_space_dim, dynamics).")
        
        # This check is important: the 'domain' of the Open object must match the actual state space of the problem it holds.
        if problem.state_space_dim != domain:
             raise ValueError(f"Open object's domain ({domain}) must match problem's state_space_dim ({problem.state_space_dim}).")

        if not all(isinstance(i, int) and 0 <= i < domain for i in exposed):
            raise ValueError(f"All exposed indices must be integers within [0, {domain-1}]. Got: {exposed}, domain: {domain}")
        if len(set(exposed)) != len(exposed):
            raise ValueError("Exposed indices must be unique.")

        self.domain = domain
        self.problem = problem
        self.exposed = sorted(list(set(exposed))) # Ensure sorted and unique

    def __repr__(self) -> str:
        return f"Open(domain={self.domain}, problem_type={type(self.problem).__name__}, problem_dim={self.problem.state_space_dim}, exposed={self.exposed})"

    def compose(self, 
                other: 'Open[P]', 
                mapping: Dict[int, int], 
                algebra: AlgebraPy[P],
                keep_mapped_vars_exposed: bool = False
                ) -> 'Open[P]':
        """Composes this open problem with another open problem using AlgebraPy.

        Args:
            other: The other Open problem to compose with.
            mapping: A dictionary mapping exposed indices of `self` to exposed
                     indices of `other`. Keys are indices into `self.exposed` list,
                     values are indices into `other.exposed` list.
            algebra: The AlgebraPy instance defining how to compose the underlying problems.
            keep_mapped_vars_exposed: If True, variables identified by the mapping
                                      will also be included in the exposed list of the
                                      resulting composed problem. Defaults to False.

        Returns:
            A new Open problem representing the composition.
        """
        if not isinstance(algebra, AlgebraPy):
            raise TypeError(f"algebra must be an instance of AlgebraPy, got {type(algebra)}")
        if not isinstance(other, Open):
            raise TypeError(f"other must be an instance of Open, got {type(other)}")

        # Validation of mapping indices (from list indices to domain indices)
        # and conversion to actual domain indices for pullback_legs.
        mapping_keys_self_domain: List[int] = []
        mapping_values_other_domain: List[int] = []

        for self_exposed_list_idx, other_exposed_list_idx in mapping.items():
            if not (isinstance(self_exposed_list_idx, int) and 0 <= self_exposed_list_idx < len(self.exposed)):
                raise ValueError(f"Invalid key {self_exposed_list_idx} in mapping for self.exposed (len {len(self.exposed)}, exposed: {self.exposed})")
            if not (isinstance(other_exposed_list_idx, int) and 0 <= other_exposed_list_idx < len(other.exposed)):
                raise ValueError(f"Invalid value {other_exposed_list_idx} in mapping for other.exposed (len {len(other.exposed)}, exposed: {other.exposed})")
            
            mapping_keys_self_domain.append(self.exposed[self_exposed_list_idx])
            mapping_values_other_domain.append(other.exposed[other_exposed_list_idx])

        # pullback_legs are the actual domain indices to be identified.
        pullback_legs: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
            tuple(mapping_keys_self_domain), 
            tuple(mapping_values_other_domain)
        )
        
        # Use the algebra's laxator to get the new problem and the mapping
        # of original domain indices to new domain indices.
        # new_composed_problem_data.state_space_dim is the new total domain size.
        # final_map is [map_for_self_vars, map_for_other_vars]
        # where map_for_self_vars[old_self_idx] = new_idx in the composed domain.
        new_composed_problem_data, final_map = algebra.laxator(self.problem, other.problem, pullback_legs)
        
        new_total_domain_size = new_composed_problem_data.state_space_dim

        # Calculate exposed indices for the new composed problem.
        # These are indices in the new_total_domain_size.

        # 1. Indices from self.exposed that were NOT in mapping_keys_self_domain (i.e., not mapped)
        #    Their new indices are found via final_map[0].
        new_exposed_from_self_unmapped = [
            final_map[0][self_domain_idx]
            for self_domain_idx in self.exposed
            if self_domain_idx not in mapping_keys_self_domain
        ]

        # 2. Indices from other.exposed that were NOT in mapping_values_other_domain (i.e., not mapped)
        #    Their new indices are found via final_map[1].
        new_exposed_from_other_unmapped = [
            final_map[1][other_domain_idx]
            for other_domain_idx in other.exposed
            if other_domain_idx not in mapping_values_other_domain
        ]

        all_potential_new_exposed = new_exposed_from_self_unmapped + new_exposed_from_other_unmapped

        if keep_mapped_vars_exposed:
            # 3. Add the new domain indices of the *mapped* variables.
            # These are the variables that were part of pullback_legs.
            # final_map[0] maps original self domain indices to new combined domain indices.
            # mapping_keys_self_domain contains original self domain indices from `self` that were mapped.
            # Since pullback_legs[0][i] is identified with pullback_legs[1][i],
            # final_map[0][pullback_legs[0][i]] == final_map[1][pullback_legs[1][i]].
            # So, using final_map[0] with mapping_keys_self_domain is sufficient.
            mapped_vars_new_indices = [
                final_map[0][self_domain_idx]
                for self_domain_idx in mapping_keys_self_domain
            ]
            all_potential_new_exposed.extend(mapped_vars_new_indices)
            
        final_new_exposed_indices = sorted(list(set(all_potential_new_exposed)))
        
        return Open(domain=new_total_domain_size, problem=new_composed_problem_data, exposed=final_new_exposed_indices)


# Example Usage (Illustrative - requires ContinuousOptPy and actual OptimizerPy instances)
if __name__ == '__main__':
    # This is a placeholder for actual usage, as OptimizerPy and ContinuousOptPy need concrete definitions.
    # from .optimizers import ContinuousOptPy # Assuming ContinuousOptPy is in optimizers.py

    # Dummy Optimizer for illustration
    class MyOptimizer(OptimizerPy):
        def __init__(self, dim, name="opt"):
            self.state_space_dim = dim
            self._name = name
        def dynamics(self, x):
            return -x # Simple dynamics
        def __repr__(self):
            return f"MyOptimizer(dim={self.state_space_dim}, name='{self._name}')"

    # Dummy Algebra for illustration
    class MyAlgebra(AlgebraPy[MyOptimizer]):
        def initial(self, n: int) -> MyOptimizer:
            return MyOptimizer(n, name=f"init({n})")

        def laxator(self, 
                    p1: MyOptimizer, 
                    p2: MyOptimizer, 
                    pullback_legs: Tuple[Tuple[int, ...], Tuple[int, ...]]
                    ) -> Tuple[MyOptimizer, Tuple[List[int], List[int]]]:
            # This is a highly simplified laxator for illustration.
            # A real one would merge state spaces and dynamics.
            
            # p1_legs = pullback_legs[0] # original domain indices from p1 to be identified
            # p2_legs = pullback_legs[1] # original domain indices from p2 to be identified
            
            # For simplicity, assume simple concatenation and map identified legs to the first set.
            new_dim = p1.state_space_dim + p2.state_space_dim - len(pullback_legs[0])
            composed_opt = MyOptimizer(new_dim, name=f"composed({p1._name},{p2._name})")
            
            # Create final_map (simplified)
            # map_p1: new_idx = old_idx for unmapped, specific mapping for mapped
            # map_p2: new_idx = p1_dim_unmapped + old_idx_unmapped_p2 for unmapped, specific mapping for mapped
            
            # A true final_map would be complex. Here's a conceptual sketch:
            final_map_p1 = list(range(p1.state_space_dim))
            final_map_p2 = [i + p1.state_space_dim for i in range(p2.state_space_dim)]
            
            # Very basic identification: mapped p2 vars point to p1's mapped var's new index
            current_offset = 0
            temp_map_p1 = [-1] * p1.state_space_dim
            next_new_idx = 0

            # Map p1 vars
            for i in range(p1.state_space_dim):
                if i in pullback_legs[0]:
                    # Find its first occurrence in pullback_legs[0] to get consistent new index
                    leg_idx = pullback_legs[0].index(i)
                    # If this is the first time we see this leg group, assign new_idx
                    # This simple logic assumes mapping_keys_self_domain handles the 'master' index
                    # A proper implementation would use a union-find or similar for shared indices.
                    # For now, just imagine a mapping is formed.
                    pass # Needs proper index calculation
                else:
                    temp_map_p1[i] = next_new_idx
                    next_new_idx +=1
            # This dummy final_map is not correct, real one is complex. 
            # For now, assume laxator returns a valid one.
            # For testing, let's assume a simple non-overlapping merge for unmapped
            # and that mapped variables from p1 take precedence.
            
            # This is a placeholder for a more realistic final_map calculation:
            # Assume state_space_dim already reflects merged variables.
            # map1 maps p1's original indices to the new composed space.
            # map2 maps p2's original indices to the new composed space.
            idx_map_p1 = list(range(p1.state_space_dim)) # Placeholder
            idx_map_p2 = [i + p1.state_space_dim for i in range(p2.state_space_dim)] # Placeholder
            
            # Simulate merging: if p1.exposed[0] (idx s1) maps to p2.exposed[0] (idx s2)
            # then in new space, idx_map_p1[s1] == idx_map_p2[s2]
            # and unmapped variables get new unique indices.
            # For a concrete example, if p1 dim 5, p2 dim 3, 1 var mapped:
            # new_dim = 5+3-1 = 7
            # if p1.exposed[0] (domain idx 1) maps to p2.exposed[0] (domain idx 0)
            # p1: [0, M, 2, 3, 4] -> new [0, 1, 2, 3, 4] (M is now 1)
            # p2: [M, 5, 6]       -> new [1, 5, 6]       (M is now 1)
            # final_map_p1: e.g. [0->0, 1->1, 2->2, 3->3, 4->4]
            # final_map_p2: e.g. [0->1, 1->5, 2->6]
            # Actual final_map is the job of the algebra implementation.
            # We will assume the algebra produces these correctly.

            # For testing, let's use the ContinuousOptPy's logic structure
            from .optimizers import ContinuousOptPy # Local import for test
            cop = ContinuousOptPy()
            return cop.laxator(p1, p2, pullback_legs) # Delegate to a real one for test structure

    # Setup for example
    alg = MyAlgebra()
    opt1 = MyOptimizer(dim=3, name="opt1") # x0, x1, x2
    opt2 = MyOptimizer(dim=2, name="opt2") # y0, y1

    # p1 exposes its x1 and x2 (indices 1, 2 of its domain)
    p1 = Open(domain=3, problem=opt1, exposed=[1, 2]) 
    # p2 exposes its y0 (index 0 of its domain)
    p2 = Open(domain=2, problem=opt2, exposed=[0])

    # Mapping: p1.exposed[0] (which is opt1's x1) to p2.exposed[0] (which is opt2's y0)
    # This means domain index 1 of opt1 is mapped to domain index 0 of opt2.
    # pullback_legs will be: ( (1,), (0,) ) 
    print(f"p1: {p1}")
    print(f"p2: {p2}")

    print("\n--- Composition 1: Default (keep_mapped_vars_exposed=False) ---")
    # The algebra's laxator for ContinuousOptPy for (3-dim, 2-dim, 1 mapped) -> 4-dim problem
    # p1 vars: 0, 1(mapped), 2. p2 vars: 0(mapped), 1.
    # New map (example from ContinuousOptPy): 
    #   final_map_p1: [0,1,2] -> [0,2,3] (idx 1 of p1 maps to new idx 2)
    #   final_map_p2: [0,1]   -> [2,4]   (idx 0 of p2 maps to new idx 2, idx 1 of p2 maps to new idx 4)
    #   (Error in manual trace, actual from code: p1:[0,1,2]->[0,2,3], p2:[0,1]->[2,4] with new domain 4. No, this example is wrong.)
    #   Correct trace for ContinuousOptPy laxator: (domain_1, domain_2, num_mapped_vars) -> new_domain = d1+d2-num_mapped
    #   p1_vars: x0, x1, x2. p2_vars: y0, y1. map x1 to y0. New vars: x0, (x1,y0), x2, y1
    #   So, new domain size = 3+2-1 = 4.
    #   final_map[0] (for p1): x0->0, x1->1, x2->2 (e.g., simple shift if x1 is the shared)
    #   final_map[1] (for p2): y0->1, y1->3
    #   This depends on how laxator assigns new indices. Let's use actual ContinuousOptPy laxator.
    
    # Using ContinuousOptPy as the algebra for a concrete test
    from .optimizers import ContinuousOptPy
    concrete_algebra = ContinuousOptPy()

    # p1 exposes domain indices [1,2]. p2 exposes domain index [0].
    # Mapping: p1.exposed_list_idx[0] (domain idx 1) maps to p2.exposed_list_idx[0] (domain idx 0)
    # mapping = {0:0} means self.exposed[0] maps to other.exposed[0]
    # self.exposed[0] is p1's domain index 1. other.exposed[0] is p2's domain index 0.
    # So, pullback_legs = ((1,), (0,))
    p_composed_default = p1.compose(p2, {0:0}, algebra=concrete_algebra, keep_mapped_vars_exposed=False)
    # Exposed from p1 not in mapping: p1.exposed[1] (domain idx 2) -> final_map[0][2]
    # Exposed from p2 not in mapping: none.
    # Expected: final_map[0] for p1_domain_idx=1 is the mapped var index.
    #           final_map[0] for p1_domain_idx=2 is p1_unmapped_exposed var index.
    # ContinuousOptPy laxator: d1=3, d2=2, legs=((1,),(0,)) -> new_dim=4
    # final_map = ([0,2,3], [2,1])  <-- This final_map from a previous trace seems to have an issue with p2 map length
    # Let's re-verify ContinuousOptPy.laxator behavior. Output should be: 
    # composed_problem (dim 4), ([map_p1_idx_to_new_idx], [map_p2_idx_to_new_idx])
    # For p1(dim 3), p2(dim 2), legs=((1,),(0,)):  new_dim = 3+2-1=4.
    #   p1_map: [0,1,2] -> [idx_for_0, idx_for_1, idx_for_2]
    #   p2_map: [0,1]   -> [idx_for_0, idx_for_1]
    #   idx_for_p1_1 == idx_for_p2_0
    #   Example map: p1_map=[0,1,3], p2_map=[1,2] (p1_idx1,p2_idx0 map to new_idx1; p1_idx0->0, p1_idx2->3, p2_idx1->2)
    #   p1.exposed = [1,2]. p1.exposed[0]=1 (mapped). p1.exposed[1]=2 (unmapped).
    #   new_exposed_from_self_unmapped = [p1_map[2]] = [3]
    #   new_exposed_from_other_unmapped = [p2_map[1]] = [2]
    #   Total exposed (default) = sorted(unique([3,2])) = [2,3]
    print(f"Composed (default): {p_composed_default}")

    print("\n--- Composition 2: keep_mapped_vars_exposed=True ---")
    p_composed_keep_exposed = p1.compose(p2, {0:0}, algebra=concrete_algebra, keep_mapped_vars_exposed=True)
    #   Default exposed: [2,3]
    #   Mapped var from self: p1.exposed[0] (domain idx 1). Its new index is p1_map[1] = 1.
    #   Total exposed (keep_mapped) = sorted(unique([2,3,1])) = [1,2,3]
    print(f"Composed (keep_exposed): {p_composed_keep_exposed}")

    # Test with more complex mapping
    optA = MyOptimizer(dim=4, name="optA") # a0,a1,a2,a3
    optB = MyOptimizer(dim=3, name="optB") # b0,b1,b2
    # pA exposes a1,a2,a3 (domain indices 1,2,3 of optA)
    pA = Open(domain=4, problem=optA, exposed=[1,2,3])
    # pB exposes b0,b1 (domain indices 0,1 of optB)
    pB = Open(domain=3, problem=optB, exposed=[0,1])
    print(f"\npA: {pA}")
    print(f"pB: {pB}")

    # Mapping: pA.exposed[0](a1) to pB.exposed[0](b0)
    #          pA.exposed[1](a2) to pB.exposed[1](b1)
    # Pullback_legs: ((1,2), (0,1))
    # New dim = 4+3-2 = 5
    # Expected map (conceptual): 
    #   pA_map: a0->0, a1->1, a2->2, a3->3
    #   pB_map: b0->1, b1->2, b2->4
    # pA.exposed=[1,2,3]. Mapped: 1,2. Unmapped: 3.
    #   new_exposed_from_A_unmapped = [pA_map[3]] = [3]
    # pB.exposed=[0,1]. Mapped: 0,1. Unmapped: none.
    #   new_exposed_from_B_unmapped = []
    # Total exposed (default) = [3]
    # Total exposed (keep_mapped) = sorted(unique([3, pA_map[1], pA_map[2]])) = sorted(unique([3,1,2])) = [1,2,3]
    mapping_AB = {0:0, 1:1} 
    print(f"Mapping for pA, pB: {mapping_AB}")
    p_AB_default = pA.compose(pB, mapping_AB, algebra=concrete_algebra, keep_mapped_vars_exposed=False)
    print(f"Composed AB (default): {p_AB_default}")
    p_AB_keep_exposed = pA.compose(pB, mapping_AB, algebra=concrete_algebra, keep_mapped_vars_exposed=True)
    print(f"Composed AB (keep_exposed): {p_AB_keep_exposed}")