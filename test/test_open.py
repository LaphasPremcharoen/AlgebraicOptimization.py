import unittest
import numpy as np
from algebraic_optimization_py.compositional_programming.open_flow_graphs import Open
from algebraic_optimization_py.compositional_programming.objectives import PrimalObjective

class TestCompose(unittest.TestCase):
    def test_simple_linear(self):
        # Define two simple linear objectives
        f = PrimalObjective(2, lambda x: x[0] + 2*x[1])
        g = PrimalObjective(2, lambda x: 3*x[0] + x[1])
        open_f = Open(2, f, [0,1])
        open_g = Open(2, g, [0,1])
        # Compose mapping x0 -> y1
        comp = open_f.compose(open_g, {0:1})

        # Check domain and exposed indices
        self.assertEqual(comp.domain, 3)
        self.assertEqual(comp.exposed, [0, 2])

        # Evaluate at a sample point
        x = np.array([1.0, 2.0, 5.0])
        # Construct expected full vectors
        full_f = np.array([x[1], x[0]])  # self_only=[1], shared=[0]
        full_g = np.array([x[2], x[1]])  # other_only=[0], shared=[0]
        expected = f(full_f) + g(full_g)
        self.assertAlmostEqual(comp(x), expected, places=6)

if __name__ == '__main__':
    unittest.main()
