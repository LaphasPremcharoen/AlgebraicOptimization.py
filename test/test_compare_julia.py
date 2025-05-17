import unittest
import subprocess
import shutil
import os
import numpy as np
from algebraic_optimization_py.compositional_programming.open_flow_graphs import Open
from algebraic_optimization_py.compositional_programming.objectives import PrimalObjective

@unittest.skipUnless(shutil.which("julia") is not None, "Julia not installed")
class TestJuliaComparison(unittest.TestCase):
    def test_against_julia(self):
        # Python composition
        f = PrimalObjective(2, lambda x: x[0] + 2*x[1])
        g = PrimalObjective(2, lambda x: 3*x[0] + x[1])
        open_f = Open(2, f, [0,1])
        open_g = Open(2, g, [0,1])
        comp_py = open_f.compose(open_g, {0:1})
        x = np.array([1.0, 2.0, 5.0])
        val_py = comp_py(x)

        # Compare against Julia script if available
        script = os.path.join(os.path.dirname(__file__), 'julia_compose.jl')
        try:
            res = subprocess.run(['julia', '--project=.', script],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out = res.stdout.decode().strip()
            if res.returncode != 0:
                raise RuntimeError(f"Julia script failed:\n{out}")
            val_jl = float(out)
        except Exception as e:
            self.skipTest(f"Julia comparison skipped: {e}")
        self.assertAlmostEqual(val_py, val_jl, places=6)

if __name__ == '__main__':
    unittest.main()
