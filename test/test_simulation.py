import sys
import os
import unittest
import numpy as np

# Ensure src/ is added to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now import the required functions
from simulate import forward_equations
from recover import inverse_equations

class TestEZDiffusionModel(unittest.TestCase):

    def test_forward_inverse_consistency(self):
        """Test if inverse equations correctly recover parameters from forward equations."""
        a, v, t = 1.0, 1.2, 0.3  # Fixed test values
        R_pred, M_pred, V_pred = forward_equations(a, v, t)
        a_est, v_est, t_est = inverse_equations(R_pred, M_pred, V_pred)

        # Allow small numerical error tolerance
        self.assertAlmostEqual(a, a_est, delta=0.2)  # Allowing a difference up to 0.2
        self.assertAlmostEqual(v, v_est, delta=0.3)
        self.assertAlmostEqual(t, t_est, delta=0.2)

if __name__ == "__main__":
    unittest.main()
