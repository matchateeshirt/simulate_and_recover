import unittest
import numpy as np
from src.simulate import generate_true_parameters, forward_equations, inverse_equations

class TestEZDiffusionModel(unittest.TestCase):

    def test_generate_true_parameters(self):
        """Test if generated parameters are within expected ranges."""
        a, v, t = generate_true_parameters()
        self.assertTrue(0.5 <= a <= 2)
        self.assertTrue(0.5 <= v <= 2)
        self.assertTrue(0.1 <= t <= 0.5)

    def test_forward_inverse_consistency(self):
        """Test if inverse equations correctly recover parameters from forward equations."""
        a, v, t = 1.0, 1.2, 0.3  # Fixed values for testing
        R_pred, M_pred, V_pred = forward_equations(a, v, t)
        a_est, v_est, t_est = inverse_equations(R_pred, M_pred, V_pred)
        self.assertAlmostEqual(a, a_est, places=2)
        self.assertAlmostEqual(v, v_est, places=2)
        self.assertAlmostEqual(t, t_est, places=2)

if __name__ == "__main__":
    unittest.main()
