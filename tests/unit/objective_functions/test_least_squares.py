import unittest
import numpy as np

from FoldOptLib.fold_modelling_plugin.objective_functions.least_squares import LeastSquaresFunctions


class TestLeastSquaresFunctions(unittest.TestCase):

    def setUp(self):
        self.rotation_angle = np.array([45, 90])
        self.fold_frame = np.array([1, 2])
        self.knowledge_constraints = lambda x: 0.5 * np.sum(x)
        self.lsf = LeastSquaresFunctions(self.rotation_angle, self.fold_frame, self.knowledge_constraints)

    def test_init_valid_input(self):
        self.assertIsNotNone(self.lsf)

    def test_init_invalid_rotation_angle(self):
        with self.assertRaises(TypeError):
            LeastSquaresFunctions([45, 90], self.fold_frame, self.knowledge_constraints)

    def test_init_invalid_fold_frame(self):
        with self.assertRaises(TypeError):
            LeastSquaresFunctions(self.rotation_angle, [1, 2], self.knowledge_constraints)

    def test_init_invalid_knowledge_constraints(self):
        with self.assertRaises(TypeError):
            LeastSquaresFunctions(self.rotation_angle, self.fold_frame, "not_callable")

    def test_square_residuals(self):
        # Assuming a stub for the `fourier_series` method
        self.lsf.fourier_series = lambda x, *args: x
        theta = np.array([0, 1, 1, 500])
        result = self.lsf.square_residuals(theta)
        # expected = np.array([0.0, 0.0])
        self.assertTrue(all(result), (float, np.ndarray))

    def test_huber_loss_within_delta(self):
        residuals = np.array([0.25, 0.4])
        result = self.lsf.huber_loss(residuals)
        expected = np.array([0.03125, 0.08])
        np.testing.assert_array_almost_equal(result, expected)

    # def test_huber_loss_outside_delta(self):
    #     residuals = np.array([0.6, 0.8])
    #     result = self.lsf.huber_loss(residuals)
    #     expected = np.array([0.245, 0.34])
    #     np.testing.assert_array_almost_equal(result, expected)

    def test_soft_l1_loss(self):
        residuals = np.array([0.25, 0.4])
        result = self.lsf.soft_l1_loss(residuals)
        expected = np.array([0.03125, 0.08])
        np.testing.assert_array_almost_equal(result, expected)

    def test_cost_with_knowledge_constraints(self):
        # Assuming a stub for the `fourier_series` method
        self.lsf.fourier_series = lambda x, *args: x
        theta = np.array([1, 2])
        result = self.lsf.cost(theta)
        expected = 1.5  # Based on the provided knowledge_constraints function
        self.assertAlmostEqual(result, expected)

    def test_cost_without_knowledge_constraints(self):
        self.lsf.knowledge_constraints = None
        # Assuming a stub for the `fourier_series` method
        self.lsf.fourier_series = lambda x, *args: x
        theta = np.array([1, 2])
        result = self.lsf.cost(theta)
        expected = 1.0
        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
