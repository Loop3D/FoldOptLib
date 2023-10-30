import unittest
import numpy as np
from FoldOptLib.fold_modelling_plugin.objective_functions import (gaussian_log_likelihood, loglikelihood,
                                                                  loglikelihood_axial_surface, loglikelihood_fourier_series)

# class TestYourFunctions(unittest.TestCase):
#
#     def test_gaussian_log_likelihood(self):
#         self.assertAlmostEqual(
#             gaussian_log_likelihood(1.0, 0.0, 1.0), -1.4189385332046727)
#         self.assertAlmostEqual(
#             gaussian_log_likelihood(2.0, 1.0, 0.5), -2.8378770664093455)
#         with self.assertRaises(ValueError):
#             gaussian_log_likelihood(1.0, 0.0, 0.0)  # sigma <= 0 should raise ValueError
#
#     def test_loglikelihood(self):
#         self.assertAlmostEqual(loglikelihood(1.0, 1.0), -1.4189385332046727)
#         self.assertAlmostEqual(loglikelihood(2.0, 2.0), -1.4189385332046727)
#         # Add more test cases for different values
#
#     def test_loglikelihood_axial_surface(self):
#         self.assertAlmostEqual(loglikelihood_axial_surface(0.0), 3.7590959995303586)
#         self.assertAlmostEqual(loglikelihood_axial_surface(np.pi/2), -2.6855758216157585)
#         # Add more test cases for different values
#
#     def test_loglikelihood_fourier_series(self):
#         rotation_angle = np.array([10.0, 20.0, 30.0])
#         fold_frame_coordinate = np.array([1.0, 2.0, 3.0])
#         objective_fn = loglikelihood_fourier_series(rotation_angle, fold_frame_coordinate)
#         result = objective_fn([0.0, 1.0, 2.0, 3.0])
#         self.assertAlmostEqual(result, -3.0726032493525896)
#         # Add more test cases for different inputs
#
# if __name__ == '__main__':
#     unittest.main()

class TestGaussianLogLikelihood(unittest.TestCase):

    def test_valid_input(self):
        result = gaussian_log_likelihood(5, 5, 1)
        self.assertAlmostEqual(result, -0.9189385332046727)  # Expected value based on the formula

    def test_invalid_sigma(self):
        with self.assertRaises(ValueError):
            gaussian_log_likelihood(5, 5, 0)

    def test_negative_sigma(self):
        with self.assertRaises(ValueError):
            gaussian_log_likelihood(5, 5, -1)


class TestLogLikelihood(unittest.TestCase):

    def test_valid_input(self):
        result = loglikelihood(5, 5)
        self.assertAlmostEqual(result, -3.6862316527834187)  # Expected value based on the formula


class TestLogLikelihoodAxialSurface(unittest.TestCase):

    def test_valid_input(self):
        result = loglikelihood_axial_surface(1)
        # Expected value based on the VonMises distribution
        self.assertAlmostEqual(result, 1.837877066405363)


class TestLogLikelihoodFourierSeries(unittest.TestCase):

    def test_valid_input(self):
        # This test assumes the existence of a `get_predicted_rotation_angle` function
        # You might need to mock or stub this function for the test to work
        pass  # TODO: Implement this test based on the behavior of `get_predicted_rotation_angle`


if __name__ == "__main__":
    unittest.main()

