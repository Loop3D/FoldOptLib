import unittest
import numpy as np
from FoldOptLib.fold_modelling_plugin.objective_functions.axial_plane import is_axial_plane_compatible


class TestIsAxialPlaneCompatible(unittest.TestCase):

    def test_valid_input(self):
        v1 = np.array([[1, 0, 0], [0, 1, 0]])
        v2 = np.array([[1, 0, 0], [0, 1, 0]])
        result = is_axial_plane_compatible(v1, v2)
        self.assertAlmostEqual(result, 0.0)  # Expected value based on the formula

    def test_mismatched_shapes(self):
        v1 = np.array([[1, 0, 0], [0, 1, 0]])
        v2 = np.array([[1, 0, 0]])
        with self.assertRaises(ValueError):
            is_axial_plane_compatible(v1, v2)

    def test_non_numpy_array_input(self):
        v1 = [1, 0, 0]
        v2 = [1, 0, 0]
        with self.assertRaises(ValueError):
            is_axial_plane_compatible(v1, v2)

    def test_angle_difference(self):
        v1 = np.array([[1, 0, 0]])
        v2 = np.array([[0, 1, 0]])
        result = is_axial_plane_compatible(v1, v2)
        self.assertAlmostEqual(result, np.pi/2)  # Expected value based on the formula


if __name__ == "__main__":
    unittest.main()


