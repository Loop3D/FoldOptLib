import unittest
import numpy as np
import pandas as pd
# import dill
import os

# from FoldModellingPlugin.FoldModellingPlugin.from_loopstructural._svariogram import SVariogram
from FoldOptLib.fold_modelling_plugin.helper.utils import *


class TestGeologicalFunctions(unittest.TestCase):

    def setUp(self):
        # Common setup for the tests
        self.fold_frame = np.array([1, 2])
        self.fold_rotation = np.array([45, 90])
        self.theta = np.array([0, 1, 1, 500])
        self.fold_frame_coordinate = 1

    def test_get_predicted_rotation_angle(self):
        result = get_predicted_rotation_angle(self.theta, self.fold_frame_coordinate)
        # Check if result is a numpy array
        self.assertIsInstance(result, np.ndarray)

    def test_fourier_series(self):
        popt = [1, 2, 3, 4]
        result = fourier_series(self.fold_frame, *popt)
        self.assertIsInstance(result, (float, np.ndarray))

    def test_fourier_series_x_intercepts(self):
        popt = [1., 2., 3., 4.]
        x = np.array([1., 2., 3., 4.])
        result = fourier_series_x_intercepts(x, popt)
        self.assertIsInstance(result, np.ndarray)

    def test_save_load_object(self):
        obj = {"key": "value"}
        file_path = "test_object.pkl"
        save_load_object(obj=obj, file_path=file_path, mode='save')
        loaded_obj = save_load_object(file_path=file_path, mode='load')
        self.assertEqual(obj, loaded_obj)
        os.remove(file_path)

    def test_strike_dip_to_vectors(self):
        strike = np.array([45, 90])
        dip = np.array([30, 60])
        result = strike_dip_to_vectors(strike, dip)
        self.assertIsInstance(result, np.ndarray)

    def test_strike_dip_to_vector(self):
        result = strike_dip_to_vector(45, 30)
        self.assertIsInstance(result, np.ndarray)

    def test_rotate_vector(self):
        v = np.array([1, 0])
        angle = np.pi / 4
        result = rotate_vector(v, angle)
        self.assertIsInstance(result, np.ndarray)

    def test_create_dict(self):
        result = create_dict(x=[1, 2], y=[1, 2], z=[1, 2], strike=[45, 90], dip=[30, 60], feature_name="test", coord=1,
                             data_type="foliation")
        self.assertIsInstance(result, dict)

    def test_create_gradient_dict(self):
        result = create_gradient_dict(x=[1, 2], y=[1, 2], z=[1, 2], nx=[0, 1], ny=[1, 0], nz=[0, 0],
                                      feature_name="test", coord=1)
        self.assertIsInstance(result, dict)

    def test_make_dataset(self):
        vec = np.array([1, 0, 0])
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = make_dataset(vec, points)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
