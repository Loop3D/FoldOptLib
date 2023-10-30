import unittest
import pandas as pd
import numpy as np
import sys

# sys.path.append("/FoldModellingPlugin/FoldModellingPlugin/fold_modelling")
from FoldOptLib.FoldModellingPlugin.fold_modelling.engine import FoldModel


class TestFoldModel(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [4, 5, 6],
            'Z': [7, 8, 9],
            'feature_name': ['s0', 's0', 's0'],
            'gx': [0.1, 0.2, 0.3],
            'gy': [0.4, 0.5, 0.6],
            'gz': [0.7, 0.8, 0.9]
        })
        self.bounding_box = np.array([[0, 0, 0], [10, 10, 10]])
        self.fold_model = FoldModel(self.data, self.bounding_box, av_fold_axis=False)

    def test_initialise_model(self):
        self.fold_model.initialise_model()
        self.assertIsNotNone(self.fold_model.model)
        self.assertIsNotNone(self.fold_model.scaled_points)

    def test_process_axial_surface_proposition(self):
        axial_normal = np.array([1., 0., 0.])
        dataset = self.fold_model.process_axial_surface_proposition(axial_normal)
        self.assertTrue(isinstance(dataset, pd.DataFrame))
        self.assertEqual(len(dataset), len(self.data) * 2)

    def test_build_fold_frame(self):
        axial_normal = np.array([1., 0., 0.])
        self.fold_model.initialise_model()
        self.fold_model.build_fold_frame(axial_normal)
        self.assertIsNotNone(self.fold_model.axial_surface)

    def test_create_and_build_fold_event(self):
        axial_normal = np.array([1., 0., 0.])
        self.fold_model.initialise_model()
        self.fold_model.build_fold_frame(axial_normal)
        fold_event = self.fold_model.create_and_build_fold_event()
        self.assertIsNotNone(fold_event)

    def test_calculate_svariogram(self):
        fold_frame = np.array([1, 2, 3])
        rotation_angles = np.array([0.1, 0.2, 0.3])
        svariogram = self.fold_model.calculate_svariogram(fold_frame, rotation_angles)
        self.assertTrue(isinstance(svariogram, np.ndarray))
        self.assertEqual(len(svariogram), 4)

    def test_fit_fourier_series(self):
        fold_frame_coordinate = np.array([1, 2, 3])
        rotation_angle = np.array([0.1, 0.2, 0.3])
        fourier_series = self.fold_model.fit_fourier_series(fold_frame_coordinate, rotation_angle)
        self.assertTrue(isinstance(fourier_series, list))
        self.assertEqual(len(fourier_series), 4)

    def test_calculate_folded_foliation_vectors(self):
        axial_normal = np.array([1, 0, 0])
        self.fold_model.initialise_model()
        self.fold_model.build_fold_frame(axial_normal)
        folded_foliation_vectors = self.fold_model.calculate_folded_foliation_vectors()
        self.assertTrue(isinstance(folded_foliation_vectors, np.ndarray))
        self.assertEqual(len(folded_foliation_vectors), len(self.data))


if __name__ == '__main__':
    unittest.main()
