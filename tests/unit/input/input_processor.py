import unittest
import pandas as pd
import numpy as np
from FoldOptLib.fold_modelling_plugin.helper.utils import *
from FoldOptLib.fold_modelling_plugin.input.input_data_processor import InputDataProcessor


class TestInputDataProcessor(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.valid_data_strike_dip = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [1, 2, 3],
            'Z': [1, 2, 3],
            'feature_name': ['fold1', 'fold2', 'fold3'],
            'strike': [90, 90, 90],
            'dip': [30, 30, 30]
        })
        self.valid_data_gradient = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [1, 2, 3],
            'Z': [1, 2, 3],
            'feature_name': ['fold1', 'fold2', 'fold3'],
            'gx': [1, 1, 1],
            'gy': [0, 0, 0],
            'gz': [0, 0, 0]
        })
        self.bounding_box = np.array([[0, 3, 0], [3, 0, 3]])
        self.knowledge = {'fold_axial_surface': {'mu': 1, 'kappa': 2, 'w': 3}}

    def test_process_data_strike_dip(self):
        processor = InputDataProcessor(self.valid_data_strike_dip, self.bounding_box, self.knowledge)
        processed_data = processor.process_data()
        self.assertIn('gx', processed_data.columns)
        self.assertIn('gy', processed_data.columns)
        self.assertIn('gz', processed_data.columns)

    def test_process_data_gradient(self):
        processor = InputDataProcessor(self.valid_data_gradient, self.bounding_box, self.knowledge)
        processed_data = processor.process_data()
        self.assertIn('gx', processed_data.columns)
        self.assertIn('gy', processed_data.columns)
        self.assertIn('gz', processed_data.columns)

    # TODO: Add more tests for different scenarios, such as invalid data, missing columns, etc.


if __name__ == '__main__':
    unittest.main()
