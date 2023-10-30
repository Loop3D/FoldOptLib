import unittest
import pandas as pd
import numpy as np
# import CheckInputData
from FoldOptLib.fold_modelling_plugin.input.input_data_checker import CheckInputData


# class TestCheckInputData(unittest.TestCase):
#
#     def setUp(self):
#         # self.invalid_foliation_data = pd.DataFrame({
#         #     'X': [1, 2, 3],
#         #     'Y': [4, 5, 6],
#         #     'Z': [7, 8, 9],
#         #     'feature_name': ['a', 'b', 'c'],
#         #     'invalid_column': [10, 11, 12]
#         # })
#         # self.invalid_bounding_box = np.array([[1, 2], [3, 4]])
#         # self.invalid_knowledge_constraints = {
#         #     'invalid_key': {'mu': 1, 'sigma': 2, 'w': 3},
#         #     'asymmetry': {'mu': 4, 'sigma': 5, 'w': 6},
#         #     'fold_wavelength': {'mu': 7, 'sigma': 8, 'w': 9},
#         #     'axial_trace': {'mu': 10, 'sigma': 11, 'w': 12},
#         #     'axial_surface': {'mu': 13, 'sigma': 14, 'w': 15}
#         # }
#         self.invalid_foliation_data = dict({
#             'X': [1, 2, 3],
#             'Y': [4, 5, 6],
#             'Z': [7, 8, 9],
#             'feature_name': ['a', 'b', 'c'],
#             'invalid_column': [10, 11, 12],
#             'gx': [0.1, 0.2, 0.3],
#             'gy': [0.4, 0.5, 0.6],
#             # 'gz': [0.7, 0.8, 0.9]
#         })
#         self.invalid_dataframe = pd.DataFrame({
#             'X': [1, 2, 3],
#             'Y': [4, 5, 6],
#             'Z': [7, 8, 9],
#             'feature_name': ['a', 'b', 'c'],
#             'invalid_column': [10, 11, 12],
#             'gx': [0.1, 0.2, 0.3],
#             'gy': [0.4, 0.5, 0.6],
#             # 'gz': [0.7, 0.8, 0.9]
#         })
#         self.invalid_bounding_box = np.array([[0, 0], [10, 10]])
#         self.invalid_knowledge_constraints = {
#             'invalid_key': {'mu': 1, 'sigma': 2, 'w': 3},
#             'asymmetry': {'mu': 4, 'sigma': 5, 'w': 6},
#             'fold_wavelength': {'mu': 7, 'sigma': 8, 'w': 9},
#             'axial_trace': {'mu': 10, 'sigma': 11, 'w': 12},
#             'axial_surface': {'sigma': 14, 'w': 15}
#         }
#
#     def test_check_foliation_data(self):
#         check_input_data = CheckInputData(self.invalid_foliation_data, None, None)
#         with self.assertRaises(TypeError):
#             check_input_data.check_foliation_data()
#         # with self.assertRaises(ValueError):
#         #     check_input_data.check_foliation_data()
#
#     def test_check_foliation_data_dataframe(self):
#         check_input_data = CheckInputData(self.invalid_dataframe, None, None)
#         with self.assertRaises(ValueError):
#             check_input_data.check_foliation_data()
#
#     def test_check_knowledge_constraints(self):
#         check_input_data = CheckInputData(None, None, self.invalid_knowledge_constraints)
#         with self.assertRaises(ValueError):
#             check_input_data.check_knowledge_constraints()
#
#     def test_check_bounding_box(self):
#         check_input_data = CheckInputData(None, self.invalid_bounding_box, None)
#         with self.assertRaises(ValueError):
#             check_input_data.check_bounding_box()
#
#     def test_check_input_data(self):
#         check_input_data = CheckInputData(self.invalid_foliation_data, self.invalid_bounding_box,
#                                           self.invalid_knowledge_constraints)
#         with self.assertRaises(ValueError):
#             check_input_data.check_input_data()
#
#
# if __name__ == '__main__':
#     unittest.main()

class TestCheckInputData(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.valid_foliation_data = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [1, 2, 3],
            'Z': [1, 2, 3],
            'feature_name': ['fold1', 'fold2', 'fold3'],
            'strike': [90, 90, 90],
            'dip': [30, 30, 30]
        })
        self.valid_bounding_box = np.array([[0, 0, 0], [3, 3, 3]])
        self.valid_knowledge = {'fold_axial_surface': {'mu': [1, 0, 0], 'kappa': 2, 'w': 3}}

    def test_check_foliation_data_valid(self):
        checker = CheckInputData(self.valid_foliation_data, self.valid_bounding_box)
        self.assertIsNone(checker.check_foliation_data())  # Expecting no exceptions

    def test_check_foliation_data_invalid(self):
        invalid_data = self.valid_foliation_data.drop(columns=['X'])
        checker = CheckInputData(invalid_data, self.valid_bounding_box)
        with self.assertRaises(ValueError):
            checker.check_foliation_data()

    def test_check_bounding_box_valid(self):
        checker = CheckInputData(self.valid_foliation_data, self.valid_bounding_box)
        self.assertIsNone(checker.check_bounding_box())  # Expecting no exceptions

    def test_check_bounding_box_invalid(self):
        invalid_box = np.array([[0, 3], [3, 0]])
        checker = CheckInputData(self.valid_foliation_data, invalid_box)
        with self.assertRaises(ValueError):
            checker.check_bounding_box()

    def test_check_input_geological_knowledge_valid(self):
        checker = CheckInputData(self.valid_foliation_data, self.valid_bounding_box, self.valid_knowledge)
        self.assertTrue(checker.check_input_geological_knowledge())

    # TODO: Add more tests for invalid geological knowledge, check_input_data, etc.

if __name__ == '__main__':
    unittest.main()
