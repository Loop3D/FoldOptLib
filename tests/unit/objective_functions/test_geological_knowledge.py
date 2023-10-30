import unittest
import numpy as np

# Import the class to be tested
from FoldOptLib.fold_modelling_plugin.objective_functions.geological_knowledge import \
    GeologicalKnowledgeFunctions


class TestGeologicalKnowledgeFunctions(unittest.TestCase):

    def setUp(self):
        # Sample constraints and x values for testing
        self.constraints = {
            'fold_limb_rotation_angle': {
                'tightness': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'asymmetry': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'fold_wavelength': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'axial_trace_1': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_2': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_3': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_4': {'mu': 10, 'sigma': 10, 'w': 10},
            },
            'fold_axis_rotation_angle': {
                'hinge_angle': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'axis_wavelength': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
            },
            'fold_axial_surface': {
                'axial_surface': {'lb': 10, 'ub': 10, 'mu': [0.68, 0.6, 0.01], 'kappa': 10, 'w': 10}
            }
        }
        self.x = np.arange(-100., 100.)
        self.gkf = GeologicalKnowledgeFunctions(self.constraints['fold_limb_rotation_angle'], self.x)

    def test_axial_surface_objective_function(self):
        x = [0., 0., 1.]
        gkf = GeologicalKnowledgeFunctions(self.constraints['fold_axial_surface'], self.x)
        result = gkf.axial_surface_objective_function(x)
        self.assertIsInstance(result, float)

    def test_axial_trace_objective_function(self):
        theta = np.array([0., 1., 1., 500.])
        gkf = GeologicalKnowledgeFunctions(self.constraints['fold_limb_rotation_angle'], self.x)
        result = gkf.axial_trace_objective_function(theta)
        self.assertIsInstance(result, (float, list))

    def test_wavelength_objective_function(self):
        theta = np.array([0, 1, 1, 500])
        result = self.gkf.wavelength_objective_function(theta)
        self.assertIsInstance(result, float)

    def test_fold_axis_wavelength_objective_function(self):
        theta = np.array([0, 1, 1, 500])
        gkf = GeologicalKnowledgeFunctions(self.constraints['fold_axis_rotation_angle'], self.x)
        result = gkf.fold_axis_wavelength_objective_function(theta)
        self.assertIsInstance(result, float)

    def test_tightness_objective_function(self):
        theta = np.array([0, 1, 1, 500])
        result = self.gkf.tightness_objective_function(theta)
        self.assertIsInstance(result, float)

    def test_hinge_angle_objective_function(self):
        theta = np.array([0, 1, 1, 500])
        gkf = GeologicalKnowledgeFunctions(self.constraints['fold_axis_rotation_angle'], self.x)
        result = gkf.hinge_angle_objective_function(theta)
        self.assertIsInstance(result, float)

    def test_asymmetry_objective_function(self):
        theta = np.array([0, 1, 1, 500])
        result = self.gkf.asymmetry_objective_function(theta)
        self.assertIsInstance(result, float)

    # Add more tests for other methods ...


if __name__ == '__main__':
    unittest.main()
