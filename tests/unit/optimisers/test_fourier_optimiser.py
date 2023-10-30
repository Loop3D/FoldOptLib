import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
from FoldOptLib.FoldModellingPlugin.optimisers.fourier_optimiser import FourierSeriesOptimiser
from FoldOptLib.FoldModellingPlugin.helper.utils import *


class TestFourierSeriesOptimiser(unittest.TestCase):

    def setUp(self):
        self.fold_frame_coordinate = np.linspace(-10, 10, 100)
        self.rotation_angle = np.linspace(-60, 60, 100)
        self.x = np.array([1, 2, 3])

        self.geological_knowledge = {
            'fold_limb_rotation_angle': {
                'tightness': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'asymmetry': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'fold_wavelength': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'axial_trace_1': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_2': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_3': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_4': {'mu': 10, 'sigma': 10, 'w': 10}
            }
        }
        self.optimiser = FourierSeriesOptimiser(self.fold_frame_coordinate, self.rotation_angle,
                                                self.x,
                                                method='differential_evolution')

    def test_prepare_and_setup_knowledge_constraints(self):
        # Test with no geological knowledge
        result = self.optimiser.prepare_and_setup_knowledge_constraints()
        self.assertIsNone(result)

        # TODO: Add test with mock geological knowledge

    def test_generate_initial_guess(self):
        # Test with no method and no wl_guess in kwargs
        guess = self.optimiser.generate_initial_guess()
        self.assertIsInstance(guess, np.ndarray)

        # TODO: Add more tests for different kwargs configurations -> Done

    def test_setup_optimisation(self):
        geological_knowledge = {
            'fold_limb_rotation_angle': {
                'tightness': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'asymmetry': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'fold_wavelength': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'axial_trace_1': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_2': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_3': {'mu': 10, 'sigma': 10, 'w': 10},
                'axial_traces_4': {'mu': 10, 'sigma': 10, 'w': 10}
            }
        }
        obj_func, geo_knowledge, solver, guess = self.optimiser.setup_optimisation(
            geological_knowledge=geological_knowledge['fold_limb_rotation_angle'])
        self.assertTrue(callable(obj_func))
        self.assertIsInstance(guess, np.ndarray)
        self.assertTrue(callable(solver))
        self.assertIsNotNone(geo_knowledge)  # As per the current setup, it should be None

    def test_optimise(self):
        result = self.optimiser.optimise(geological_knowledge=self.geological_knowledge['fold_limb_rotation_angle'])
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
