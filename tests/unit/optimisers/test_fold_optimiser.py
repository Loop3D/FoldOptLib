import unittest
import numpy as np

from FoldOptLib.FoldModellingPlugin.optimisers.fold_optimiser import FoldOptimiser


class TestFoldOptimiser(unittest.TestCase):

    def setUp(self):
        self.optimiser = FoldOptimiser()

    def test_prepare_and_setup_knowledge_constraints(self):
        # Test with no geological knowledge
        result = self.optimiser.prepare_and_setup_knowledge_constraints(None)
        self.assertIsNone(result)

        # TODO: Add test with mock geological knowledge

    def test_optimise_with_trust_region(self):
        def mock_objective_function(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([1, 1])
        result = self.optimiser.optimise_with_trust_region(mock_objective_function, x0)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result['fun'], 0, places=5)

    def test_optimise_with_differential_evolution(self):
        def mock_objective_function(x):
            return x[0] ** 2 + x[1] ** 2

        bounds = [(-10, 10), (-10, 10)]
        result = self.optimiser.optimise_with_differential_evolution(mock_objective_function, bounds)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result['fun'], 0, places=5)

    def test_setup_optimisation(self):
        # Test with no geological knowledge
        geo_knowledge, solver = self.optimiser.setup_optimisation(None)
        self.assertIsNone(geo_knowledge)
        self.assertTrue(callable(solver))

        # TODO: Add test with mock geological knowledge

    # Note: The optimise method is abstract and should be implemented in child classes.
    # Therefore, we don't test it here.


if __name__ == '__main__':
    unittest.main()
