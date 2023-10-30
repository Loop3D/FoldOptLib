import unittest
import numpy as np
from typing import List
from FoldOptLib.fold_modelling_plugin.objective_functions import VonMisesFisher


class TestVonMisesFisher(unittest.TestCase):

    def setUp(self):
        self.mu = [1, 0, 0]
        self.kappa = 1
        self.vmf = VonMisesFisher(self.mu, self.kappa)

    def test_init_valid_input(self):
        self.assertIsNotNone(self.vmf)

    def test_init_invalid_mu(self):
        with self.assertRaises(ValueError):
            VonMisesFisher([1, 0], self.kappa)

    def test_init_invalid_kappa(self):
        with self.assertRaises(ValueError):
            VonMisesFisher(self.mu, -1)

    def test_pdf(self):
        x = np.array([[1, 0, 0], [0, 1, 0]])
        result = self.vmf.pdf(x)
        self.assertEqual(result.shape, (2,))

    def test_logpdf(self):
        x = np.array([[1, 0, 0], [0, 1, 0]])
        result = self.vmf.logpdf(x)
        self.assertEqual(result.shape, (2,))

    def test_draw_samples_valid_input(self):
        samples = self.vmf.draw_samples(size=5, random_state=42)
        self.assertEqual(samples.shape, (5, 3))

    def test_draw_samples_invalid_size(self):
        with self.assertRaises(TypeError):
            self.vmf.draw_samples(size="5", random_state=42)

    def test_draw_samples_invalid_random_state(self):
        with self.assertRaises(TypeError):
            self.vmf.draw_samples(size=5, random_state="42")


if __name__ == "__main__":
    unittest.main()
