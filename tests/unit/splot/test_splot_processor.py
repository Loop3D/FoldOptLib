import unittest
import numpy as np
from LoopStructural.modelling.features.fold import fourier_series
from FoldOptLib.fold_modelling_plugin.splot.splot_processor import SPlotProcessor
from FoldOptLib.fold_modelling_plugin.helper.utils import fourier_series_x_intercepts


class TestSPlotProcessor(unittest.TestCase):

    def test_init(self):
        splot_processor = SPlotProcessor()
        self.assertIsNone(splot_processor.x)
        # self.assertEqual(splot_processor.splot_cache, {})
        self.assertEqual(splot_processor.splot_function_map, {4: fourier_series})
        self.assertEqual(splot_processor.intercept_function_map, {4: fourier_series_x_intercepts})
        # self.assertEqual(splot_processor.constraints, {})

    def test_find_amax_amin(self):
        splot_processor = SPlotProcessor()
        x = np.linspace(0, 10, 100)
        splot_processor.x = x

        theta = np.array([0, 1, 1, 500])
        amax, amin = splot_processor.find_amax_amin(theta)
        # self.assertAlmostEqual(amax, np.arctan(np.deg2rad(30.0)))
        # self.assertAlmostEqual(amin, np.arctan(np.deg2rad(-30.0)))
        self.assertTrue(amin < amax)

    def test_calculate_splot(self):
        splot_processor = SPlotProcessor()
        x = np.linspace(0, 10, 100)
        splot_processor.x = x

        theta = np.array([0, 1, 1, 500])
        curve = splot_processor.calculate_splot(x, theta)
        self.assertTrue(len(curve) == len(x))
        # self.assertAlmostEqual(curve[0], 0.0)
        # self.assertAlmostEqual(curve[-1], 2.0)

    def test_calculate_tightness(self):
        splot_processor = SPlotProcessor()
        x = np.linspace(0, 10, 100)
        splot_processor.x = x

        theta = np.array([0, 1, 1, 500])
        tightness = splot_processor.calculate_tightness(theta)
        self.assertTrue(tightness > 0.0)
        self.assertTrue(tightness < 180.0)

    def test_calculate_asymmetry(self):
        splot_processor = SPlotProcessor()
        x = np.linspace(0, 10, 100)
        splot_processor.x = x

        theta = np.array([0, 1, 1, 500])
        asymmetry = splot_processor.calculate_asymmetry(theta)
        self.assertTrue(asymmetry >= 0.0)


if __name__ == "__main__":
    unittest.main()
