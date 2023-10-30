from ..helper.utils import *
import numpy as np
from LoopStructural.modelling.features.fold import fourier_series


class SPlotProcessor:
    def __init__(self):
        self.x = None
        # self.splot_cache = {}
        self.splot_function_map = {
            4: fourier_series,
            # 7: fourier_series_2
        }
        self.intercept_function_map = {
            4: fourier_series_x_intercepts,
            # 8: other_intercept_function
        }
        # self.constraints = constraints

    def find_amax_amin(self, theta, v='radians'):
        """
            Helper method to find the amax and amin values of a fold rotation angle curve

            Parameters
            ----------
            theta : float
                The fourier parameters

            Returns
            -------
            amax : float
                The maximum value of the curve in radians
            amin : float
                The minimum value of the curve in radians
            """

        curve = self.calculate_splot(self.x, theta)

        if v == 'radians':
            amax = np.arctan(np.deg2rad(curve.max()))
            amin = np.arctan(np.deg2rad(curve.min()))

            return amax, amin

        if v == 'degrees':
            amax = curve.max()
            amin = curve.min()

            return amax, amin, curve

    def calculate_splot(self, fold_frame, theta):
        """
            Calculates the curve using the passed ref_fold_frame and theta values, it uses
            the passed coeff value to determine the specific function to use.

            Parameters
            ----------
            fold_frame : array
                The fold frame coordinates of the fold rotation angles
            theta : float
                The fourier parameters

            Returns
            -------
            curve : array
                The calculated curve.

            """
        coeff = len(theta)
        # key = str(theta)
        result = None
        if coeff == 4:
            result = self.splot_function_map[coeff](fold_frame, *theta)
        # TODO: implement fourier series with variable number of coefficients
        # if coeff == 7:
        #     result = self.splot_function_map[coeff](fold_frame, theta)

        return result

    def calculate_tightness(self, theta):
        """
            Calculates the tightness metric of the curve using the passed theta value

            Parameters
            ----------
            theta : float
                The parameter that is used to calculate the curve

            Returns
            -------
            tightness : float
                The tightness metric of the curve
            """
        amax, amin = self.find_amax_amin(theta, v='radians')
        return 180 - np.rad2deg(2 * np.tan((amax - amin) / 2))

    def calculate_asymmetry(self, theta):
        """
            Calculates the fold asymmetry for a given set of fourier parameters.
            The asymmetry of a curve is a measure of how much it deviates from symmetry.

            Parameters:
            -----------
            theta : float
                The parameter used to calculate the curve

            Returns:
            --------
            asymmetry: float
                The asymmetry of the curve measured in degrees.

            """

        amax, amin, curve = self.find_amax_amin(theta, v='degrees')
        median = np.median(curve)
        tightness_range = amax + np.abs(amin)
        asymmetry = median / tightness_range

        return asymmetry
