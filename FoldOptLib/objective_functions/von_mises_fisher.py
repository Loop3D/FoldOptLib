from typing import List
from scipy.stats import vonmises_fisher
import numpy as np


class VonMisesFisher:
    """
    This class defines objective functions based on the Von Mises-Fisher (vMF) distribution.
    It can be used in Maximum Likelihood Estimation (MLE) optimization problems,
    and for drawing samples from a vMF distribution.
    """

    def __init__(self, mu: List[float], kappa: float):
        """
        Initialise the Von Mises-Fisher distribution class.

        Parameters
        ----------
        mu : List[float]
            Mean direction vector, in the form [mu_x, mu_y, mu_z].
        kappa : float
            Concentration parameter (always positive).

        Raises
        ------
        ValueError
            If `mu` is not a three-element vector or `kappa` is not a positive number.
        """
        # Check if mu is a three-element vector
        if len(mu) != 3:
            raise ValueError("`mu` should be a three-element vector.")
        # Check if kappa is a positive number
        if kappa <= 0:
            raise ValueError("`kappa` should be a positive number.")

        # Initialise the Von Mises Fisher distribution
        mu /= np.linalg.norm(mu)
        self.vmf = vonmises_fisher(mu, kappa)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Probability Density Function (PDF) of the Von Mises-Fisher distribution at a given point.

        Parameters
        ----------
        x : np.ndarray
            Points at which the PDF should be evaluated.

        Returns
        -------
        np.ndarray
            The evaluated PDF at each point in `x`.
        """
        # Evaluate the PDF of the Von Mises-Fisher distribution at each point in `x`
        pdf_values = self.vmf.pdf(x)

        return pdf_values

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the logarithm of the Probability Density Function (PDF) of the
        Von Mises-Fisher distribution at a given point.

        Parameters
        ----------
        x : np.ndarray
            Points at which the logarithm of the PDF should be evaluated.

        Returns
        -------
        np.ndarray
            The evaluated logarithm of the PDF at each point in `x`.
        """
        # Evaluate the natural logarithm of the PDF of the Von Mises-Fisher distribution at each point in `x`
        logpdf_values = self.vmf.logpdf(x)

        return logpdf_values


    def draw_samples(self, size: int = 1, random_state: int = 1) -> np.ndarray:
        """
        Draw random samples from the Von Mises-Fisher distribution.

        Parameters
        ----------
        size : int, optional
            The number of random samples to draw. Default is 1.
        random_state : int, optional
            The seed for the random number generator. Default is 1.

        Returns
        -------
        np.ndarray
            The array of drawn random samples. Each sample is a 3D vector

        Raises
        ------
        TypeError
            If `size` or `random_state` is not an integer.
        """
        # Check if `size` is an integer
        if not isinstance(size, int):
            raise TypeError("`size` should be an integer.")
        # Check if `random_state` is an integer
        if not isinstance(random_state, int):
            raise TypeError("`random_state` should be an integer.")

        # Draw random samples from the Von Mises-Fisher distribution
        random_samples = self.vmf.rvs(size=size, random_state=random_state)

        return random_samples
