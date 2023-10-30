import numpy as np
from typing import Union
from ..helper.utils import get_predicted_rotation_angle
from scipy.stats import vonmises


def gaussian_log_likelihood(b: Union[int, float], mu: Union[int, float], sigma: Union[int, float]) -> float:
    """
    Calculate the log-likelihood of a Gaussian distribution.

    This function calculates the log-likelihood of a Gaussian distribution for a given observed value, mean,
    and standard deviation. The formula used is the standard formula for the log-likelihood of a Gaussian distribution.

    Parameters
    ----------
    b : Union[int, float]
        The observed value.
    mu : Union[int, float]
        The mean of the Gaussian distribution.
    sigma : Union[int, float]
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    float
        The calculated log-likelihood.

    Raises
    ------
    ValueError
        If `sigma` is less than or equal to 0.
    """
    # Check if sigma is greater than 0
    if sigma <= 0:
        raise ValueError("`sigma` should be greater than 0.")

    # Calculate the log-likelihood
    likelihood = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (b - mu) ** 2 / sigma ** 2

    # Return the log-likelihood
    return likelihood


def loglikelihood(y, y_pred):
    sigma = 1e-2
    likelihood = -gaussian_log_likelihood(y, y_pred, sigma)
    return likelihood


def loglikelihood_axial_surface(x: float) -> Union[int, float]:
    """
    Objective function for the axial surface.
    This function calculates the loglikelihood of an axial surface using the VonMisesFisher distribution.

    Parameters
    ----------
    x : float
        represents the angle between the observed folded foliation and the predicted one.

    Returns
    -------
    Union[int, float]
        The logpdf value from the VonMises distribution.
    """
    # Define the mu and kappa of the VonMises distribution
    # mu = 0 because we want to minimises the angle between the observed and predicted folded foliation
    # kappa = 100 because we want to have a sharp distribution very close to the mean 0 (mu)
    mu = 1e-10
    kappa = 100

    # Create a VonMises distribution with the given parameters
    vm = vonmises(mu, kappa)

    # Calculate the logpdf of the input array
    vm_logpdf = -vm.logpdf(x)

    if isinstance(vm_logpdf, np.ndarray):
        vm_logpdf = vm_logpdf.sum()
    else:
        pass
    return vm_logpdf


def loglikelihood_fourier_series(rotation_angle, fold_frame_coordinate):
    def objective_fourier_series(theta):
        y = rotation_angle
        y_pred = get_predicted_rotation_angle(theta, fold_frame_coordinate)
        log_likelihood = 0
        for fr, fd in zip(y, y_pred):
            log_likelihood += loglikelihood(fr, fd)

        return log_likelihood

    return objective_fourier_series
