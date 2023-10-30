import numpy as np
from scipy.optimize import NonlinearConstraint, BFGS
# from LoopStructural.modelling.features.fold import fourier_series
from typing import Union, Dict, List
from ..helper._helper import *
from ..helper.utils import *
from ..splot.splot_processor import SPlotProcessor
from .von_mises_fisher import VonMisesFisher
from .gaussian import gaussian_log_likelihood


def check_fourier_parameters(theta):
    if not isinstance(theta, np.ndarray):
        raise TypeError("`theta` should be a numpy array.")
    if len(theta) < 4:
        raise ValueError("`theta` should have at least 4 Fourier series parameters.")


class GeologicalKnowledgeFunctions(SPlotProcessor):
    """
    Base class for geological knowledge objective functions

    """

    def __init__(self, constraints: Dict[str, float]):

        # TODO add attribute to use any custom function otherwise use gaussian likelihood
        """
        Initialize the GeologicalKnowledgeConstraints class.

        Parameters
        ----------
        x : np.ndarray
            The values of the fold frame coordinates (0 or 1) used to calculate the fitted fold rotation angle curve.

        constraints : Dict[str, float]
            The constraints for the geological knowledge.
            The constraints dictionary should have the following structure:
            {
                'fold_limb_rotation_angle': {
                    'tightness': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
                    'asymmetry': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
                    'fold_wavelength': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
                    'axial_trace_1': {'mu':10, 'sigma':10},
                    'axial_traces_2': {'mu':10, 'sigma':10},
                    'axial_traces_3': {'mu':10, 'sigma':10},
                    'axial_traces_4': {'mu':10, 'sigma':10},
                },
                'fold_axis_rotation_angle': {
                    'hinge_angle': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
                    'fold_axis_wavelength': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
                },
                'fold_axial_surface': {
                    'axial_surface': {'lb':10, 'ub':10, 'mu':10, 'kappa':10, 'w':10}
                }
            }
                lb and ub are the lower and the upper bounds of the constraints and are used only for a restricted
                optimisation mode.
                #TODO Add initialisation check for dictionary

        """
        # Initialize the x values, constraints
        SPlotProcessor.__init__(self)
        self.x = None
        self.constraints = constraints

        # Define the constraint names
        self.constraint_names = ['asymmetry', 'tightness', 'fold_wavelength',
                                 'axial_trace', 'hinge_angle',
                                 'fold_axis_wavelength', 'axial_surface']

        # Initialize the constraint function map
        self.objective_functions_map = None

        # Define the intercept function and splot function
        self.intercept_function = fourier_series_x_intercepts
        self.splot_function = fourier_series

    def axial_surface_objective_function(self, v: np.ndarray) -> float:
        """
        Objective function for the axial surface.
        This function calculates the loglikelihood of an axial surface using the VonMisesFisher distribution.

        Parameters
        ----------
        v : np.ndarray
            the unit vector that represents the axial surface.
            This is evaluated every iteration in the optimisation process

        Returns
        -------
        float
            The logpdf value from the VonMisesFisher distribution.

         Raises
        ------
        KeyError
        If the required keys are not found in the constraints dictionary.

        """
        # Check if the required keys exist in the constraints dictionary
        # if 'axial_surface' not in self.constraints:
        #     raise KeyError("'axial_surface' key not found in constraints dictionary.")
        # if 'mu' not in self.constraints['axial_surface']:
        #     raise KeyError("'mu' key not found in 'axial_surface' dictionary.")
        # if 'kappa' not in self.constraints['axial_surface']:
        #     raise KeyError("'kappa' key not found in 'axial_surface' dictionary.")

        # Extract parameters for the VonMisesFisher distribution
        mu = self.constraints['axial_surface']['mu']
        kappa = self.constraints['axial_surface']['kappa']
        w = self.constraints['axial_surface']['w']

        # Create a VonMisesFisher distribution with the given parameters
        vmf = VonMisesFisher(mu, kappa)

        # Calculate the logpdf of the input array
        vmf_logpdf = vmf.logpdf(v) * w

        return vmf_logpdf

    def axial_trace_objective_function(self, theta: np.ndarray) -> Union[int, float]:
        """
        Calculate the objective function for the 'axial_trace' constraint.

        This function calculates the negative likelihood of the axial trace(s) given the provided knowledge constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to calculate
            the axial traces.

        Returns
        -------
        Union[int, float]
            The calculated objective function value. This is the likelihood of the axial trace(s).
            If there are no axial traces, the function returns a predefined constant value (999) that
             penalise the minimisation algorithm.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Calculate the intercepts using the provided theta values and the x values of the class
        intercepts = self.intercept_function(self.x, theta)
        if len(intercepts) == 0:
            # If there are no intercepts, return the predefined constant value
            return 999
        else:
            # Initialize the likelihood to 0
            likelihood = 0
            # Iterate over the knowledge constraints dictionary that have "axial_trace" in their key
            for key, trace in filter(lambda item: "axial_trace" in item[0], self.constraints.items()):
                # Get the mu, sigma, and weight values of a given axial trace
                # These values represent the mean and standard deviation of the location of a given axial trace
                mu = trace['mu']
                sigma = trace['sigma']
                w = trace['w']
                # Calculate the distance between mu and the axial trace
                dist = mu - intercepts
                # Update the likelihood using the Gaussian log likelihood function
                likelihood += -gaussian_log_likelihood(intercepts[np.argmin(dist)], mu, sigma) * w

        return likelihood

    def wavelength_objective_function(self, theta: np.ndarray) -> float:
        """
        Calculate the objective function for the fold wavelength constraint.

        This function calculates the negative likelihood of the fold wavelength given the knowledge constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold limb rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold wavelength.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu and sigma values from the constraints dictionary
        # These values represent the mean and standard deviation of the fold wavelength
        mu = self.constraints['fold_wavelength']['mu']
        sigma = self.constraints['fold_wavelength']['sigma']

        # Calculate the likelihood of the fold wavelength
        # The likelihood is calculated using the Gaussian log likelihood function
        likelihood = -gaussian_log_likelihood(theta[3], mu, sigma)

        # Get the weight of the constraint
        # The weight is used to adjust the influence of this constraint on the overall objective function
        w = self.constraints['fold_wavelength']['w']

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= w

        return likelihood

    def fold_axis_wavelength_objective_function(self, theta: np.ndarray) -> float:
        """
        Calculate the objective function for the fold axis wavelength constraint.

        This function calculates the negative likelihood of the fold axis wavelength given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold axis rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold axis wavelength.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu and sigma values from the constraints dictionary
        # These values represent the mean and standard deviation of the fold axis wavelength
        mu = self.constraints['axis_wavelength']['mu']
        sigma = self.constraints['axis_wavelength']['sigma']

        # Calculate the likelihood of the fold axis rotation angle wavelength
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -gaussian_log_likelihood(theta[3], mu, sigma)

        # Get the weight of the constraint
        # The weight is used to adjust the influence of this constraint on the overall objective function
        w = self.constraints['axis_wavelength']['w']

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= w

        return likelihood

    def tightness_objective_function(self, theta: np.ndarray) -> float:
        """
        Calculate the objective function for the 'tightness' constraint.

        This function calculates the likelihood of the fold tightness given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold limb rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold tightness.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu, sigma, and weight values from the constraints dictionary
        # These values represent the mean, standard deviation, and weight of the fold tightness
        mu = self.constraints['tightness']['mu']
        sigma = self.constraints['tightness']['sigma']
        w = self.constraints['tightness']['w']

        # Calculate the tightness of the fold
        tightness = self.calculate_tightness(theta)

        # Calculate the likelihood of the fold tightness
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -gaussian_log_likelihood(tightness, mu, sigma)

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= w

        return likelihood

    def hinge_angle_objective_function(self, theta: np.ndarray) -> float:
        """
        Calculate the objective function for the 'hinge_angle' constraint.

        This function calculates the likelihood of the fold hinge angle given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used
            to calculate the fold axis rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold hinge angle.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu, sigma, and weight values from the constraints dictionary
        # These values represent the mean, standard deviation, and weight of the fold hinge angle
        mu = self.constraints['hinge_angle']['mu']
        sigma = self.constraints['hinge_angle']['sigma']
        w = self.constraints['hinge_angle']['w']

        # Calculate the hinge angle of the fold
        hinge_angle = self.calculate_tightness(theta)

        # Calculate the likelihood of the fold hinge angle
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -gaussian_log_likelihood(hinge_angle, mu, sigma)

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= w

        return likelihood

    def asymmetry_objective_function(self, theta: np.ndarray) -> float:
        """
        Calculate the objective function for the 'asymmetry' constraint.

        This function calculates the likelihood of the fold asymmetry degree given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold limb rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold asymmetry degree.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu, sigma, and weight values from the constraints dictionary
        # These values represent the mean, standard deviation, and weight of the fold asymmetry degree
        mu = self.constraints['asymmetry']['mu']
        sigma = self.constraints['asymmetry']['sigma']
        w = self.constraints['asymmetry']['w']

        # Calculate the asymmetry of the fold
        asymmetry = self.calculate_asymmetry(theta)

        # Calculate the likelihood of the fold asymmetry
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -gaussian_log_likelihood(asymmetry, mu, sigma)

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= w

        return likelihood

    def setup_objective_functions(self):
        """
        Setup the mapping between constraint names and their corresponding objective function methods.

        This function creates a dictionary where the keys are the names of the constraints and the values are the
        corresponding objective function methods. This mapping is used to dynamically call the correct objective
        function based on the constraint name.
        """
        # Create a dictionary to map the constraint names to their corresponding objective function methods
        self.objective_functions_map = {

            'asymmetry': self.asymmetry_objective_function,

            'tightness': self.tightness_objective_function,

            'fold_wavelength': self.wavelength_objective_function,

            'fold_axis_wavelength': self.fold_axis_wavelength_objective_function,

            'axial_trace': self.axial_trace_objective_function,

            'hinge_angle': self.hinge_angle_objective_function,

            'axial_surface': self.axial_surface_objective_function
        }

    def __call__(self, theta: np.ndarray) -> float:
        """
        Calculate the total geological knowledge objective function value for all constraints by summing up the
        objective function values for all constraints. This objective function represent only the
        knowledge constraints and it is minimised with the main objective function that calculates the residuals.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold limb and axis rotation angle curve.

        Returns
        -------
        float
            The total objective function value.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        # self.check_fourier_parameters(theta)

        # Setup the constraint objective functions
        self.setup_objective_functions()

        # Initialize the total objective function value to 0
        total_objective_value = 0
        # Iterate over all constraint names
        for key in self.constraint_names:
            # If the constraint is in the constraints dictionary
            if key in self.constraints:
                # Add the objective function value for this constraint to the total
                total_objective_value += self.objective_functions_map[key](theta)
            else:
                # If the constraint is not in the constraints dictionary, do nothing
                pass

        # Return the total objective function value
        return total_objective_value

    def setup_objective_functions_for_restricted_mode(self) -> List[NonlinearConstraint]:
        """

        This function prepares the constraints by calculating the lower and upper bounds for each constraint and
        setting up the NonlinearConstraint objects from scipy.optimize. It also sets up the constraint functions.

        This function is used only when the optimisation is in a restricted mode. The restricted mode means that the
        optimisation algorithm cannot leave the parameter space defined by the geological knowledge constraints.
        The drawback of fitting a fold rotation angle model in this mode is that if the constraints provided are not
        representative of the studied fold geometry, the fitted model will be as well not representative.

        Returns
        -------
        List[NonlinearConstraint]
            A list of NonlinearConstraint objects for each constraint.

        Raises
        ------
        TypeError
            If the constraint info is not a dictionary.
        """
        # Setup the constraint functions
        self.setup_objective_functions()

        # Initialize the list of constraints
        constraints = []
        # Iterate over all constraints
        for constraint_name, constraint_info in self.constraints.items():
            # Check if constraint_info is a dictionary
            if not isinstance(constraint_info, dict):
                raise TypeError("`constraint_info` should be a dictionary.")

            # Get the lower and upper bounds, mu, and sigma values from the constraint info
            lb = constraint_info['lb']
            ub = constraint_info['ub']
            mu = constraint_info['mu']
            sigma = constraint_info['sigma']

            # Calculate the negative Gaussian log likelihood for a range of values between the lower and upper bounds
            val = -gaussian_log_likelihood(np.linspace(lb, ub, 100), mu, sigma)
            # Create a NonlinearConstraint object for this constraint
            nlc = NonlinearConstraint(self.objective_functions_map[constraint_name],
                                      val.min(), val.max(),
                                      jac='2-point', hess=BFGS())
            # Add the NonlinearConstraint object to the list of constraints
            constraints.append(nlc)

        # Return the list of constraints
        return constraints
