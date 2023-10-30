from abc import ABC
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
# import mplstereonet
# from sklearn.preprocessing import StandardScaler
# from scipy.optimize import minimize, differential_evolution
# from scipy.stats import vonmises_fisher, vonmises
from .fold_optimiser import FoldOptimiser
from ..objective_functions import GeologicalKnowledgeFunctions
from ..input import CheckInputData
from ..helper.utils import strike_dip_to_vector, normal_vector_to_strike_and_dip, get_predicted_rotation_angle
from ..objective_functions import VonMisesFisher
from ..objective_functions import is_axial_plane_compatible
from ..fold_modelling import FoldModel
from ..objective_functions import loglikelihood_axial_surface


def calculate_intersection_lineation(axial_surface, folded_foliation):
    """
    Calculate the intersection lineation of the axial surface and the folded foliation.

    Parameters:
    axial_surface (np.ndarray): The normal vector of the axial surface.
    folded_foliation (np.ndarray): The normal vector of the folded foliation.

    Returns:
    np.ndarray: The normalised intersection lineation vector.
    """
    # Check if the inputs are numpy arrays
    if not isinstance(axial_surface, np.ndarray):
        raise TypeError("Axial surface vector must be a numpy array.")
    if not isinstance(folded_foliation, np.ndarray):
        raise TypeError("Folded foliation vector must be a numpy array.")

    # Check if the inputs have the same shape
    if axial_surface.shape != folded_foliation.shape:
        raise ValueError("Axial surface and folded foliation arrays must have the same shape.")

    # Calculate cross product of the axial surface and folded foliation normal vectors
    li = np.cross(axial_surface, folded_foliation)

    # Normalise the intersection lineation vector
    li /= np.linalg.norm(li, axis=1)[:, None]

    return li


# def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
#     return -(value - mu)**2


class AxialSurfaceOptimiser(FoldOptimiser):
    """
        Optimiser for Axial Surfaces.

        This class inherits from FoldOptimiser, FoldModel. It is used to optimise the axial surfaces
        based on the provided data, bounding box, geological knowledge.

    """

    def __init__(self, data: pd.DataFrame,
                 bounding_box: Union[list, np.ndarray],
                 geological_knowledge: Optional[Dict[str, Any]] = None,
                 **kwargs: Dict[str, Any]):

        """
                Initialise the AxialSurfaceOptimiser with data, bounding box, geological knowledge and other parameters.

                Parameters
                ----------
                data : pd.DataFrame
                    The input data for optimisation.
                bounding_box : Union[list, np.ndarray]
                    The bounding box for the optimisation.
                geological_knowledge : Optional[Dict[str, Any]], optional
                    The geological knowledge used for optimisation, by default None.
                **kwargs : Dict[str, Any]
                    Other optional parameters for optimisation. Can include scipy optimisation parameters for
                    differential evolution and trust-constr methods.
                    mode : str, optional, optimisation mode, can be 'restricted' or 'unrestricted',
                    by default 'unrestricted'. only unrestricted mode is supported for now.
                    method : str, optional, optimisation method, can be 'differential_evolution' or 'trust-region',
                    by default 'differential_evolution'.

                """

        # Check the input data
        check_input = CheckInputData(data, bounding_box,
                                     geological_knowledge=geological_knowledge)
        check_input.check_input_data()

        FoldOptimiser.__init__(self, **kwargs)
        self.fold_engine = FoldModel(data, bounding_box,
                                     geological_knowledge=geological_knowledge,
                                     **kwargs)

        self.data = data
        self.bounding_box = bounding_box
        self.geological_knowledge = geological_knowledge
        self.kwargs = kwargs
        self.gradient_data = self.data[['gx', 'gy', 'gz']].to_numpy()
        self.geo_objective = None
        self.objective_function = None
        self.guess = None
        self.solver = None

    def generate_initial_guess(self):
        """
        Generate the initial guess for the optimisation for differential evolution algorithm.
        The initial guess is generated using the Von Mises Fisher distribution.

        """
        if 'axial_surface_guess' in self.kwargs:
            guess = self.kwargs['axial_surface_guess']
            if len(guess) == 2:
                # Create a VonMisesFisher distribution with the given parameters
                mu = strike_dip_to_vector(*guess)
                kappa = 5
                vmf = VonMisesFisher(mu, kappa)
                # Sample from the distribution
                initial_guess = vmf.draw_samples(size=20, random_state=180)
                initial_guess = normal_vector_to_strike_and_dip(initial_guess)
                return initial_guess

            if len(guess) == 3:
                mu = guess
                # normalise guess
                mu /= np.linalg.norm(mu)
                kappa = 5
                vmf = VonMisesFisher(mu, kappa)
                # Sample from the distribution
                initial_guess = vmf.draw_samples(size=20, random_state=180)
                initial_guess = normal_vector_to_strike_and_dip(initial_guess)
                return initial_guess
            else:
                raise ValueError("'axial_surface_guess' should be a list or a np.array "
                                 "of the form [strike, dip] or a 3D unit vector")

        if 'axial_surface_guess' not in self.kwargs:
            # use the halton method to initialise the optimisation
            return 'halton'

    def loglikelihood(self, x, predicted_foliation: np.ndarray,
                      geological_knowledge: GeologicalKnowledgeFunctions) -> float:
        """
         Calculate the maximum likelihood estimate of the axial surface and the geological knowledge.

         Parameters
         ----------
         x : np.ndarray
             The axial surface normal vector to be optimised.
         predicted_foliation : np.ndarray
             The predicted foliation data.
         geological_knowledge : GeologicalKnowledgeFunctions
             The geological knowledge functions.

         Returns
         -------
         float
             The calculated loglikelihood of the axial surface. Returns None if input is not valid.
         """

        # Calculate the angle difference between the predicted and observed foliation
        angle_difference = is_axial_plane_compatible(predicted_foliation, self.gradient_data)
        # Calculate the loglikelihood of the axial surface
        loglikelihood = loglikelihood_axial_surface(angle_difference) + geological_knowledge(x)

        return loglikelihood

    def mle_optimisation(self, strike_dip: Tuple[float, float]):
        """
        Performs Maximum Likelihood Estimation (MLE) optimisation.

        Parameters
        ----------
        strike_dip : tuple
            A tuple containing strike and dip values of the estimated axial surface.

        Returns
        -------
        logpdf : float
            The log-likelihood of the MLE.

        Notes
        -----
        This function performs MLE optimisation and used when geological knowledge constraints are provided.
        The function returns the log-likelihood of the MLE that is optimised
        """

        axial_normal = strike_dip_to_vector(*strike_dip)
        axial_normal /= np.linalg.norm(axial_normal)

        predicted_foliation = self.fold_engine.get_predicted_foliation(axial_normal)
        logpdf = self.loglikelihood(axial_normal, predicted_foliation, self.geo_objective)
        return logpdf

    def angle_optimisation(self, strike_dip: Tuple[float, float]):
        """
            Minimises the angle between the observed and predicted folded foliation.

            Parameters
            ----------
            strike_dip : tuple
                A tuple containing strike and dip values of the estimated axial surface.

            Returns
            -------
            angle_difference : float
                The difference between the predicted and actual angle.

            Notes
            -----
            This function optimises the axial surface by minimising the angle between the predicted and observed folded
            foliation. This function is used when there are no geological knowledge constraints provided.
        """

        axial_normal = strike_dip_to_vector(strike_dip[0], strike_dip[1])
        axial_normal /= np.linalg.norm(axial_normal)
        predicted_foliation = self.fold_engine.get_predicted_foliation(axial_normal)
        angle_difference = is_axial_plane_compatible(predicted_foliation, self.gradient_data)
        return angle_difference

    def setup_optimisation(self, geological_knowledge: Optional[Dict[str, Any]] = None):
        """
           Sets up the optimisation algorithm.

           Parameters
           ----------
           geological_knowledge : dict, optional
               A dictionary containing geological knowledge. Default is None.

           Returns
           -------
           objective_function : callable
               The objective function to be minimised.
           _geological_knowledge : dict or None
               The geological knowledge objective functions.
           solver : BaseOptimiser
               The solver from BaseOptimiser to be used for optimisation.
           guess : Union[np.ndarray, str]
               The initial guess for the optimisation.
        """
        # TODO - check format of the geological knowledge dictionary
        _geological_knowledge, solver = super().setup_optimisation(geological_knowledge)
        # guess = self.generate_initial_guess()

        # Generate initial guess
        guess = self.generate_initial_guess()
        # Setup objective function
        if _geological_knowledge is not None:
            # if _geological_knowledge exists then use the negative logpdf of the Von Mises distribution
            # as the objective function to minimise
            objective_function = self.mle_optimisation

        # if no geological knowledge is provided then use the angle difference between the predicted and observed
        # foliation as the objective function to minimise
        else:
            objective_function = self.angle_optimisation

        return objective_function, _geological_knowledge, solver, guess

    def optimise(self, geological_knowledge: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs the optimisation.

        Parameters
        ----------
        geological_knowledge : dict, optional
            A dictionary containing geological knowledge. Default is None.

        Returns
        -------
        opt : Dict
            The result of the optimisation.

        Notes
        -----
        This function runs the optimisation by setting up the optimisation problem,
        checking if geological knowledge exists, and running the solver.
        """

        bounds = [(0, 360), (0, 90)]
        if geological_knowledge is None:
            self.objective_function, self.geo_objective, self.solver, self.guess = \
                self.setup_optimisation()
            if self.method == 'differential_evolution':
                opt = self.solver(self.objective_function, bounds, init=self.guess)

            else:
                opt = self.solver(self.objective_function, bounds, init=self.guess)

            return opt

        # Check if geological knowledge exists
        if geological_knowledge is not None:
            # Setup optimisation
            if len(geological_knowledge['fold_axial_surface']) == 0:
                self.objective_function, self.geo_objective, self.solver, self.guess = \
                    self.setup_optimisation(geological_knowledge=geological_knowledge)
                if self.method == 'differential_evolution':
                    if 'axial_surface_guess' in self.kwargs:
                        opt = self.solver(self.objective_function, bounds, init=self.guess)

                        return opt

                else:
                    opt = self.solver(self.objective_function, x0=self.guess)

                return opt

            if len(geological_knowledge['fold_axial_surface']) != 0:
                self.objective_function, self.geo_objective, self.solver, self.guess = \
                    self.setup_optimisation(geological_knowledge['fold_axial_surface'])

                if self.method == 'differential_evolution':
                    opt = self.solver(self.objective_function, bounds, init=self.guess)

                else:
                    opt = self.solver(self.objective_function, x0=self.guess)

                return opt

            # TODO: add support for restricted optimisation mode
            # Check if mode is restricted
            # if 'mode' in self.kwargs and self.kwargs['mode'] == 'restricted':
            #     opt = self.solver(self.objective_function, x0=self.guess, constraints=self.geo_objective)
            #     return opt
            # else:

            # return opt
