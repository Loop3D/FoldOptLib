from typing import Tuple, Callable, Union, Any, Optional, Dict

# from modified_loopstructural.extra_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
import numpy as np
# from LoopStructural.modelling.features.fold import fourier_series
# from uncertainty_quantification.fold_uncertainty import *
# from _helper import *
# from scipy.optimize import minimize, differential_evolution

from ..objective_functions.geological_knowledge import GeologicalKnowledgeFunctions
from .fold_optimiser import FoldOptimiser
from ..objective_functions.gaussian import loglikelihood_fourier_series
from ..helper.utils import *


# from ..helper._helper import *

#
# def scale(data):
#     mms = MinMaxScaler()
#     mms.fit(data)
#     data_transformed = mms.transform(data)
#
#     return data_transformed


class FourierSeriesOptimiser(FoldOptimiser):
    """
    A class used to represent a Fourier Series Optimiser.

    ...

    Attributes
    ----------
    fold_frame_coordinate : float
        The fold frame coordinate for the optimiser.
    rotation_angle : float
        The rotation angle for the optimiser.
    knowledge_constraints : dict, optional
        The knowledge constraints for the optimiser.
    x : float
        The interpolated fold frame coordinate z or y: np.linspace(z.min(), z.max(), 100).
        It's used to calculate the optimised Fourier series everywhere in the model space.
    kwargs : dict
        Additional keyword arguments.

    Methods
    -------
    TODO: Add methods here.
    """

    def __init__(self, fold_frame_coordinate: Union[list, np.ndarray], rotation_angle: Union[list, np.ndarray],
                 x: Union[list, np.ndarray],
                 # geological_knowledge: Optional[Dict[str, Any]] = None,
                 method='differential_evolution',
                 **kwargs: Dict[str, Any]):
        """
        Constructs all the necessary attributes for the Fourier Series Optimiser object.

        Parameters
        ----------
            fold_frame_coordinate : float
                The fold frame coordinate for the optimiser.
            rotation_angle : float
                The rotation angle for the optimiser.
            geological_knowledge : dict, optional
                The knowledge constraints for the optimiser.
            x : np.ndarray
                The interpolated fold frame coordinate z or y: np.linspace(z.min(), z.max(), 100).
                It's used to calculate the optimised Fourier series everywhere in the model space.
            **kwargs : dict
                Additional keyword arguments.
        """
        FoldOptimiser.__init__(self, method=method, **kwargs)
        self.objective_value = 0
        self.fold_frame_coordinate = fold_frame_coordinate
        self.rotation_angle = np.tan(np.deg2rad(rotation_angle))
        # TODO: Add a check if the knowledge constraints are in the correct format
        self.method = method
        # TODO: check how to initialise self.x = x in self.geological_knowledge
        self.x = x
        self.kwargs = kwargs

    # def prepare_and_setup_knowledge_constraints(self, geological_knowledge: Optional[Dict[str, Any]] = None) -> \
    #         Optional[Union[GeologicalKnowledgeFunctions, None]]:
    #     """
    #     Prepare the geological knowledge constraints and objective functions.
    #
    #     Returns
    #     -------
    #     GeologicalKnowledgeFunctions or None
    #         Returns the geological knowledge constraints and objective functions if they exist,
    #         otherwise returns None.
    #     """
    #
    #     # Check if knowledge constraints exist
    #     if geological_knowledge is not None:
    #         _geological_knowledge = super().prepare_and_setup_knowledge_constraints(geological_knowledge)
    #         return _geological_knowledge
    #
    #     # If knowledge constraints do not exist, return None
    #     if geological_knowledge is None:
    #         return None

    def generate_initial_guess(self) -> Union[np.ndarray, Any]:
        """
        Generate an initial guess for the optimisation.
        It generates a guess of the wavelength for the Fourier series. The format of the guess depends
        on the method of optimisation - differential evolution or trust region. If it's differential evolution,
        it will generate the bounds for the optimisation. If it's trust region, it will generate the initial guess of
        the wavelength.

        Returns
        -------
        np.ndarray or Any
            Returns the initial guess or bounds for the optimisation.
        """

        # Check if method is specified in kwargs
        if self.method == 'differential_evolution':
            if 'wl_guess' in self.kwargs:
                wl = get_wavelength_guesses(self.kwargs['wl_guess'], 1000)
                # bounds = np.array([(-1, 1), (-1, 1), (-1, 1), (wl[wl > 0].min() / 2, wl.max())], dtype=object)
                bounds = np.array([(-1, 1), (-1, 1), (-1, 1), (wl[wl > 0].min() / 2, wl.max())], dtype=object)
                return bounds
            else:
                # Calculate semivariogram and get the wavelength guess
                guess, lags, variogram = calculate_semivariogram(self.fold_frame_coordinate, self.rotation_angle)
                wl = get_wavelength_guesses(guess[3], 1000)
                bounds = np.array([(-1, 1), (-1, 1), (-1, 1), (wl[wl > 0].min() / 2, wl.max()/3)], dtype=object)
                return bounds

        # Check if wl_guess is specified in kwargs
        if 'wl_guess' in self.kwargs:
            guess = np.array([0, 1, 1, self.kwargs['wl_guess']], dtype=object)
            return guess
        else:
            # Calculate semivariogram and get the wavelength guess
            guess, lags, variogram = calculate_semivariogram(self.fold_frame_coordinate, self.rotation_angle)

            return guess

    def setup_optimisation(self, geological_knowledge=None) -> Tuple[Callable,
                                                                     Union[
                                                                         GeologicalKnowledgeFunctions, None], Callable,
                                                                     Any]:
        """
        Setup Fourier series optimisation.

        Returns
        -------
        tuple
            Returns a tuple containing the objective function, geological knowledge, solver, and initial guess.
        """

        # # Check if method is specified in kwargs and assign the appropriate solver
        # if 'method' in self.kwargs and self.kwargs['method'] == 'differential_evolution':
        #     solver = self.optimise_with_differential_evolution
        # else:
        #     solver = self.optimise_with_trust_region

        # Setup objective function
        objective_function = loglikelihood_fourier_series(self.rotation_angle, self.fold_frame_coordinate)

        # Prepare and setup knowledge constraints
        geological_knowledge, solver = super().setup_optimisation(geological_knowledge=geological_knowledge)

        # Generate initial guess
        guess = self.generate_initial_guess()

        return objective_function, geological_knowledge, solver, guess

    def optimise(self, geological_knowledge=None, **kwargs) -> Dict[str, Any]:
        """
        Optimise the Fourier series.

        Returns
        -------
        Dict[str, Any]
            Returns the result of the optimisation.
        """

        # Setup optimisation
        objective_function, geological_knowledge, solver, guess = self.setup_optimisation(
            geological_knowledge=geological_knowledge)
        if self.method == 'differential_evolution':

            # Check if geological knowledge exists
            if geological_knowledge is not None:
                # Check if mode is restricted
                if 'mode' in self.kwargs and self.kwargs['mode'] == 'restricted':
                    opt = solver(objective_function, x0=guess, constraints=geological_knowledge)

                    return opt
                else:
                    # Wrap to produce an objective function that
                    # takes into account the geological knowledge functions
                    objective_function = objective_wrapper(objective_function, geological_knowledge)
                    # bounds = np.array([(-1, 1), (-1, 1), (-1, 1), (guess[3] / 2, guess[3] * 2)], dtype=object)
                    opt = solver(objective_function, bounds=guess)

                    return opt
            else:
                opt = solver(objective_function, bounds=guess)

                return opt

        else:
            opt = solver(objective_function, guess)

        return opt
