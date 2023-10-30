from typing import Optional, Dict, Any, Union, Callable, Dict, Any, Tuple
# import pandas as pd
import numpy as np
# from ..input.input_data_checker import CheckInputData
# from ..helper._helper import *
from ..helper.utils import *
# from .base_optimiser import BaseOptimiser
from abc import ABC, abstractmethod
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint

from ..objective_functions.geological_knowledge import GeologicalKnowledgeFunctions


class FoldOptimiser(ABC):
    """
    Base class for fold geometry optimisation.
    """

    def __init__(self, method='differential_evolution', **kwargs: Dict[str, Any]):
        """
        Constructs all the necessary attributes for the Fold Optimiser object.

        Parameters
        ----------
            kwargs : dict
                Additional keyword arguments.
        """
        self.method = method
        self.kwargs = kwargs

    def prepare_and_setup_knowledge_constraints(self, geological_knowledge=None) -> \
            Union[list[NonlinearConstraint], GeologicalKnowledgeFunctions, None]:
        """
        Prepare the knowledge constraints data
        """
        # Check if knowledge constraints exist
        if geological_knowledge is not None:
            # TODO: Add a check if the knowledge constraints are in the correct format
            # Check if mode is restricted
            # TODO: Update to use only restricted_mode as kwarg that takes a boolean True or False value
            if 'mode' in self.kwargs and self.kwargs['mode'] == 'restricted':
                geological_knowledge = GeologicalKnowledgeFunctions(geological_knowledge)
                ready_constraints = geological_knowledge.setup_objective_functions_for_restricted_mode()

                return ready_constraints
            else:
                geological_knowledge = GeologicalKnowledgeFunctions(geological_knowledge)

                return geological_knowledge
        if not geological_knowledge:
            # If knowledge constraints do not exist, return None
            if geological_knowledge is None:
                return None


        # If knowledge constraints do not exist, return None

    @abstractmethod
    def generate_initial_guess(self):
        """
        Generate an initial guess for the optimisation
        It generates a guess depending on the type of optimisation, if it's fourier series
        it will generate a guess of the wavelength, if it's axial surface it will generate a guess
        using the methods of the Differential Evolution algorithm (Stern and Price, 1997) or uses the
        Von Mises Fisher distribution (Fisher, 1953).
        """

        pass

    def optimise_with_trust_region(self, objective_function: Callable,
                                   x0: np.ndarray, constraints=None, **kwargs) -> Dict:
        """
        Solves the optimization problem using the trust region method.

        Parameters
        ----------
            x0 : np.ndarray
                Initial guess of the parameters to be optimised.

        Returns
        -------
            opt : Dict
                The solution of the optimisation.

        """

        opt = minimize(objective_function, x0,
                       method='trust-constr', jac='2-point',
                       constraints=constraints, **kwargs)

        return opt

    def optimise_with_differential_evolution(self, objective_function: Callable, bounds: Tuple, init: str = 'halton',
                                             maxiter: int = 5000, seed: int = 80,
                                             polish: bool = False, strategy: str = 'best2exp',
                                             mutation: Tuple[float, float] = (0.3, 0.99), **kwargs) -> Dict:
        """
        Solves the optimization problem using the differential evolution method.
        Check Scipy documentation for more info

        Parameters
        ----------
            bounds : Tuple
                Bounds for variables. ``(min, max)`` pairs for each element in ``x``,
                defining the bounds on that parameter.
            init : str
                Specify how population initialisation is performed. Default is 'halton'.
            maxiter : int
                The maximum number of generations over which the entire population is evolved. Default is 5000.
            seed : int
                The seed for the pseudo-random number generator. Default is 80.
            polish : bool
                If True (default), then differential evolution is followed by a polishing phase.
            strategy : str
                The differential evolution strategy to use. Default is 'best2exp'.
            mutation : Tuple[float, float]
                The mutation constant. Default is (0.3, 0.99) and it was tested and have proven to explore the parameter
                space efficiently.

        Returns
        -------
            opt : Dict
                The solution of the optimization.
                :param objective_function:
        """

        opt = differential_evolution(objective_function, bounds=bounds, init=init,
                                     maxiter=maxiter, seed=seed, polish=polish,
                                     strategy=strategy, mutation=mutation, **kwargs)

        return opt

    @abstractmethod
    def setup_optimisation(self, geological_knowledge=None) -> tuple:
        """
        Setup optimisation.

        Returns
        -------
        tuple
            Returns a tuple containing the geological knowledge objective functions, a solver, and the initial guess.
        """

        # Check if method is specified in kwargs and assign the appropriate solver
        if self.method == 'differential_evolution':
            solver = self.optimise_with_differential_evolution
        else:
            solver = self.optimise_with_trust_region

        # Prepare and setup knowledge constraints
        geological_knowledge = self.prepare_and_setup_knowledge_constraints(geological_knowledge=geological_knowledge)

        return geological_knowledge, solver

    @abstractmethod
    def optimise(self, *args, **kwargs):
        """
        Run the optimisation
        """

        pass
