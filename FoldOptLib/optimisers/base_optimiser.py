# from typing import Callable, Dict, Any, Tuple
#
# import numpy as np
# from scipy.optimize import minimize, differential_evolution
# import functools
# from abc import ABC, abstractmethod
#
# # TODO: merge this class with FoldOptimiser
# class BaseOptimiser(ABC):
#     """
#     A base class that to represent an abstract Optimiser.
#
#     ...
#
#     Attributes
#     ----------
#     objective_function : Callable
#         a function which is to be minimised.
#     kwargs : Dict[str, Any]
#         additional keyword arguments.
#
#     Methods
#     -------
#     solve_with_trust_region()
#         Solves the optimisation problem using the trust region method.
#     solve_with_differential_evolution()
#         Solves the optimisation problem using the differential evolution method.
#     """
#
#
#
#     @abstractmethod
#     def optimise(self, *args, **kwargs) -> Any:
#         pass
