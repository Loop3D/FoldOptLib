import numpy as np
from typing import Callable, Dict, Any


class LeastSquaresFunctions:
    """
    A class used to represent the Least Squares Functions.

    ...

    Attributes
    ----------
    rotation_angle : np.ndarray
        Either the calculated fold limb or axis rotation angles.
    fold_frame : np.ndarray
        The fold frame z or y.
    knowledge_constraints : Callable
        A callable function that calculate the likelihood of input knowledge constraints.
    kwargs : Dict[str, Any]
        Additional keyword arguments.

    Methods
    -------
    square_residuals(theta: float) -> np.ndarray:
        Calculates the square residuals between observations y and predicted rotation angle _y.
    huber_loss(residuals: np.ndarray, delta: float = 0.5) -> np.ndarray:
        Calculates the Huber loss.
    soft_l1_loss(residuals: np.ndarray, delta: float = 0.5) -> np.ndarray:
        Calculates the Soft L1 loss.
    cost(theta: float) -> float:
        Calculates the cost function.
    """

    def __init__(self, rotation_angle: np.ndarray, fold_frame: np.ndarray,
                 knowledge_constraints: Callable, **kwargs: Dict[str, Any]) -> None:

        """
        Constructs all the necessary attributes for the Least Squares Functions object.

        Parameters
        ----------
        rotation_angle : float
            Either the calculated fold limb or axis rotation angles.
        fold_frame : np.ndarray
            The fold frame z or y.
        knowledge_constraints : Callable
            A callable function that calculate the likelihood of input knowledge constraints.
        kwargs : Dict[str, Any]
            Additional keyword arguments.
        """
        # handle errors for the input parameters here and raise errors
        # if the rotation angle is not a numpy array
        if not isinstance(rotation_angle, np.ndarray):
            raise TypeError("The rotation angle must be a numpy array.")
        if not isinstance(fold_frame, np.ndarray):
            raise TypeError("The fold frame must be a numpy array.")
        if not callable(knowledge_constraints):
            raise TypeError("The knowledge constraints must be a callable function.")

        self.rotation_angle = rotation_angle
        self.fold_frame = fold_frame
        self.knowledge_constraints = knowledge_constraints
        self.kwargs = kwargs

    def square_residuals(self, theta: np.ndarray) -> np.ndarray:

        """
        Calculates the square residuals between observations y and predicted rotation angle _y.

        Parameters
        ----------
            theta : float
                The theta value.

        Returns
        -------
            residuals : np.ndarray
                The calculated residuals.
        """
        # The calculated rotation angle
        y = np.tan(np.deg2rad(self.rotation_angle))

        # The predicted rotation angle
        _y = np.tan(np.deg2rad(self.fourier_series(self.fold_frame, *theta)))
        residuals = (y - _y) ** 2

        return residuals

    def huber_loss(self, residuals: np.ndarray, delta: float = 0.5) -> np.ndarray:

        """
        Calculates the Huber loss function.

        Parameters
        ----------
            residuals : np.ndarray
                The residuals.
            delta : float
                The delta value. Default is 0.5.

        Returns
        -------
            s : np.ndarray
                The calculated Huber loss.
        """
        s = np.zeros(len(residuals))

        for i, residual in enumerate(residuals):
            if abs(residual) <= delta:
                s[i] = 0.5 * residual ** 2
            else:
                s[i] = delta * abs(residual) - 0.5 * delta ** 2

        return s

    def soft_l1_loss(self, residuals: np.ndarray, delta: float = 0.5) -> np.ndarray:

        """
        Calculates the Soft L1 loss function.

        Parameters
        ----------
            residuals : np.ndarray
                The residuals.
            delta : float
                The delta value. Default is 0.5.

        Returns
        -------
            l1 : np.ndarray
                The calculated Soft L1 loss.
        """
        l1 = np.zeros(len(residuals))

        for i, residual in enumerate(residuals):
            l1[i] = 2 * delta ** 2 * (np.sqrt(1 + (residual / delta) ** 2) - 1)

        return l1

    def cost(self, theta: np.ndarray) -> float:
        """
        Calculates the cost function.

        Parameters
        ----------
            theta : float
                The theta value.

        Returns
        -------
            cost : float
                The calculated cost.
        """
        if self.knowledge_constraints is not None:
            return 0.5 * np.sum(
                self.huber_loss(self.square_residuals(theta))) + self.knowledge_constraints(theta)
        else:
            return 0.5 * np.sum(
                self.huber_loss(self.square_residuals(theta)))
