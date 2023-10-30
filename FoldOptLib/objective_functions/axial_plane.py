import numpy as np


def is_axial_plane_compatible(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle difference between the predicted bedding and the observed one.

    This is an objective function that verifies if the predicted and observed bedding or folded foliation are
    geometrically compatible. If the sum of the calculated angles is close to 0 degrees, the axial plane of the
    predicted folded foliation is representative of the axial surface of observed folded foliation.

    Parameters
    ----------
    v1 : np.ndarray
        The first vector representing the predicted bedding.
    v2 : np.ndarray
        The second vector representing the observed bedding.

    Returns
    -------
    np.ndarray
        The angle difference between the predicted and observed bedding.

    Raises
    ------
    ValueError
        If `v1` and `v2` are not numpy arrays of the same shape.
    """
    # Check if `v1` and `v2` are numpy arrays of the same shape
    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        raise ValueError("`v1` and `v2` should be numpy arrays.")
    if v1.shape != v2.shape:
        raise ValueError("`v1` and `v2` should have the same shape.")

    # Calculate the dot product of `v1` and `v2`
    dot_product = np.einsum("ij,ij->i", v1, v2)
    # Calculate the angle between v1 and v2
    angle_difference = np.arccos(dot_product)

    objective_func = angle_difference.sum()

    return objective_func
