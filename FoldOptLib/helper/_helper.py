# from modified_loopstructural.extra_utils import *
import numpy as np
import joblib as jb
from LoopStructural.modelling.features.fold import fourier_series
# from uncertainty_quantification.fold_uncertainty import *
# import dill
import mplstereonet
import pandas as pd


# def gaussian_func(b, mu, sigma):
#     return 0.5 * np.exp(- (b - mu) ** 2 / (
#             2 * sigma ** 2))

def gaussian_func(b, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (b - mu) ** 2 / sigma ** 2


def fourier_series_2(x, theta):
    c0 = theta[0]
    T = np.array([theta[1:4], theta[4:]])
    v = 0
    for i in range(len(T)):
        t = np.concatenate([[c0], T[i]])
        v += fourier_series(x, *t)
        # v = v.astype(float)
    # v1 = c0 + c1 * np.cos(2 * np.pi / w * x) + c2 * np.sin(2 * np.pi / w * x)
    return np.rad2deg(np.arctan(v))


def parallel(function, array, jobs=1):
    results = jb.Parallel(n_jobs=jobs, verbose=1, prefer='threads')(jb.delayed(function)(i) for i in array)
    results = np.asarray(results, dtype='object')
    return results


def axial_plane_stereonet(strike, dip):
    """
    Calculate the axial plane in a stereonet given the strike and dip angles.
    credit: https://mplstereonet.readthedocs.io/en/latest/examples/axial_plane.html

    Parameters:
    strike (np.ndarray): The strike angles in degrees.
    dip (np.ndarray): The dip angles in degrees.

    Returns:
    tuple: The axial strike and dip angles in degrees.
    """
    # Check if the inputs are numpy arrays
    if not isinstance(strike, np.ndarray):
        raise TypeError(f"Expected strike to be a numpy array, got {type(strike).__name__}")
    if not isinstance(dip, np.ndarray):
        raise TypeError(f"Expected dip to be a numpy array, got {type(dip).__name__}")

    # Check if the inputs have the same shape
    if strike.shape != dip.shape:
        raise ValueError("Strike and dip arrays must have the same shape.")

    # Find the two modes
    centers = mplstereonet.kmeans(strike, dip, num=2, measurement='poles')

    # Fit a girdle to the two modes
    axis_s, axis_d = mplstereonet.fit_girdle(*zip(*centers), measurement='radians')

    # Find the midpoint
    mid, _ = mplstereonet.find_mean_vector(*zip(*centers), measurement='radians')
    midx, midy = mplstereonet.line(*mid)

    # Find the axial plane by fitting another girdle to the midpoint and the pole of the plunge axis
    xp, yp = mplstereonet.pole(axis_s, axis_d)
    x, y = [xp, midx], [yp, midy]
    axial_s, axial_dip = mplstereonet.fit_girdle(x, y, measurement='radians')

    return axial_s, axial_dip


def get_fold_curves(geological_feature, fold_frame=0):
    """
    Calculate the fold axis and limb rotation angle curves of a geological feature.

    Parameters:
    geological_feature: The geological feature to calculate the fold rotation angle curves for.
    fold_frame (optional): The fold frame coordinate to use, default 0 for fold limb rotation angle.
    for fold axis rotation angle use 1.
    If not provided, the function will use a default axis.

    Returns:
    tuple: The x values and the corresponding fold curve values.
    """
    # Check if the geological_feature has the required attributes
    if not hasattr(geological_feature, 'fold') or not hasattr(geological_feature.fold, 'foldframe') or not hasattr(
            geological_feature.fold, 'fold_axis_rotation') or not hasattr(geological_feature.fold,
                                                                          'fold_limb_rotation'):
        raise AttributeError(
            "Geological feature must have a 'fold' attribute with 'foldframe', "
            "'fold_axis_rotation', and 'fold_limb_rotation' attributes.")

    # Determine the axis to use for the calculation
    coordinate_to_use = fold_frame

    # Calculate the fold frame coordinate values x and the fold rotation angle curve
    x = np.linspace(geological_feature.fold.foldframe[coordinate_to_use].min(),
                    geological_feature.fold.foldframe[coordinate_to_use].max(), 200)
    curve = geological_feature.fold.fold_axis_rotation(
        x) if fold_frame is 1 else geological_feature.fold.fold_limb_rotation(x)

    return x, curve


def create_dict(x=None, y=None, z=None, strike=None, dip=None, feature_name=None,
                coord=None, data_type=None, **kwargs):
    if data_type == 'foliation':
        fn = np.empty(len(x)).astype(str)
        fn.fill(feature_name)
        c = np.empty((len(x))).astype(int)
        c.fill(coord)
        dictionary = {'X': x,
                      'Y': y,
                      'Z': z,
                      'strike': strike,
                      'dip': dip,
                      'feature_name': fn,
                      'coord': c}
        return dictionary

    if data_type == 'fold_axis':
        fn = np.empty(len(x)).astype(str)
        fn.fill(feature_name)
        c = np.empty((len(x))).astype(int)
        c.fill(coord)
        dictionary = {'X': x,
                      'Y': y,
                      'Z': z,
                      'plunge_dir': strike,
                      'plunge': dip,
                      'feature_name': fn,
                      'coord': c}
        return dictionary


def create_gradient_dict(x=None, y=None, z=None,
                         nx=None, ny=None, nz=None,
                         feature_name=None, coord=None,
                         **kwargs):
    fn = np.empty(len(x)).astype(str)
    fn.fill(feature_name)
    c = np.empty((len(x))).astype(int)
    c.fill(coord)
    dictionary = {'X': x,
                  'Y': y,
                  'Z': z,
                  'gx': nx,
                  'gy': ny,
                  'gz': nz,
                  'feature_name': fn,
                  'coord': c}
    return dictionary


def make_dataset(vec: np.ndarray, points: np.ndarray,
                 name: str = 's0', coord: int = 0) -> pd.DataFrame:
    """

    Make a dataset from one unit vector and xyz points of the folded feature data.

    Parameters
    ----------
    vec : np.ndarray
        The unit vector to be used as the gradient.
    points : np.ndarray
        The xyz coordinates of the data points.
    name : str, optional
        The name of the feature, by default 's0'.
    coord : int, optional
        The coordinate, by default 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a data point with its coordinates (X, Y, Z),
        gradient (gx, gy, gz), feature name, and coordinate.
    """
    g = np.tile(vec, (len(points), 1))
    dataset = pd.DataFrame()
    dataset['X'] = points[:, 0]
    dataset['Y'] = points[:, 1]
    dataset['Z'] = points[:, 2]
    dataset['gx'] = g[:, 0]
    dataset['gy'] = g[:, 1]
    dataset['gz'] = g[:, 2]
    dataset['feature_name'] = name
    dataset['coord'] = coord

    return dataset
