from typing import Union, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from LoopStructural import GeologicalModel
from ..helper._helper import *
from ..helper.utils import *
# from
from ..input.input_data_processor import InputDataProcessor
from ..from_loopstructural._fold import FoldEvent
from ..from_loopstructural._fold_frame import FoldFrame
# from from_loopstructural._svariogram import SVariogram
from .base_fold_frame_builder import BaseFoldFrameBuilder
from ..optimisers.fourier_optimiser import FourierSeriesOptimiser
import gc

def fold_function(params):
    def rot_func(x):
        return np.rad2deg(
            np.arctan(fourier_series(x, *params)))

    return rot_func


class FoldModel(BaseFoldFrameBuilder):
    """
    A class used to build a fold model.

    ...

    Attributes
    ----------
    data : pd.DataFrame
        processed data
    bounding_box : list
        bounding box for the model
    model : Any
        the GeologicalModel object to be built, initialised as None
    gradient_data : np.ndarray
        gradient data extracted from the data DataFrame
    points : np.ndarray
        coordinates of the data points
    kwargs : dict
        additional keyword arguments
    axial_surface : Any
        axial surface of the model, initialised as None
    scaled_points : Any
        scaled from UTM to the model scale, initialised as None

    Methods
    -------
    No methods defined yet.
    """

    def __init__(self, data: pd.DataFrame,
                 bounding_box: Union[list, np.ndarray],
                 geological_knowledge: Optional[Dict[str, Any]] = None,
                 **kwargs: Dict[str, Any]):
        """
        Constructs all the necessary attributes for the FoldModel object.

        Parameters
        ----------
            data : pd.DataFrame
                structural data
            bounding_box : list or np.ndarray
                bounding box for the model
            **kwargs : dict
                additional keyword arguments
        """
        data_processor = InputDataProcessor(data, bounding_box,
                                            geological_knowledge=geological_knowledge)
        self.data = data_processor.process_data()
        self.bounding_box = bounding_box
        self.model = None
        self.gradient_data = self.data[['gx', 'gy', 'gz']].to_numpy()
        self.points = self.data[['X', 'Y', 'Z']].to_numpy()  # coordinates of the data points
        assert len(self.points) == len(self.gradient_data), "coordinates must have the same length as data"
        self.kwargs = kwargs
        self.axial_surface = None
        self.scaled_points = None
        self.geological_knowledge = geological_knowledge

    def initialise_model(self) -> None:
        """
        Initialises the geological model and scales the points (xyz).

        The GeologicalModel class is initialised with the bounding box.
        The xyz points of the FoldModel object are then scaled using the geological model.

        Returns
        -------
        None
        """
        self.model = GeologicalModel(self.bounding_box[0, :], self.bounding_box[1, :])
        self.scaled_points = self.model.scale(self.points)

    def process_axial_surface_proposition(self, axial_normal: np.ndarray) -> pd.DataFrame:
        """
        Process the axial surface proposition at each iteration by creating a dataset from the axial surface normal.

        The axial surface normal is first normalised. A dataset is then created from the axial surface normal.
        A rotated vector is created by rotating the axial normal by 90 degrees to be perpendicular to the axial surface.
        The rotated vector is the Y axis of the fold frame.

        Parameters
        ----------
        axial_normal : np.ndarray
            The axial surface normal vector.

        Returns
        -------
        pd.DataFrame
            The dataset to use to build a fold frame.
        """
        # normalise axial surface normal
        axial_normal /= np.linalg.norm(axial_normal)
        # create a dataset from the axial surface normal
        dataset = create_dataset(axial_normal, self.points, name='s1', coord=0)

        assert len(self.points) == len(self.gradient_data), "coordinates must have the same length as data"

        # rotate the axial normal by 90 degrees to create the Y axis of the fold frame
        y = rotate_vector(axial_normal, np.pi / 2, dimension=3)
        # create a dataset from the Y axis of the fold frame
        y_coord = create_dataset(y, self.points, name='s1', coord=1)

        # append the two datasets together
        dataset = pd.concat([dataset, y_coord])

        return dataset

    def build_fold_frame(self, axial_normal: np.ndarray) -> None:
        """
        Builds a fold frame.

        This function processes the axial surface proposition (iteration),
        updates the model data, creates and adds a fold frame to the model, and then interpolates the fold frame.

        Parameters
        ----------
        axial_normal : np.ndarray
            The axial surface normal vector proposition to be processed.

        Returns
        -------
        None
        """
        # process the axial surface proposition and get the dataset
        dataset = self.process_axial_surface_proposition(axial_normal)

        # update the model data with the dataset
        self.model.data = dataset

        # create and add a fold frame to the model
        self.axial_surface = self.model.create_and_add_fold_frame('s1',
                                                                  buffer=0.6,
                                                                  solver='pyamg',
                                                                  nelements=1e3,
                                                                  damp=True)

        # update the model
        self.model.update(progressbar=False)

    def create_and_build_fold_event(self) -> FoldEvent:
        """
        Creates and builds a fold event.

        A fold frame object is first created from the axial surface proposition.
        The gradient of the axial surface is then calculated and normalised.
        The fold limb rotation angle is calculated depending on the argument 'av_fold_axis'. If the argument is set True,
        the fold limb rotation angle is calculated using the average fold axis. If the argument is set False, the fold
        limb rotation angle is calculated down a folded fold axis, which is the case of noncylindrical folds.

        Returns
        -------
        FoldEvent
            The created fold event.
        """
        # create a fold frame object from the axial surface
        foldframe = FoldFrame('s1', self.axial_surface)

        # calculate the gradient of the axial surface
        s1g = self.axial_surface[0].evaluate_gradient(self.scaled_points)

        # normalise the gradient
        s1g /= np.linalg.norm(s1g, axis=1)[:, None]

        # check if 'av_fold_axis' is in kwargs and if it's True
        if 'av_fold_axis' in self.kwargs:
            # calculate intersection lineation li
            li = calculate_intersection_lineation(s1g, self.gradient_data)

            # calculate the mean of intersection lineation li and normalise it
            av_fold_axis = li.mean(0)
            av_fold_axis /= np.linalg.norm(av_fold_axis)

            # calculate the fold limb rotation angle
            flr, fld = foldframe.calculate_fold_limb_rotation(self.scaled_points,
                                                              self.gradient_data)

            # fit a fourier series to the fold limb rotation
            fitted_flr = self.fit_fourier_series(fld, flr, knowledge_type='fold_limb_rotation_angle')

            # create a fold function from the fitted fourier series
            fold_limb_rotation_function = fold_function(fitted_flr)

            # create a fold event with the axial surface, fold limb rotation function and fold axis
            fold = FoldEvent(self.axial_surface,
                             fold_limb_rotation=fold_limb_rotation_function, fold_axis=av_fold_axis)

            return fold

        # check if 'av_fold_axis' is not in kwargs or if it's False
        if 'av_fold_axis' not in self.kwargs:
            # calculate the fold axis rotation angle
            far, fad = foldframe.calculate_fold_axis_rotation(self.scaled_points, s1g)

            # fit a fourier series to the calculated fold axis rotation angle
            fitted_far = self.fit_fourier_series(fad, far, knowledge_type='fold_axis_rotation_angle')

            # create a fold function from the fitted fourier series
            fold_axis_rotation_function = fold_function(fitted_far)

            # create a fold event with the axial surface and fold axis rotation function
            fold = FoldEvent(self.axial_surface,
                             fold_axis_rotation=fold_axis_rotation_function)

            # calculate the fold limb rotation angle
            flr, fld = foldframe.calculate_fold_limb_rotation(self.scaled_points, self.gradient_data,
                                                              axis=fold.get_fold_axis_orientation)

            # fit a fourier series to the fold limb rotation
            fitted_flr = self.fit_fourier_series(fld, flr, knowledge_type='fold_limb_rotation_angle')

            # create a fold function from the fitted fourier series
            fold_limb_rotation_function = fold_function(fitted_flr)

            # set the fold limb rotation of the fold event
            fold.fold_limb_rotation = fold_limb_rotation_function

            return fold

    def calculate_svariogram(self, fold_frame: np.ndarray,
                             rotation_angles: np.ndarray) -> np.ndarray:
        """
        Calculates the S-Variogram (semi-variogram) of the fold rotation angles.
        for more details about the S-Variogram, see Grose et al (2017)

        Depending on the keyword arguments 'axis_wl' and 'limb_wl',
        this function either returns the initial guess of the fold axis or fold limb rotation angles
        to optimise a fourier series.

        Parameters
        ----------
        fold_frame : np.ndarray
            The fold frame to be used for the semi-variogram calculation.
        rotation_angles : np.ndarray
            The rotation angles to be used for the semi-variogram calculation.

        Returns
        -------
        np.ndarray
            the initial guess for fourier series optimisation.
        """
        # check if 'axis_wl' is in kwargs
        if 'axis_wl' in self.kwargs:
            return np.array([0, 1, 1, self.kwargs['axis_wl']])

        # check if 'limb_wl' is in kwargs
        if 'limb_wl' in self.kwargs:
            return np.array([0, 1, 1, self.kwargs['limb_wl']])

        # if neither 'axis_wl' nor 'limb_wl' is in kwargs
        else:
            # check if the length of rotation_angles is less than 5
            if len(rotation_angles) < 5:
                # calculate the pairwise distance
                pdist = np.abs(fold_frame[:, None] - fold_frame[None, :])
                pdist[pdist == 0.] = np.nan
                lagx = np.nanmean(np.nanmin(pdist, axis=1))

                # calculate the semivariogram
                theta, lags, variogram = calculate_semivariogram(fold_frame, rotation_angles,
                                                                 lag=lagx,
                                                                 nlag=60)


            else:
                # calculate the semivariogram
                theta, lags, variogram = calculate_semivariogram(fold_frame, rotation_angles,
                                                                 lag=None,
                                                                 nlag=None)

        return theta

    def fit_fourier_series(self, fold_frame_coordinate: np.ndarray, rotation_angle: np.ndarray,
                           knowledge_type: str = 'fold_limb_rotation_angle') -> List[float]:
        """
        Fit the Fourier series.

        Parameters
        ----------
        fold_frame_coordinate : np.ndarray
            The fold frame coordinate.
        rotation_angle :  np.ndarray
            The fold limb or axis rotation angle.
        knowledge_type : str, optional
            The type of knowledge, use 'fold_limb_rotation_angle' or 'fold_axis_rotation_angle',
            by default 'fold_limb_rotation_angle'.

        Returns
        -------
        List[float]
            Returns the result of the optimisation.
        """

        # Check the type of knowledge and generate x accordingly
        if knowledge_type == 'fold_axis_rotation_angle':
            x = np.linspace(self.axial_surface[1].min(), self.axial_surface[1].max(), 100)
        else:
            x = np.linspace(self.axial_surface[0].min(), self.axial_surface[0].max(), 100)

        # Create a FourierSeriesOptimiser instance
        fourier_optimiser = FourierSeriesOptimiser(fold_frame_coordinate, rotation_angle, x)

        if self.geological_knowledge is not None:

            opt = fourier_optimiser.optimise(geological_knowledge=self.geological_knowledge[knowledge_type])

            return opt.x

        else:

            # Optimise the Fourier series
            opt = fourier_optimiser.optimise()

            return opt.x

    def calculate_folded_foliation_vectors(self) -> np.ndarray:
        """
        Calculate the folded foliation vectors.

        Returns
        -------
        List[float]
            Returns the predicted bedding.
        """

        # Create and build fold event
        fold = self.create_and_build_fold_event()

        # Evaluate and normalize the gradient
        s1g = self.axial_surface[0].evaluate_gradient(self.scaled_points)
        s1g /= np.linalg.norm(s1g, axis=1)[:, None]

        # Get deformed orientation and normalize the fold direction
        fold_direction, fold_axis, gz = fold.get_deformed_orientation(self.scaled_points)
        fold_direction /= np.linalg.norm(fold_direction, axis=1)[:, None]

        # Correct any fold_direction vector to be consistent with the axial surface orientation
        dot = np.einsum('ij,ij->i', s1g, fold_direction)
        fold_direction[dot < 0] *= -1

        # Calculate predicted bedding and normalize it
        predicted_foliation = np.cross(fold_axis, fold_direction)
        predicted_foliation /= np.linalg.norm(predicted_foliation, axis=1)[:, None]

        # TODO write function that free up memory
        # Free up memory
        del fold_direction, fold_axis, fold, dot, s1g
        gc.collect()

        return predicted_foliation

    def get_predicted_foliation(self, axial_normal: np.ndarray) -> np.ndarray:
        """
        Get the predicted foliation.

        Parameters
        ----------
        axial_normal : np.ndarray
            The axial foliation normal vector.

        Returns
        -------
        np.ndarray
            Returns the normal vectors to the predicted foliation.
        """
        self.initialise_model()

        # Build the fold frame
        self.build_fold_frame(axial_normal)

        # Create and build fold event
        self.create_and_build_fold_event()

        # Calculate folded foliation vectors
        predicted_foliation = self.calculate_folded_foliation_vectors()

        return predicted_foliation
