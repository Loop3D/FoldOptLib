from abc import ABC, abstractmethod
import numpy as np


class BaseFoldFrameBuilder(ABC):

    @abstractmethod
    def initialise_model(self):
        """
        Setup the bounding box of the fold frame
        """
        pass

    # @abstractmethod
    # def import_data(self):
    #     """
    #     Import the data from the input file
    #     """
    #     pass
    #
    # @abstractmethod
    # def setup_fold_frame_interpolation(self):
    #     """
    #     Setup the fold frame interpolation
    #     """
    #     pass

    @abstractmethod
    def build_fold_frame(self, axial_normal: np.ndarray) -> None:
        """
        Build the fold frame
        """
        pass
