import pandas as pd
from typing import List, Optional, Dict
import numpy as np
from .input_data_checker import CheckInputData
from ..helper._helper import *
from ..helper.utils import *


def _normalise_gradient(gradient: np.ndarray) -> np.ndarray:
    """Normalise gradient vectors."""
    return gradient / np.linalg.norm(gradient, axis=1)[:, None]


class InputDataProcessor(CheckInputData):

    def __init__(self, data: pd.DataFrame, bounding_box: np.ndarray,
                 geological_knowledge: Dict = None) -> None:
        """
        Constructs all the necessary attributes for the InputDataProcessor object.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be processed.
        bounding_box : np.ndarray
            Bounding box for the model.
        geological_knowledge : Dict, optional
            geological knowledge dictionary.
        """
        super().__init__(data, bounding_box, geological_knowledge)  # Assuming parent class requires this
        self.data = data
        self.bounding_box = bounding_box
        self.knowledge_constraints = geological_knowledge

    def process_data(self):
        check_data = CheckInputData(self.data, self.bounding_box, self.knowledge_constraints)
        check_data.check_input_data()  # check the input data is valid

        if 'strike' in self.data.columns and 'dip' in self.data.columns:
            strike = self.data['strike'].to_numpy()
            dip = self.data['dip'].to_numpy()
            gradient = strike_dip_to_vectors(strike, dip)
        elif 'gx' in self.data.columns and 'gy' in self.data.columns and 'gz' in self.data.columns:
            gradient = self.data[['gx', 'gy', 'gz']].to_numpy()
        else:
            return None

        gradient = _normalise_gradient(gradient)
        self.data['gx'], self.data['gy'], self.data['gz'] = gradient[:, 0], gradient[:, 1], gradient[:, 2]

        return self.data
