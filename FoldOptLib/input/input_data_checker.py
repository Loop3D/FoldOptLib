import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class CheckInputData:
    """
    A class used to check the input data for the optimisation.

    ...

    Attributes
    ----------
    folded_foliation_data : pd.DataFrame
        The data related to a folded foliation or bedding
    bounding_box : nd.array
        The bounding box of the model area
    knowledge_constraints : dict
        The knowledge constraints data (default is None)


    Methods
    -------
    check_foliation_data():
        Checks if the foliation data is a pandas dataframe and has the correct columns.
    check_input_geological_knowledge():
        Checks if the input geological knowledge constraints is a dictionary,
        and has the correct format.
    check_bounding_box():
        Checks if the bounding box is a numpy array and has the correct format.
    check_input_data():
        Checks all the input data for the optimisation.
    """

    def __init__(self, folded_foliation_data, bounding_box, geological_knowledge=None):
        """
        Constructs all the necessary attributes for the CheckInputData object.

        """

        self.folded_foliation_data = folded_foliation_data
        self.bounding_box = bounding_box
        self.geological_knowledge = geological_knowledge

    def check_foliation_data(self):
        """
        Check the foliation data is a pandas dataframe and has the correct columns: X, Y, Z, feature_name and
        either strike, dip, or gx, gy, gz
        """
        # # check if the foliation data is a pandas dataframe
        if not isinstance(self.folded_foliation_data, pd.DataFrame):
            raise TypeError("Foliation data must be a pandas DataFrame.")
        required_columns = ['X', 'Y', 'Z', 'feature_name']
        if not all(column in self.folded_foliation_data.columns for column in required_columns):
            raise ValueError("Foliation data must have the columns: X, Y, Z, feature_name.")
        if not (all(column in self.folded_foliation_data.columns for column in ['strike', 'dip']) or
                all(column in self.folded_foliation_data.columns for column in ['gx', 'gy', 'gz'])):
            raise ValueError("Foliation data must have either strike, dip or gx, gy, gz columns.")

    # TODO : 1. rewrite check_knowledge_constraints
    #  2. then test it before
    #  3. implementing it in the geological knowledge class

    def check_input_geological_knowledge(self):
        """
        verify the format of the provided nested dictionary.
        TODO : add support for dict check for restricted mode of optimisation

        Parameters
        ----------

        Raises
        ------
        ValueError
            If the dictionary format is not as expected.

        Returns
        -------
        bool
            True if the format is correct, otherwise raises an error.
        """

        # Define the expected keys for different types of inner dictionaries
        general_keys = {'mu', 'sigma', 'w'}
        # axial_trace_keys = ['mu', 'sigma', 'w']
        axial_surface_keys = {'mu', 'kappa', 'w'}

        # Helper function to validate the format of the innermost dictionaries
        def validate_inner_dict(d, expected_keys):
            if not set(d.keys()) == set(expected_keys):
                raise ValueError(f"Expected keys {expected_keys} but got {d.keys()}")
            for key, value in d.items():
                if not isinstance(value, (int, float, list)):
                    raise ValueError(f"Expected {key} to have numeric value or a "
                                     f"list but got {type(value)}")

        for key, sub_dict in self.geological_knowledge.items():
            if key == 'fold_axial_surface':
                # for inner_key, values in sub_dict.items():
                validate_inner_dict(sub_dict, axial_surface_keys)
            elif key.startswith('axial_trace_'):
                validate_inner_dict(sub_dict, general_keys)
            else:
                validate_inner_dict(sub_dict, general_keys)

        return True

    def check_bounding_box(self):
        """
        check if the bounding_box is an numpy array of the following format
        [[minX, minY, minZ], [maxX, maxY, maxZ]]
        """
        # check if the bounding box is a numpy array or a list
        if not isinstance(self.bounding_box, np.ndarray):
            raise TypeError("Bounding box must be a numpy array.")
        # check if the bounding box is empty
        if self.bounding_box.size == 0:
            raise ValueError("bounding_box array is empty.")
        # check if the bounding box has the correct format
        if not len(self.bounding_box[0]) == 3 and not len(self.bounding_box[1]) == 3:
            raise ValueError("Bounding box must have the following format: [[minX, maxX, minY], [maxY, minZ, maxZ]]")

    # write a function that checks all the input data
    def check_input_data(self):
        """
        Check the input data for the optimisation
        """
        self.check_bounding_box()
        self.check_foliation_data()
        if self.geological_knowledge is not None:
            self.check_input_geological_knowledge()

