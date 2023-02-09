# library imports
import os
import pandas as pd

# project imports
from consts import *


class Experiment:
    """
    A virtual class of an experiment_properties
    """

    def __init__(self):
        self.results = {}

    def run(self,
            algo,
            dataset: pd.DataFrame,
            anomaly_sample: list,
            f_diff: list):
        """
        This method runs an algorithm on the experiment_properties's data and stores the results needed for this experiment_properties
        """
        pass

    def test_results(self):
        """
        This method analyze the results of the algorithm, even more than one run and produces a result data structure
        """
        pass
