# library imports
import os
import pandas as pd

# project imports
from solvers.solver import Solver
from explanation_analysis.afes.afes_metric import AfesMetric
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo


class Experiment:
    """
    A virtual class of an experiment_properties
    """

    def __init__(self,
                 time_limit_seconds: float):
        self.results = {}
        self._time_limit_seconds = time_limit_seconds

    def run(self,
            anomaly_algo: AnomalyAlgo,
            solver: Solver,
            scorer: AfesMetric,
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
