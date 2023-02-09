# library imports
import pandas as pd

# project imports
from consts import *
from anomaly_detection.z_score import Zscore


class AnomalyDetectionRunner:
    """
    A class with multiple anomaly detection algorithms that standardize them so it will be easy to test
    """

    # ANOMALY ALGORITHMS #
    ALGOS = {Zscore.NAME: Zscore}
    PROPERTIES = {Zscore.NAME: {Zscore.PROPERTY_NAME: 1.0}}

    # END - ANOMALY ALGORITHMS #

    def __init__(self):
        self.models = {}

    def train(self,
              x: pd.DataFrame,
              y: pd.DataFrame = None):
        """
        This method runs an algorithm on the experiment's data and stores the results needed for this experiment
        """
        self.models = {}
        for name, algo in AnomalyDetectionRunner.ALGOS.items():
            self.models[name] = algo().train(x=x,
                                             y=y,
                                             properties=AnomalyDetectionRunner.PROPERTIES[name])

    def test(self,
             x: pd.DataFrame,
             y: pd.DataFrame,
             metric):
        """
        This method analyze the results of the algorithm, even more than one run and produces a result data structure
        """
        results = {}
        for name, model in self.models.items():
            results[name] = model.test(x=x,
                                       y=y,
                                       metric=metric)
        return results
