# library imports
import numpy as np
import pandas as pd

# project imports
from old_version.anomaly_detection_algos.anomaly_algo import AnomalyAlgo


class Zscore(AnomalyAlgo):
    """
    A z-score
    """

    # CONSTS #
    PROPERTY_NAME = "z"
    TRASH_UP = True
    TRASH_DOWN = False
    NAME = "Zscore"

    # END - CONSTS #

    def __init__(self,
                 properties: dict = None):
        AnomalyAlgo.__init__(self,
                             properties=properties)
        self.thresholds = {}
        self.edge_case = False

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame = None):
        # Make sure we can work
        assert Zscore.PROPERTY_NAME in self.properties
        # the properties of the algo
        self.thresholds = {}
        self.edge_case = False
        for col in list(x):
            mean_val = x[col].mean()
            std_val = x[col].std()
            if np.isnan(std_val):
                std_val = 0
                self.edge_case = True
                break
            self.thresholds[col] = {
                Zscore.TRASH_UP: mean_val + self.properties[Zscore.PROPERTY_NAME] * std_val,
                Zscore.TRASH_DOWN: mean_val - self.properties[Zscore.PROPERTY_NAME] * std_val,
            }

    def predict(self,
                x: pd.DataFrame):
        if self.edge_case:
            return x.apply(lambda row: False)
        return [
            any([row[col] > self.thresholds[col][Zscore.TRASH_UP] or row[col] < self.thresholds[col][Zscore.TRASH_DOWN]
                 for col in list(x)])
            for row_index, row in x.iterrows()]
