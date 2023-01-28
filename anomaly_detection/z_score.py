# library imports
import pandas as pd

# project imports
from consts import *
from anomaly_detection.anomaly_algo import AnomalyAlgo


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

    def __init__(self):
        AnomalyAlgo.__init__(self)
        self.trasholds = {}

    def train(self,
              x: pd.DataFrame,
              y: pd.DataFrame = None,
              properties: dict = None):
        # Make sure we can work
        assert Zscore.PROPERTY_NAME in properties
        # the properties of the algo
        self.trasholds = {}
        for col in list(x):
            mean_val = x[col].mean()
            std_val = x[col].std()
            self.trasholds[col][Zscore.TRASH_UP] = mean_val + properties[Zscore.PROPERTY_NAME] * std_val
            self.trasholds[col][Zscore.TRASH_DOWN] = mean_val - properties[Zscore.PROPERTY_NAME] * std_val

    def predict(self,
                x: pd.DataFrame):
        return x.apply(lambda row: any([row[col] > self.trasholds[col][Zscore.TRASH_UP] or
                                        row[col] < self.trasholds[col][Zscore.TRASH_DOWN]
                                        for col in row]))
