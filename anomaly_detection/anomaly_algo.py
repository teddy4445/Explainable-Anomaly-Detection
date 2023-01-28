# library imports
import pandas as pd
from sklearn.metrics import f1_score

# project imports
from consts import *


class AnomalyAlgo:
    """
    A virtual class of an Anomaly Algorithm
    """

    # CONSTS #
    NAME = ""
    # END - CONSTS #

    def __init__(self):
        pass

    def train(self,
              x: pd.DataFrame,
              y: pd.DataFrame = None,
              properties: dict = None):
        """
        This method used to train the algorithm
        """
        # TODO: finish later
        pass

    def test(self,
             x: pd.DataFrame,
             y: pd.DataFrame,
             metric=None):
        """
        This method  used to train the algorithm
        """
        if metric is None:
            metric = f1_score
        y_pred = self.predict(x=x)
        return metric(y_pred, y)

    def predict(self,
                x: pd.DataFrame):
        """
        This method  used to train the algorithm
        """
        # TODO: handle the case in which x is 1-dim and make it 2-dim so it will work
        if mertic is None:
            metric = accuracy_score
        # TODO: finish later
