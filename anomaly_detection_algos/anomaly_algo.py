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

    def __init__(self,
                 properties: dict = None):
        self.properties = properties if isinstance(properties, dict) else {}

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame = None):
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
        This method used to inference
        """
        # TODO: handle the case in which x is 1-dim and make it 2-dim so it will work
        # if metric is None:
        # metric = accuracy_score
        # TODO: finish later

    def fit_than_predict(self,
                         x: pd.DataFrame,
                         x_predict: pd.DataFrame,
                         y: pd.DataFrame = None):
        """
        This method used to train the algorithm and than make a prediction right after
        """
        self.fit(x=x,
                 y=y)
        return self.predict(x=x_predict)

    def fit_and_self_predict(self,
                             x: pd.DataFrame,
                             y: pd.DataFrame = None):
        """
        This method used to train the algorithm and than make a prediction right after
        """
        self.fit(x=x,
                 y=y)
        return self.predict(x=x)
