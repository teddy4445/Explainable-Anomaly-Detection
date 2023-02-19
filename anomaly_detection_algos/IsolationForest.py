# library imports
import pandas as pd
from pyod.models.iforest import IForest

# project imports
from consts import *
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo


class IsolationForest(AnomalyAlgo):
    """
    A ...
    """

    # CONSTS #
    NAME = "IsolationForest"
    # END - CONSTS #

    def __init__(self,
                 n_estimators: int = 100,
                 max_samples='auto',
                 contamination: float = 0.1,
                 max_features: int = 1.0,
                 bootstrap=False,
                 n_jobs: int = 1,
                 behaviour='old',
                 random_state: int = None,
                 verbose: int =0):
        AnomalyAlgo.__init__(self)
        self.model = IForest(n_estimators=n_estimators,
                             max_samples=max_samples,
                             contamination=contamination,
                             max_features=max_features,
                             bootstrap=bootstrap,
                             n_jobs=n_jobs,
                             behaviour=behaviour,
                             random_state=random_state,
                             verbose=verbose)

    def fit(self,
              x: pd.DataFrame,
              y: pd.DataFrame = None,
              properties: dict = None):
        """
        This method used to train the algorithm
        """
        self.model.fit(X=x, y=y)

    def predict(self,
                x: pd.DataFrame):
        """
        This method used to inference
        """
        return self.model.predict(X=x)
