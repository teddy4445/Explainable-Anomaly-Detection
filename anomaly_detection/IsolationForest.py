# library imports
import pandas as pd
from pyod.models.iforest import IForest

# project imports
from consts import *
from anomaly_detection.anomaly_algo import AnomalyAlgo


class IsolationForest(AnomalyAlgo):
    """
    A ...
    """

    # CONSTS #
    NAME = ""
    # END - CONSTS #

    def __init__(self,
                 n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=1,
                 behaviour='old', random_state=None, verbose=0):
        AnomalyAlgo.__init__(self)
        self.model = IForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination,
                             max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs, behaviour=behaviour,
                             random_state=random_state, verbose=verbose)

    def train(self,
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


if __name__ == '__main__':
    print()

    # ex = IsolationForest()
    # ex.train()
    # ex.predict()