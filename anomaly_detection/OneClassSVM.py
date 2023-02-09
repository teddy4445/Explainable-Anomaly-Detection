# library imports
import pandas as pd
from pyod.models.ocsvm import OCSVM

# project imports
from consts import *
from anomaly_detection.anomaly_algo import AnomalyAlgo


class OneClassSVM(AnomalyAlgo):
    """
    A ...
    """

    # CONSTS #
    NAME = ""
    # END - CONSTS #

    def __init__(self,
                 kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, contamination=0.1):
        AnomalyAlgo.__init__(self)
        self.model = OCSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,shrinking=shrinking,
                           cache_size=cache_size, verbose=verbose, max_iter=max_iter, contamination=contamination)

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
