# library imports
import pandas as pd
from sklearn.covariance import EllipticEnvelope

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
                 store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1,
                 random_state=None):
        AnomalyAlgo.__init__(self)
        self.model = EllipticEnvelope(store_precision=store_precision, assume_centered=assume_centered,
                                      support_fraction=support_fraction, contamination=contamination,
                                      random_state=random_state)
        self._data = None

    def train(self,
              x: pd.DataFrame
              y: pd.DataFrame = None,
              properties: dict = None):
        """
        This method used to train the algorithm
        """
        if not self._data:
            self._data = x
            self.model.fit(X=x, y=y)

    def predict(self,
                x: pd.DataFrame):
        """
        This method used to inference
        """
        return self.model.predict(X=x)


if __name__ == '__main__':
    print()
