# library imports
import pandas as pd
from pyod.models.ocsvm import OCSVM

# project imports
from ad_algos.anomaly_algo import AnomalyAlgo


class OneClassSVMwrapper(AnomalyAlgo):
    """
    A ...
    """

    # CONSTS #
    NAME = "OneClassSVM"

    # END - CONSTS #

    def __init__(self,
                 kernel='rbf',
                 degree: int = 3,
                 gamma='auto',
                 coef0: float = 0.0,
                 tol: float = 0.001,
                 nu: float = 0.5,
                 shrinking=True,
                 cache_size=200,
                 verbose=False,
                 max_iter: int = -1,
                 contamination: float = 0.1):
        AnomalyAlgo.__init__(self)
        self.model = OCSVM(kernel=kernel,
                           degree=degree,
                           gamma=gamma,
                           coef0=coef0,
                           tol=tol,
                           nu=nu,
                           shrinking=shrinking,
                           cache_size=cache_size,
                           verbose=verbose,
                           max_iter=max_iter,
                           contamination=contamination)

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame = None):
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
