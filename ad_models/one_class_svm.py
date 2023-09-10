# library imports
import pandas as pd
from pyod.models.ocsvm import OCSVM

# project imports
from ad_models.ad_model import AnomalyDetectionModel


class OneClassSVM(AnomalyDetectionModel):
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
        AnomalyDetectionModel.__init__(self)
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

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None):
        self.model.fit(X=x)

    def predict(self, x: pd.DataFrame):
        return self.model.predict(X=x)

    def anomaly_scores(self, x):
        # decision_function(X) - The anomaly score of the input samples
        # predict_proba(X, method='linear', return_confidence=False) - Probability of a sample being outlier
        return self.model.predict_proba(X=x)
