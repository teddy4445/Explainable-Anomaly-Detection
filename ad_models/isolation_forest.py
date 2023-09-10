# library imports
import pandas as pd
from pyod.models.iforest import IForest

# project imports
from ad_models.ad_model import AnomalyDetectionModel


class IsolationForest(AnomalyDetectionModel):
    def __init__(self,
                 n_estimators: int = 100,
                 max_samples='auto',
                 contamination: float = 0.1,
                 max_features: int = 1.0,
                 bootstrap=False,
                 n_jobs: int = 1,
                 behaviour='old',
                 random_state: int = None,
                 verbose: int = 0):
        AnomalyDetectionModel.__init__(self)
        self.model = IForest(n_estimators=n_estimators,
                             max_samples=max_samples,
                             contamination=contamination,
                             max_features=max_features,
                             bootstrap=bootstrap,
                             n_jobs=n_jobs,
                             behaviour=behaviour,
                             random_state=random_state,
                             verbose=verbose)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None):
        self.model.fit(X=x)

    def predict(self, x: pd.DataFrame):
        return self.model.predict(X=x)

    def anomaly_scores(self, x):
        # decision_function(X) - The anomaly score of the input samples
        # predict_proba(X, method='linear', return_confidence=False) - Probability of a sample being outlier
        return self.model.predict_proba(X=x)
