# library imports
import pandas as pd
from sklearn.cluster import DBSCAN

# project imports
from ad_models.ad_model import AnomalyDetectionModel


class DBSCAN(AnomalyDetectionModel):
    """
    DBSCAN  ------------------ Not being used
    """
    def __init__(self,
                 eps=0.5,
                 min_samples=5,
                 metric='euclidean',
                 metric_params=None, algorithm='auto',
                 leaf_size=30,
                 p=None,
                 n_jobs=None):
        AnomalyDetectionModel.__init__(self)
        self.model = DBSCAN(eps=eps,
                            min_samples=min_samples,
                            metric=metric,
                            metric_params=metric_params,
                            algorithm=algorithm,
                            leaf_size=leaf_size,
                            p=p,
                            n_jobs=n_jobs)

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame = None):
        self.model.fit(X=x, y=y)

    def predict(self,
                x: pd.DataFrame):
        return [val == -1 for val in self.model.fit_predict(X=x)]
