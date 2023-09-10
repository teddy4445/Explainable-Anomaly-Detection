# library imports
import pandas as pd
from sklearn.svm import SVC

# project imports
from clf_models.clf_model import ClassifierModel


class SVM(ClassifierModel):
    def __init__(self):
        ClassifierModel.__init__(self)
        self.model = SVC()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None):
        self.model.fit(X=x, y=y)

    def predict(self, x: pd.DataFrame):
        return self.model.predict(X=x)

    def anomaly_scores(self, x):
        # decision_function(X) - The anomaly score of the input samples
        # predict_proba(X, method='linear', return_confidence=False) - Probability of a sample being outlier
        return self.model.predict_proba(X=x)[:,1]
