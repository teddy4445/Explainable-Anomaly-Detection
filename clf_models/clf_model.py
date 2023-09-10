# library imports
import pandas as pd


class ClassifierModel:
    def __init__(self, properties: dict = None):
        self.properties = properties if isinstance(properties, dict) else {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        pass

    def predict(self, x: pd.DataFrame):
        pass

    def anomaly_scores(self, x: pd.DataFrame):
        return self.predict(x)
