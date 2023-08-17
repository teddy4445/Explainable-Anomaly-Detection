import numpy as np


class BaselineExplainer:
    """
    An abstract class for the baseline_explainers classes
    """

    def __init__(self, data, classifier):
        self.data = data
        self.dataset_size = len(self.data)
        self.features = list(self.data.columns.values)
        self.features_num = len(self.features)
        self.data_labels = self.dataset_size * [0]
        self.classifier = classifier

    def trained_pseudo_ad_model(self, anomaly):
        # Combine data and anomaly
        combined_data = np.vstack([self.data] + self.dataset_size * [anomaly])
        combined_labels = self.dataset_size * [0] + self.dataset_size * [1]  # The last one is an anomaly

        # Train a DecisionTree model
        clf = self.classifier()
        clf.fit(combined_data, combined_labels)
        return clf

    def get_explanation(self, anomaly):
        pass
