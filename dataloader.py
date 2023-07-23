from collections import OrderedDict
import os
import pandas as pd

from ad_algos.IsolationForest import IsolationForestwrapper

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'datasets')


class Data:
    def __init__(self, data, anomaly_row, data_wo_anomaly):
        self._raw_data = data
        self.anomaly_row = anomaly_row
        self.data_wo_anomaly = data_wo_anomaly


class Dataloader:
    def __init__(self):
        pass

    @staticmethod
    def get_supervised_anomaly(df):
        dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
        anomaly_sample = dataset.loc[df['assoc'] == 2].iloc[-1]
        dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)
        return anomaly_sample, dataset_wo_anomaly

    @staticmethod
    def get_unsupervised_anomaly(df):
        # TODO: Change to an AD-based
        ad_algo = IsolationForestwrapper()
        ad_algo.fit(df)
        labels = ad_algo.predict(df)
        anomaly_indexes = [i for i in range(len(labels)) if labels[i] == 1]

        anomaly_sample = df.iloc[anomaly_indexes[0]]  # first anomaly
        dataset_wo_anomaly = pd.concat(
            [df.iloc[:anomaly_indexes[0]], df.iloc[anomaly_indexes[0] + 1:]],
            ignore_index=True)
        return anomaly_sample, dataset_wo_anomaly

    def load_dataset(self, data_filename, supervised=False):
        supervised_directory = 'supervised' if supervised else 'unsupervised'
        df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, supervised_directory, data_filename))
        anomaly_row, dataset_wo_anomaly = self.get_supervised_anomaly(df) if supervised \
            else self.get_unsupervised_anomaly(df)
        return Data(data=df, anomaly_row=anomaly_row, data_wo_anomaly=dataset_wo_anomaly)
