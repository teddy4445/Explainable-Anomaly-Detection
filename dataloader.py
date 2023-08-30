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
        # anomaly_scores = ad_algo.predict(df)
        anomaly_index = ad_algo.predict_scores(df).argmax()

        anomaly_sample = df.iloc[anomaly_index]  # most anomalous
        dataset_wo_anomaly = pd.concat(
            [df.iloc[:anomaly_index], df.iloc[anomaly_index + 1:]],
            ignore_index=True)
        return anomaly_sample, dataset_wo_anomaly

    def load_dataset(self, data_filename, supervised=False):
        directory = 'supervised' if supervised else 'unsupervised'
        df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, directory, data_filename))
        anomaly_row, dataset_wo_anomaly = self.get_supervised_anomaly(df) if supervised \
            else self.get_unsupervised_anomaly(df)
        return Data(data=df, anomaly_row=anomaly_row, data_wo_anomaly=dataset_wo_anomaly)
