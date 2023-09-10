from collections import OrderedDict
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from models import AD_MODELS, CLF_MODELS
from baseline_explainers import BASELINE_EXPLAINERS

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
    def encode_categorical_data(df, mode="one-hot"):
        all_cat_cols = df.select_dtypes(include=['object']).columns
        affective_cat_cols = [col for col in all_cat_cols if df[col].nunique() > 2]
        simple_cat_cols = [col for col in all_cat_cols if col not in affective_cat_cols]
        non_cat_cols = [col for col in df.columns if col not in all_cat_cols]

        if mode == 'label':
            encoder = LabelEncoder()
            for column in all_cat_cols:
                df[column] = encoder.fit_transform(df[column])
        elif mode == 'one-hot':
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            encoded_cat_feats = encoder.fit_transform(df.loc[:, affective_cat_cols])
            encoded_cat_feats_name = encoder.get_feature_names_out(affective_cat_cols)
            encoded_df = pd.DataFrame(encoded_cat_feats, columns=encoded_cat_feats_name)

            label_encoder = LabelEncoder()
            for column in simple_cat_cols:
                encoded_df[column] = label_encoder.fit_transform(df[column])

            for column in non_cat_cols:
                encoded_df[column] = df[column]

        return encoded_df

    @staticmethod
    def get_anomaly(df, model_name, supervised):
        if not supervised:
            model = AD_MODELS[model_name['type']]()  # TODO: add params
            model.fit(X=df)
            anomaly_index = model.predict_proba(df)[:, 1].argmax()

            anomaly_sample = df.iloc[anomaly_index]  # most anomalous
            dataset_wo_anomaly = pd.concat([df.iloc[:anomaly_index],
                                            df.iloc[anomaly_index + 1:]],
                                           ignore_index=True)
        else:
            model = CLF_MODELS[model_name['type']]()  # TODO: add params
            dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
            anomaly_sample = dataset.loc[df['assoc'] == 2].iloc[-1]
            dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)

            combined_data = np.vstack([dataset_wo_anomaly] + len(dataset_wo_anomaly) * [anomaly_sample])
            combined_labels = len(dataset_wo_anomaly) * [0] + len(dataset_wo_anomaly) * [1]
            model.fit(combined_data, combined_labels)

        return anomaly_sample, dataset_wo_anomaly, model

    def load_dataset(self, data_filename, supervised=False, model_name=None):
        directory = 'supervised' if supervised else 'unsupervised'
        df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, directory, data_filename))
        df = self.encode_categorical_data(df=df, mode="one-hot")
        anomaly_row, dataset_wo_anomaly, model = self.get_anomaly(df=df, model_name=model_name, supervised=supervised)
        return Data(data=df, anomaly_row=anomaly_row, data_wo_anomaly=dataset_wo_anomaly), model
