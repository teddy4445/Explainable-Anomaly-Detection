# library imports
import random
import numpy as np
import pandas as pd
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo

# project imports
from consts import *


class SyntheticDatasetGeneration:
    """
    A class responsible for generating synthetic data with wanted properties
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_one(anomaly_detection_algorithm: AnomalyAlgo,
                     row_count: int,
                     cols_dist_functions: dict,
                     f_diff: list,
                     d_tag_size: int,
                     save_csv: str = None) -> tuple:
        """
        This function generates a single CSV file, and saves it if a path given such that the user controls the number of rows, cols, anomalise, and data dist
        :param anomaly_detection_algorithm: The function to determine if a sample is anomly or not given the entire dataset
        :param row_count: number of rows in the dataset
        :param cols_dist_functions: the distribution of the functions
        :param d_tag_size: the size of the d_tag dataset size
        :param f_diff: list of the features' indices that would determinate the anomalies
        :param save_csv: if not None, save the results into a csv file
        :return: pd.DataFrame with the dataset
        """
        # make sure the inputs are legit
        assert d_tag_size < row_count
        assert issubclass(type(anomaly_detection_algorithm), AnomalyAlgo)

        # get the feature functions as list to query later
        cols_functions_list = list(cols_dist_functions.values())

        # generate first D
        d = pd.DataFrame([[feature_func.sample() for feature_name, feature_func in cols_dist_functions.items()] for _ in
                          range(row_count)])
        # generate D until works fine
        d_anomalies = anomaly_detection_algorithm.fit_and_self_predict(x=d)
        while sum(d_anomalies) > 0:
            d = d[[not val for val in d_anomalies]]
            d = pd.concat([d, pd.DataFrame(
                [[feature_func.sample() for feature_name, feature_func in cols_dist_functions.items()] for _ in
                 range(row_count - d.shape[0])])], ignore_index=True)
            d_anomalies = anomaly_detection_algorithm.fit_and_self_predict(x=d)

        # at this point we have D fine and wish to generate a single d_tag
        d_tag_anomalies = [True]
        while sum(d_tag_anomalies) > 0:
            d_tag_index_list = list(range(d.shape[0]))
            random.shuffle(d_tag_index_list)
            rows_indexes = d_tag_index_list[:d_tag_size]
            d_tag = d.iloc[rows_indexes, :]
            d_tag_anomalies = anomaly_detection_algorithm.fit_and_self_predict(x=d_tag)

        # chose index at random
        s = d.iloc[-1, :]
        d_tag.append([s])
        # generate s
        while sum(anomaly_detection_algorithm.fit_and_self_predict(x=d_tag)) != 1 and sum(
                anomaly_detection_algorithm.fit_and_self_predict(x=d)) != 1:
            anomaly_sample = d.iloc[-1, :]
            anomaly_sample = [anomaly_sample[index] if index not in f_diff else cols_functions_list[index].sample() for
                              index in range(len(anomaly_sample))]
            d.iloc[-1, :] = anomaly_sample
            d_tag.iloc[-1, :] = anomaly_sample

        # if have path, save it as CSV file
        if save_csv is not None and isinstance(save_csv, str) and os.path.exists(os.path.dirname(save_csv)):
            d.to_csv(save_csv, index=False)
        return d, d_tag

    @staticmethod
    def generate_many(anomaly_detection_algorithm: AnomalyAlgo,
                      row_count: int,
                      cols_dist_functions: dict,
                      f_diff: list,
                      d_tag_size: int,
                      save_csvs: list,
                      count: int):
        return [SyntheticDatasetGeneration.generate_one(anomaly_detection_algorithm=anomaly_detection_algorithm,
                                                        row_count=row_count,
                                                        cols_dist_functions=cols_dist_functions,
                                                        d_tag_size=d_tag_size,
                                                        f_diff=f_diff,
                                                        save_csv=save_csvs[index])
                for index in range(count)]
