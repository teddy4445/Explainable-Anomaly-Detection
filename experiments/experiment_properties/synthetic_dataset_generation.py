# library imports
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
                     f_diff_list: list,
                     d_tag_size_list: list,
                     save_csv: str = None) -> pd.DataFrame:
        """
        This function generates a single CSV file, and saves it if a path given such that the user controls the number of rows, cols, anomalise, and data dist
        :param anomaly_detection_algorithm: The function to determine if a sample is anomly or not given the entire dataset
        :param row_count: number of rows in the dataset
        :param d_tag_size_list: list of the sizes of the d_tag sizes
        :param f_diff_list: list of list of the features' indices that would determinate the anomalies
        :param cols_dist_functions: the distribution of the functions
        :param save_csv: if not None, save the results into a csv file
        :return: pd.DataFrame with the dataset
        """
        # make sure the inputs are legit
        assert sum(d_tag_size_list) > len(d_tag_size_list)
        assert sum(d_tag_size_list) + len(d_tag_size_list) < row_count
        assert len(d_tag_size_list) == len(f_diff_list)
        assert issubclass(type(anomaly_detection_algorithm), AnomalyAlgo)

        # get the feature functions as list to query later
        cols_functions_list = list(cols_dist_functions.values())

        samples = []
        for anomaly_index, d_tag_size in enumerate(d_tag_size_list):
            d_tag = [[feature_func.sample() for feature_name, feature_func in cols_dist_functions.items()] for _ in range(d_tag_size)]
            # build d_tag
            while sum(anomaly_detection_algorithm.fit_and_self_predict(x=pd.DataFrame(d_tag))) != 1:
                # try again
                d_tag = [[feature_func.sample() for feature_name, feature_func in cols_dist_functions.items()] for _ in
                         range(d_tag_size)]
            # convert the last line to be an anomaly
            # this would work due to the entropy of random walk in high dimension
            anomaly_sample = [feature_func.sample() for feature_name, feature_func in cols_dist_functions.items()]
            while not anomaly_detection_algorithm.fit_than_predict(x=pd.DataFrame(d_tag),
                                                                   x_predict=pd.DataFrame([anomaly_sample]))[0]:
                # update the last sample which would be the anomaly with another try of the f_diff_list until an anomaly is obtained
                anomaly_sample = [anomaly_sample[index] if index not in f_diff_list[anomaly_index] else cols_functions_list[index].sample()
                                  for index in range(len(anomaly_sample))]
            # at this point we have an explain with the right f_diff features
            samples.extend(d_tag)
        # the remaining can be added simply such that we do not introduce more anomalies
        while len(samples) < row_count:
            a = [feature_func.sample() for feature_name, feature_func in cols_dist_functions.items()]
            samples.append(a)
            if sum(anomaly_detection_algorithm.fit_and_self_predict(x=pd.DataFrame(samples))) != len(d_tag_size_list):
                samples.remove(a)
        df = pd.DataFrame(samples,
                          columns=list(cols_dist_functions.keys()))
        # if have path, save it as CSV file
        if save_csv is not None and isinstance(save_csv, str) and os.path.exists(os.path.dirname(save_csv)):
            df.to_csv(save_csv, index=False)
        return df

    @staticmethod
    def generate_many(anomaly_detection_algorithm: AnomalyAlgo,
                      row_count: int,
                      cols_dist_functions: dict,
                      f_diff_list: list,
                      d_tag_size_list: list,
                      save_csvs: list,
                      count: int):
        return [SyntheticDatasetGeneration.generate_one(anomaly_detection_algorithm=anomaly_detection_algorithm,
                                                        row_count=row_count,
                                                        cols_dist_functions=cols_dist_functions,
                                                        d_tag_size_list=d_tag_size_list,
                                                        f_diff_list=f_diff_list,
                                                        save_csv=save_csvs[index])
                for index in range(count)]
