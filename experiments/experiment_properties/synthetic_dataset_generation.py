# library imports
import numpy as np
import pandas as pd

# project imports
from consts import *


class SyntheticDatasetGeneration:
    """
    A class responsible for generating synthetic data with wanted properties
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_one(anomaly_detection_algorithm,
                     row_count: int,
                     cols_dist_functions: dict,
                     anomaly_cases: int,
                     f_diff: list,
                     save_csv: str = None):
        """
        This function generates a single CSV file, and saves it if a path given such that the user controls the number of rows, cols, anomalise, and data dist
        :param anomaly_detection_algorithm: The function to determine if a sample is anomly or not given the entire dataset
        :param row_count: number of rows in the dataset
        :param anomaly_cases: number of anomaly rows in the dataset
        :param f_diff: list of features that would determinate the anomalies
        :param cols_dist_functions: the distribution of the functions
        :param save_csv: if not None, save the results into a csv file
        :return: pd.DataFrame with the dataset
        """

        # first, generate the dataframe
        df = pd.DataFrame(
            data=np.transpose(np.array([col_func.sample_col(row_count) for col_func in cols_dist_functions])),
            columns=[col_name for col_name in cols_dist_functions])
        # second, make sure I have anomalies
        # TODO: think about it

        # if have path, save it as CSV file
        if save_csv is not None and isinstance(save_csv, str) and os.path.exists(os.path.dirname(save_csv)):
            df.to_csv(save_csv, index=False)
        return df

    @staticmethod
    def generate_many(anomaly_detection_algorithm,
                      row_count: int,
                      cols_dist_functions: dict,
                      anomaly_cases: int,
                      f_diff: list,
                      save_csvs: list,
                      count: int):
        return [SyntheticDatasetGeneration.generate_one(anomaly_detection_algorithm=anomaly_detection_algorithm,
                                                        row_count=row_count,
                                                        cols_dist_functions=cols_dist_functions,
                                                        anomaly_cases=anomaly_cases,
                                                        f_diff=f_diff,
                                                        save_csv=save_csvs[index])
                for index in range(count)]
