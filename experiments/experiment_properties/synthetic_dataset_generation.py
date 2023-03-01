# library imports
import random
import numpy as np
import pandas as pd
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo

# project imports
from consts import *
from explanation_analysis.similarity_metrices.sim_metric import SimMetric
from anomaly_detection_algos.DBSCAN import DBSCANwrapper
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from experiments.experiment_properties.feature_distribution_normal import FeatureDistributionNormal


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
                     performance_metric: SimMetric = None,
                     performance_metric_attempts: int = 0,
                     save_csv: str = None) -> tuple:
        """
        This function generates a single CSV file, and saves it if a path given such that the user controls the number of rows, cols, anomalise, and data dist
        :param anomaly_detection_algorithm: The function to determine if a sample is anomly or not given the entire dataset
        :param row_count: number of rows in the dataset
        :param cols_dist_functions: the distribution of the functions
        :param d_tag_size: the size of the d_tag dataset size
        :param f_diff: list of the features' indices that would determinate the anomalies
        :param save_csv: if not None, save the results into a csv file
        :param performance_metric: a metric to score D'
        :param performance_metric_attempts: number of times we try to generate better D'
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
            rows_indexes = d_tag_index_list[:d_tag_size - 1]
            d_tag = d.iloc[rows_indexes, :]
            d_tag_anomalies = anomaly_detection_algorithm.fit_and_self_predict(x=d_tag)

        # chose index at random
        s = d.iloc[d_tag_index_list[d_tag_size], :]
        d_tag = d_tag.append([s])

        #  if metric is given, we can try and improve D'
        if performance_metric_attempts > 0 and isinstance(performance_metric, SimMetric):
            best_score = 0
            for attempt in range(performance_metric_attempts):
                d_tag_anomalies = [True]
                while sum(d_tag_anomalies) > 0:
                    d_tag_index_list = list(range(d.shape[0]))
                    random.shuffle(d_tag_index_list)
                    rows_indexes = d_tag_index_list[:d_tag_size - 1]
                    d_tag = d.iloc[rows_indexes, :]
                    d_tag_anomalies = anomaly_detection_algorithm.fit_and_self_predict(x=d_tag)
                while sum(anomaly_detection_algorithm.fit_and_self_predict(x=d_tag)) != 1 or sum(
                        anomaly_detection_algorithm.fit_and_self_predict(x=d)) != 1:
                    anomaly_sample = d_tag.iloc[-1, :]
                    anomaly_sample = [
                        anomaly_sample[index] if index not in f_diff else cols_functions_list[index].sample()
                        for index in range(len(anomaly_sample))]
                    d.iloc[d_tag_index_list[d_tag_size], :] = anomaly_sample
                    d_tag.iloc[-1, :] = anomaly_sample
                # calc new score
                current_score = performance_metric.sim(d=d_tag,
                                                       s=anomaly_sample,
                                                       f_diff=f_diff,
                                                       f_sim=list(set(list(d)) - set(f_diff)))
                if current_score < best_score:
                    best_score = current_score
                    best_d_tag = d_tag
            d_tag = best_d_tag
        else:
            while sum(anomaly_detection_algorithm.fit_and_self_predict(x=d_tag)) != 1 or sum(
                    anomaly_detection_algorithm.fit_and_self_predict(x=d)) != 1:
                anomaly_sample = d_tag.iloc[-1, :]
                anomaly_sample = [anomaly_sample[index] if index not in f_diff else cols_functions_list[index].sample()
                                  for index in range(len(anomaly_sample))]
                d.iloc[d_tag_index_list[d_tag_size], :] = anomaly_sample
                d_tag.iloc[-1, :] = anomaly_sample

        assoc = list(np.zeros(row_count, dtype=int))
        for index in rows_indexes:
            assoc[index] = 1
        # assoc = [1 for i in range(row_count) if i in rows_indexes]
        assoc[d_tag_index_list[d_tag_size]] = 2
        d_inf = d
        d_inf['assoc'] = assoc

        # if have path, save it as CSV file
        if save_csv is not None and isinstance(save_csv, str) and os.path.exists(os.path.dirname(save_csv)):
            d.to_csv(save_csv + ".csv", index=False)
            d_inf.to_csv(save_csv + "_inf.csv", index=False)

        return d, d_tag, d_inf

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


if __name__ == '__main__':
    # 1) prepare IO
    for path in SETUP_FOLDERS:
        os.makedirs(path, exist_ok=True)

    # 2) generate dataset
    iterations = 1

    for iteration in range(iterations):
        print(f"generate dataset {iteration}")
        f_diff = [0, 1]
        d_tag_size = 10
        dataset, d_tag, d_inf = SyntheticDatasetGeneration.generate_one(
            anomaly_detection_algorithm=DBSCANwrapper(),
            row_count=50,
            cols_dist_functions={
                "a": FeatureDistributionNormal(mean=1,
                                               std=0.5),
                "b": FeatureDistributionNormal(mean=0.8,
                                               std=0.7),
                "c": FeatureDistributionNormal(mean=5, std=1)
            },
            f_diff=f_diff,
            d_tag_size=d_tag_size,
            save_csv=os.path.join(RESULTS_FOLDER_PATH,
                                  f"synt_example_iter{iteration}")
        )
