# library imports
import os
import time

import pandas as pd

# project imports
from solvers.solver import Solver
from explanation_analysis.afes.afes_metric import AfesMetric
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo


class Experiment:
    """
    A virtual class of an experiment_properties
    """

    def __init__(self,
                 time_limit_seconds: float):
        self.results = {}
        self.baseline = {}
        self._time_limit_seconds = time_limit_seconds

    def run(self,
            anomaly_algo: AnomalyAlgo,
            solver: Solver,
            scorer: AfesMetric,
            dataset: pd.DataFrame,
            anomaly_sample: list,
            d_tags: list,
            f_diff_list: list):
        """
        This method runs an algorithm on the experiment_properties's data and stores the results needed for this experiment_properties
        """
        solving_start_time = time.perf_counter()
        best_ans, best_ans_score = solver.solve(d=dataset,
                                    anomaly_algo=anomaly_algo,
                                    s=anomaly_sample,
                                    time_limit_seconds=self._time_limit_seconds,
                                    scorer=scorer)
        solving_end_time = time.perf_counter()
        solving_time = solving_end_time - solving_start_time

        self.results = {"best_ans": best_ans, "best_ans_score": best_ans_score, "solving_time": solving_time}
        self.baseline = {
            "d_tags": d_tags,
            "f_diff_list": f_diff_list
        }

    def test_results_report(self):
        """
        This method analyze the results of the algorithm, even more than one run and produces a result data structure
        """
        pass
