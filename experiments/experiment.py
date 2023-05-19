# library imports
import os
import time
import pandas as pd

# project imports
from solvers.solver import Solver
from explanation_analysis.afes.afes_metric import AfesMetric
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo

TRACKED_METRICS = {'d_tag', 'shape', 'f_diff', 'f_sim', 'best_score', 'best_ss', 'best_ls', 'best_ld', 'best_cov'}


class Experiment:
    """
    A virtual class of an experiment_properties
    """

    def __init__(self,
                 time_limit_seconds: float):
        self.results = {}
        self.convert_process = None
        self._time_limit_seconds = time_limit_seconds
        self.results = {'d_tag': None,
                        'f_diff': None,
                        'f_sim': None,
                        'best_score': None,
                        'best_ss': None,
                        'best_ls': None,
                        'best_ld': None,
                        'best_cov': None}

    def run(self,
            anomaly_algo: AnomalyAlgo,
            solver: Solver,
            scorer: AfesMetric,
            dataset: pd.DataFrame,
            anomaly_sample: list,
            save_conv=False):
        """
        This method runs an algorithm on the experiment_properties's data and stores the results needed for this experiment_properties
        """
        solving_start_time = time.perf_counter()
        results, assoc = solver.solve(d=dataset,
                                      anomaly_algo=anomaly_algo,
                                      s=anomaly_sample,
                                      time_limit_seconds=self._time_limit_seconds,
                                      scorer=scorer)
        solving_time = time.perf_counter() - solving_start_time

        # Save results
        for k, v in results.items():
            self.results[k] = v
        self.results["solving_time"] = solving_time

        # Save d_inf
        d_inf = pd.concat([dataset, pd.DataFrame(anomaly_sample).T], ignore_index=True)
        assoc.append(2)
        d_inf['assoc'] = assoc
        self.results["d_inf"] = d_inf

        # Save conversion process
        if save_conv:
            self.convert_process = solver.convert_process

    def test_results_report(self):
        """
        This method analyze the results of the algorithm, even more than one run and produces a result data structure
        """
        pass
