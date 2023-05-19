# library imports
import pandas as pd

# project imports
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class Solver:
    """
    An abstract class for the project's solver classes
    """

    def __init__(self,
                 param: dict = None):
        self._param = param if param is not None else {}
        self.convert_process = {
            "time": [],
            "rows_indexes": [],
            "f_diff": [],
            "shape": [],
            "self_sim": [],
            "local_sim": [],
            "local_diff": [],
            "coverage": [],
            "score": []
        }

    def solve(self,
              anomaly_algo: AnomalyAlgo,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: float,
              scorer: AfesMetric) -> tuple:
        pass

    def close_convergence_process(self, time_limit_seconds):
        if not len(self.convert_process["time"]):
            pass
        elif self.convert_process["time"][-1] < time_limit_seconds:
            self.convert_process["time"].append(time_limit_seconds)
            for metric in self.convert_process.keys():
                self.convert_process[metric].append(self.convert_process[metric][-1])
