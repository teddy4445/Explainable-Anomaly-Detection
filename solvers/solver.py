# library imports
import pandas as pd

# project imports
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class Solver:
    """
    An abstract class for the project's solver classes
    """

    def __init__(self):
        self._param = None
        self.convert_process = {
            "time": [],
            "rows_indexes": [],
            "cols_indexes": [],
            "shape": [],
            "score": []
        }

    def solve(self,
              anomaly_algo: AnomalyAlgo,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: float,
              scorer: AfesMetric) -> tuple:
        pass

    def get_convergence_process(self) -> list:
        return self.convert_process
