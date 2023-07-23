# library imports
from collections import Callable
import pandas as pd

# project imports
from scorers.similarity_metrics.sim_metric import SimMetric


class ScoreFunction:
    TRACKED_SCORES = [
        "self_sim",
        "local_sim",
        "sim_cluster",
        "local_diff",
        "diff_cluster",
        "coverage",
        "conciseness"
    ]

    def __init__(self,
                 sim_module: SimMetric):
        self.sim_module = sim_module

    def compute(self,
                overall_size: int,
                d: pd.DataFrame,
                s: list,
                f_sim: list,
                f_diff: list):
        pass

    def compute_parts(self,
                      overall_size: int,
                      d: pd.DataFrame,
                      s: list,
                      f_sim: list,
                      f_diff: list):
        pass
