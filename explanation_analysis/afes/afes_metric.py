# library imports
from collections import Callable
import pandas as pd

# project imports
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class AfesMetric:
    """
    TODO: noam please add here
    """

    def __init__(self,
                 sim_module: SimMetric):
        self.sim_module = sim_module

    def compute(self,
                d: pd.DataFrame,
                s: list,
                f_sim: list,
                f_diff: list):
        pass

    def compute_parts(self,
                      d: pd.DataFrame,
                      s: list,
                      f_sim: list,
                      f_diff: list):
        pass
