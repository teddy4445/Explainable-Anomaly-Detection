# library imports
from collections import Callable
import pandas as pd


class AfesMetric:
    """
    TODO: noam please add here
    """

    def __init__(self,
                 sim: Callable):
        self.sim = sim

    def compute(self,
                d: pd.DataFrame,
                s: list,
                f_sim: list,
                f_diff: list):
        pass

    def compute_all_features(self,
                             d: pd.DataFrame,
                             s: list):
        pass
