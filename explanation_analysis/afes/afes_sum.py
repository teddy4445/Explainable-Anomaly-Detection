# library imports
from collections import Callable
import pandas as pd

# project imports
from explanation_analysis.afes.afes_metric import AfesMetric


class AfesSum(AfesMetric):
    """
    A weighted sum of the tree components
    """

    def __init__(self,
                 w_gsim: float = 1,
                 w_lsim: float = 1,
                 w_ldiff: float = 1):
        self._w_gsim = w_gsim
        self._w_lsim = w_lsim
        self._w_ldiff = w_ldiff

    def afes(self,
             d: pd.DataFrame,
             s: pd.Series,
             f_sim: list,
             f_diff: list,
             sim: Callable):
        global_sim = self._w_gsim * sum(sim(d=d, s=row, f_sim=f_sim, f_diff=None) for index, row in d.iterrows())
        local_sim = self._w_lsim * sim(d=d, s=s, f_sim=f_sim, f_diff=None)
        local_diff = self._w_ldiff * sim(d=d, s=s, f_sim=f_diff, f_diff=None)
        return global_sim + local_sim + local_diff
