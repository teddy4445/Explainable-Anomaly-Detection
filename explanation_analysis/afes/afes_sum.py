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
                 sim: Callable,
                 w_gsim: float = 1,
                 w_lsim: float = 1,
                 w_ldiff: float = 1):
        AfesMetric.__init__(self,
                            sim=sim)
        self._w_gsim = w_gsim
        self._w_lsim = w_lsim
        self._w_ldiff = w_ldiff

    def compute(self,
                d: pd.DataFrame,
                s: list,
                f_sim: list,
                f_diff: list):
        global_sim = self._w_gsim * sum(self.sim(d=d, s=row, f_sim=f_sim, f_diff=f_diff) for index, row in d.iterrows())
        local_sim = self._w_lsim * self.sim(d=d, s=s, f_sim=f_sim, f_diff=f_diff)
        local_diff = self._w_ldiff * self.sim(d=d, s=s, f_sim=f_sim, f_diff=f_diff)
        return global_sim + local_sim + local_diff

    def compute_all_features(self,
                             d: pd.DataFrame,
                             s: list):
        global_sim = self._w_gsim * sum(self.sim(d=d, s=row, f_sim=list(range(d.shape[1])), f_diff=None) for index, row in d.iterrows())
        local_sim = self._w_lsim * self.sim(d=d, s=s, f_sim=list(range(d.shape[1])), f_diff=None)
        local_diff = self._w_ldiff * self.sim(d=d, s=s, f_sim=list(range(d.shape[1])), f_diff=None)
        return global_sim + local_sim + local_diff
