# library imports
from collections import Callable
import pandas as pd

# project imports
from explanation_analysis.afes.afes_metric import AfesMetric
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class AfesSum(AfesMetric):
    """
    A weighted sum of the tree components
    """

    def __init__(self,
                 sim_module: SimMetric,
                 w_gsim: float = 1,
                 w_lsim: float = 1,
                 w_ldiff: float = 1):
        AfesMetric.__init__(self,
                            sim_module=sim_module)
        self._w_gsim = w_gsim
        self._w_lsim = w_lsim
        self._w_ldiff = w_ldiff

    def compute(self,
                d: pd.DataFrame,
                s: list,
                f_sim: list,
                f_diff: list):
        global_sim, local_sim, local_diff = self.compute_parts(d=d, s=s, f_sim=f_sim, f_diff=f_diff)
        return global_sim + local_sim - local_diff

    def compute_parts(self,
                      d: pd.DataFrame,
                      s: list,
                      f_sim: list,
                      f_diff: list):
        global_sim = self._w_gsim * 1 / len(d) * sum(
            self.sim_module.sim(d=d, s=row, features=d.columns.values) for index, row in d.iterrows())
        local_sim = self._w_lsim * self.sim_module.sim(d=d, s=s, features=f_sim)
        local_diff = self._w_ldiff * self.sim_module.sim(d=d, s=s, features=f_diff)
        return global_sim, local_sim, local_diff
