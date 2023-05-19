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
                 w_ldiff: float = 1,
                 w_cov: float = 1):
        AfesMetric.__init__(self,
                            sim_module=sim_module)
        self._w_gsim = w_gsim
        self._w_lsim = w_lsim
        self._w_ldiff = w_ldiff
        self._w_cov = w_cov

    def compute(self,
                overall_size: int,
                d: pd.DataFrame,
                s: list,
                f_sim: list,
                f_diff: list):
        self_sim, local_sim, local_diff, coverage = self.compute_parts(d=d, s=s, f_sim=f_sim, f_diff=f_diff,
                                                                       overall_size=overall_size)
        return self_sim + local_sim - local_diff + coverage

    def compute_parts(self,
                      overall_size: int,
                      d: pd.DataFrame,
                      s: list,
                      f_sim: list,
                      f_diff: list):
        plain_d = d.reset_index(drop=True)
        self_sim = self._w_gsim * 1 / len(d) * sum(
            self.sim_module.sim(d=pd.concat([plain_d.iloc[:index], plain_d.iloc[index + 1:]], ignore_index=True), s=row,
                                features=d.columns.values, mode='sim') for index, row in plain_d.iterrows())
        local_sim = self._w_lsim * self.sim_module.sim(d=d, s=s, features=f_sim, mode='sim')
        local_diff = self._w_ldiff * self.sim_module.sim(d=d, s=s, features=f_diff, mode='diff')
        coverage = self._w_cov * len(d) / overall_size
        return self_sim, local_sim, local_diff, coverage
