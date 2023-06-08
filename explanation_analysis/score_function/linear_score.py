# library imports
from collections import Callable
import pandas as pd

# project imports
from explanation_analysis.score_function.score_function import ScoreFunction
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class LinearScore(ScoreFunction):
    """
    A weighted sum of the tree components
    """

    def __init__(self,
                 sim_module: SimMetric,
                 w_self_sim: float = 1.0,
                 w_local_sim: float = 1.0,
                 w_cluster_sim: float = 1.0,
                 w_local_diff: float = 1.0,
                 w_cluster_diff: float = 1.0,
                 w_cov: float = 0):
        ScoreFunction.__init__(self, sim_module=sim_module)
        self.w_self_sim = w_self_sim
        self.w_local_sim = w_local_sim
        self.w_cluster_sim = w_cluster_sim
        self.w_local_diff = w_local_diff
        self.w_cluster_diff = w_cluster_diff
        self.w_cov = w_cov

    def compute(self,
                overall_size: int,
                d: pd.DataFrame,
                s: list,
                f_sim: list,
                f_diff: list):
        scores = self.compute_parts(d=d, s=s, f_sim=f_sim, f_diff=f_diff, overall_size=overall_size)
        score = self.w_self_sim * scores["self_sim"] \
                + self.w_local_sim * scores["local_sim"] + self.w_cluster_sim * scores["sim_cluster_score"] \
                - self.w_local_diff * scores["local_diff"] + self.w_cluster_diff * scores["diff_cluster_score"] \
                + self.w_cov * scores["coverage"]
        return score, scores

    def compute_parts(self,
                      overall_size: int,
                      d: pd.DataFrame,
                      s: list,
                      f_sim: list,
                      f_diff: list):
        plain_d = d.reset_index(drop=True)

        # similarity scores
        self_sim = 1 / len(d) * sum(
            self.sim_module.sim(d=pd.concat([plain_d.iloc[:index], plain_d.iloc[index + 1:]], ignore_index=True),
                                s=row, features=d.columns.values, mode='max')
            for index, row in plain_d.iterrows()
        )
        local_sim = self.sim_module.sim(d=d, s=s, features=f_sim, mode='max')
        local_diff = self.sim_module.sim(d=d, s=s, features=f_diff, mode='min')

        # cluster scores
        sim_cluster_score = 0  # TODO: Add cluster score
        diff_cluster_score = 0  # TODO: Add cluster score

        # coverage score
        coverage = self.w_cov * len(d) / overall_size

        scores = {
            "self_sim": self_sim,
            "local_sim": local_sim,
            "sim_cluster_score": sim_cluster_score,
            "local_diff": local_diff,
            "diff_cluster_score": diff_cluster_score,
            "coverage": coverage
        }
        return scores
