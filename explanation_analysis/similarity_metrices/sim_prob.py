# library imports
import pandas as pd

# project imports
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class ProbSim(SimMetric):
    """
    TODO: noam add later
    """

    def sim(self,
            d: pd.DataFrame,
            s: pd.Series,
            f_sim: list,
            f_diff: list):
        reduced_d = d
        for feature in f_sim:
            reduced_d = reduced_d.loc[reduced_d[feature] == s[feature]]
        return len(reduced_d) / len(d)
