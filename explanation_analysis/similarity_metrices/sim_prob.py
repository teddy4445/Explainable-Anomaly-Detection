# library imports
from __future__ import annotations
import pandas as pd

# project imports
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class ProbSim(SimMetric):
    """
    TODO: noam add later
    """

    def sim(self,
            d: list | pd.pd.DataFrame,
            s: list | pd.pd.Series,
            features: list,
            mode: str):
        reduced_d = d.copy()
        for feature in features:
            reduced_d = reduced_d.loc[reduced_d[feature] == s[feature]]
        return len(reduced_d) / len(d)
