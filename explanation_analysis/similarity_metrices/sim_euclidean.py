# library imports
from __future__ import annotations
import numpy as np
import pandas as pd

# project imports
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class EuclideanSim(SimMetric):
    """
    Euclidean similarity between a vector and a matrix
    """

    # CONSTS #
    NAME = "EuclideanSim"

    def sim(self,
            d: list | pd.DataFrame,
            s: list | pd.Series,
            features: list,
            mode: str):
        if isinstance(s, pd.Series):
            s = s[features].values
        if isinstance(d, pd.DataFrame):
            d = d.mean(axis=0)[features].values
        elif isinstance(d, list) and len(d) > 0 and len(d[0]) > 0:
            d = np.array(d[features]).mean(0)
        else:
            raise ValueError("The argument 'd' must be either a 2-dim list or a pd.DataFrame")

        # at this stage both 's' and 'd' are lists
        return - np.linalg.norm(np.array(d) - np.array(s))
