# library imports
from __future__ import annotations
import numpy as np
import pandas as pd

# project imports
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class InverseMinSim(SimMetric):
    """
    Inverse Euclidean similarity between a vector and a matrix
    """

    # CONSTS #
    NAME = "InverseMinSim"

    def sim(self,
            d: list | pd.DataFrame,
            s: list | pd.Series,
            features: list):
        if isinstance(s, pd.Series):
            s = s[features].to_numpy()
        if isinstance(d, pd.DataFrame):
            d = d[features].to_numpy()
        elif isinstance(d, list) and len(d) > 0 and len(d[0]) > 0:
            d = np.array(d[features]).mean(0)
        else:
            raise ValueError("The argument 'd' must be either a 2-dim list or a pd.DataFrame")

        dist_array = np.linalg.norm(d - s, axis=1)
        # at this stage both 's' and 'd' are lists
        return 1 / (1 + np.min(dist_array))
