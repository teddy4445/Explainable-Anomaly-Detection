# library imports
from __future__ import annotations
import numpy as np
import pandas as pd

# project imports
from old_version.explanation_analysis.similarity_metrices.sim_metric import SimMetric


class InverseEuclideanSim(SimMetric):
    """
    Inverse Euclidean similarity between a vector and a matrix
    """

    # CONSTS #
    NAME = "InverseEuclideanSim"

    def sim(self,
            d: list | pd.DataFrame,
            s: list | pd.Series,
            features: list,
            mode: str):
        if isinstance(s, pd.Series):
            s = s[features].to_numpy()
        if isinstance(d, pd.DataFrame):
            if 0 in d.shape:
                return 0
            d = d[features].to_numpy()
        elif isinstance(d, list) and len(d) > 0 and len(d[0]) > 0:
            d = np.array(d[features]).mean(0)
        else:
            raise ValueError("The argument 'd' must be either a 2-dim list or a pd.DataFrame")

        dist_array = np.linalg.norm(d - s, axis=1)
        # at this stage both 's' and 'd' are lists
        if len(dist_array) == 0:
            return 0
        if mode == 'max':
            if len(features) == 0:
                return 0
            return 1 / (1 + np.max(dist_array))
        elif mode == 'min':
            if len(features) == 0:
                return 1
            return 1 / (1 + np.min(dist_array))
