# library imports
from __future__ import annotations
import numpy as np
import pandas as pd

# project imports
from explanation_analysis.similarity_metrices.sim_metric import SimMetric


class MeanEntropySim(SimMetric):
    """
    The mean entropy distance between a vector and a matrix.
    Note: this metric is assuming the data is probelities and won't work eitherwise
    """

    def sim(self,
            d: list | pd.DataFrame,
            s: list | pd.Series,
            f_sim: list,
            f_diff: list):
        if isinstance(s, pd.Series):
            s = list(s)
            s = np.array(s)
        elif isinstance(s, list):
            s = np.array(s)
        else:
            raise ValueError("The argument 's' must be either a non-empty 1-dim list or a pd.Series")
        if isinstance(d, pd.DataFrame):
            pass
        elif isinstance(d, list) and len(d) > 0 and len(d[0]) > 0:
            d = pd.DataFrame(d)
        else:
            raise ValueError("The argument 'd' must be either a non-empty 2-dim list or a pd.DataFrame")

        # at this stage both 's' and 'd' are lists
        s = np.array(s)

        ans = 0
        counter = 0
        for row_index, row in d.iterrows():
            ans += -np.sum(s * np.log(row) + (1 - s) * np.log(1 - row))
            counter += 1
        return ans/counter
