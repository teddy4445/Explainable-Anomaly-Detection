# library imports
from __future__ import annotations
import pandas as pd


class SimMetric:
    """
    An abstract class for a similarity used in this project
    """

    def sim(self,
            d: list | pd.DataFrame,
            s: list | pd.pd.Series,
            f_sim: list,
            f_diff: list):
        pass
