# library imports
from __future__ import annotations
import pandas as pd


class SimMetric:
    """
    An abstract class for a similarity used in this project
    """

    def __init__(self):
        pass

    def sim(self,
            d: list | pd.DataFrame,
            s: list | pd.pd.Series,
            features: list):
        pass
