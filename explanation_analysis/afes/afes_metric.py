# library imports
from collections import Callable
import pandas as pd


class AfesMetric:
    """
    An abstract class for a similarity used in this project
    """

    def afes(self,
             d: pd.DataFrame,
             s: pd.Series,
             f_sim: list,
             f_diff: list,
             sim: Callable):
        pass
