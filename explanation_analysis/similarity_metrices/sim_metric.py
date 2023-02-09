# library imports
import pandas as pd


class SimMetric:
    """
    An abstract class for a similarity used in this project
    """

    def sim(self,
            d: pd.DataFrame,
            s: pd.Series,
            f_sim: list,
            f_diff: list):
        pass
