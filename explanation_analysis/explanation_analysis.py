# library imports
import pandas as pd
from typing import Callable

# project imports


class ExplanationAnalysis:
    """
    A virtual class of an experiment_properties
    """

    # def __init__(self):
    #     self.results = {}

    @staticmethod
    def score(d: pd.DataFrame,
              s: list,
              f_sim: list,
              f_diff: list,
              afes: Callable,
              sim: Callable) -> float:
        """
        This method
        """
        return afes(d=d,
                    s=s,
                    f_sim=f_sim,
                    f_diff=f_diff,
                    sim=sim)

