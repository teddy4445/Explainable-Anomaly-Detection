# library imports
import os
import pandas as pd

# project imports
from consts import *


class SimWrapper:
    def sim(self, d: pd.DataFrame, s: pd.Series, f_sim: list, f_diff: list):
        pass


class ProbSim(SimWrapper):
    def __init__(self, a=1):
        self._a = a

    def sim(self, d: pd.DataFrame, s: pd.Series, f_sim: list, f_diff: list):
        reduced_d = d
        for feature in f_sim:
            reduced_d = reduced_d.loc[reduced_d[feature] == s[feature]]

        return len(reduced_d) / len(d)
