# library imports
import os
import numpy as np
import pandas as pd
from typing import Callable

# project imports
from consts import *
from experiment_synthetic.sim_functions import ProbSim
from experiment_synthetic.afes_functions import SumAfes


class ExplanationAnalysis:
    """
    A virtual class of an experiment
    """

    # def __init__(self):
    #     self.results = {}

    @staticmethod
    def score(d: pd.DataFrame, s: list, f_sim: list, f_diff: list,
              afes: Callable, sim: Callable) -> float:
        """
        This method
        """

        return afes(d=d, s=s, f_sim=f_sim, f_diff=f_diff, sim=sim)

