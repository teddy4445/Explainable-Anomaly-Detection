# library imports
import os
from typing import Callable
import numpy as np
import pandas as pd

# project imports
from consts import *
from sim_functions import ProbSim
from afes_functions import SumAfes


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


if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(0,100,size=(10, 4)), columns=['A', 'B', 'C', 'D'])
    # print(df.head())
    data_point = df.iloc[0]
    # print(data_point)

    features_sim = ['A', 'B']
    features_diff = ['C', 'D']

    b = ExplanationAnalysis.score(d=df, s=data_point, f_sim=features_sim, f_diff=features_diff,
                                  afes=SumAfes().afes, sim=ProbSim().sim)
    print(b)
