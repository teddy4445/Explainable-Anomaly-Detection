# library imports
import pandas as pd
from sklearn.metrics import f1_score

# project imports
from consts import *


class Solver:
    def solve(self, d: pd.DataFrame, s: list, time_limit_seconds: int) -> pd.DataFrame:
        pass

    def get_convergence_process(self) -> list:
        pass


class MonteCarloSolver(Solver):
    def __init__(self, param=1):
        self._param = param

    def solve(self, d: pd.DataFrame, s: list, time_limit_seconds: int) -> pd.DataFrame:
        pass

    def get_convergence_process(self) -> list:
        pass


class KnnSolver(Solver):
    def __init__(self, param=1):
        self._param = param

    def solve(self, d: pd.DataFrame, s: list, time_limit_seconds: int) -> pd.DataFrame:
        pass

    def get_convergence_process(self) -> list:
        pass
