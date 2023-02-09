# library imports
import pandas as pd
from solvers.solver import Solver


class KnnSolver(Solver):
    """
    A KNN approach for the rows (D') and a Top-k approach for F_{diff} such that k is searched using a grid search
    """

    def __init__(self,
                 param=1):
        self._param = param

    def solve(self,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: int) -> pd.DataFrame:
        pass

    def get_convergence_process(self) -> list:
        pass
