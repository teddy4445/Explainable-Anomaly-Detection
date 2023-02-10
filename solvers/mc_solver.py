# library imports
import pandas as pd
from solvers.solver import Solver


class MonteCarloSolver(Solver):
    """
    A Monte Carlo approach for the rows (D') and columns (F_{diff})
    """

    def __init__(self):
        self._param = None

    def solve(self,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: int) -> pd.DataFrame:
        pass

    def get_convergence_process(self) -> list:
        pass
