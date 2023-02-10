# library imports
from collections import Callable

import pandas as pd


class Solver:
    """
    An abstract class for the project's solver classes
    """

    def __init__(self):
        self._param = None
        self.convert_process = []

    def solve(self,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: int,
              scorer: Callable) -> tuple:
        pass

    def get_convergence_process(self) -> list:
        return self.convert_process
