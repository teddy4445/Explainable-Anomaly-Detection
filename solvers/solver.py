# library imports
import pandas as pd


class Solver:
    """
    An abstract class for the project's solver classes
    """

    def solve(self,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: int) -> pd.DataFrame:
        pass

    def get_convergence_process(self) -> list:
        pass
