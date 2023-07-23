# library imports
import random
from time import time
from tqdm import tqdm

# project imports
from solvers.solver import Solver


class MonteCarloSolver(Solver):
    """
    A Monte Carlo approach for the rows (D') and columns (F_{diff})
    """

    def __init__(self, scorer, data, anomaly, time_limit_seconds):
        super().__init__(scorer=scorer, data=data, anomaly=anomaly)
        self.time_limit_seconds = time_limit_seconds

    def solve(self) -> tuple:
        start_time = time()

        # run until the time is over
        with tqdm(total=self.time_limit_seconds, desc=f"{'time [sec]':<20}") as pbar:
            while (time() - start_time) < self.time_limit_seconds:
                pbar.update((time() - start_time) - pbar.n)
                # pick rows and cols for D' at random
                rows_indexes = random.sample(list(range(self.dataset_size)), random.randint(1, self.dataset_size))
                cols_indexes = random.sample(self.features, random.randint(1, self.features_num))
                self.evaluate(rows_indexes=rows_indexes, f_diff=cols_indexes)

        self._add_informed_dataframe()
        return self.solution
