# library imports
import random
import pandas as pd
from time import time

# project imports
from solvers.solver import Solver
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class GeneticAlgorithmSolver(Solver):
    """
    A solver that uses genetic algorithm to solve the task
    """

    def __init__(self,
                 generations: int,
                 mutation_rate: float,
                 royalty_rate: float,
                 population_size: int):
        Solver.__init__(self)
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.royalty_rate = royalty_rate
        self.population_size = population_size

    def solve(self,
              d: pd.DataFrame,
              anomaly_algo: AnomalyAlgo,
              s: list,
              time_limit_seconds: int,
              scorer: AfesMetric) -> tuple:
        start_time = time()
        self.convert_process.append({
            "rows_indexes": rows,
            "cols_indexes": cols,
            "score": scorer.compute_all_features(d.iloc[rows, cols],
                                                 s)}
        )
        # return the best so far
        ans = d.iloc[rows, cols]
        return ans, scorer.compute_all_features(ans, s)
