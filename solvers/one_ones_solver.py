# library imports
import random
import pandas as pd
from time import time

# project imports
from solvers.solver import Solver
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class OneOneSolver(Solver):
    """
    A solver that tries to get the best row and column each time
    """

    def __init__(self,
                 d_tag_size: int,
                 f_diff_size: int):
        Solver.__init__(self)
        self.d_tag_size = d_tag_size
        self.f_diff_size = f_diff_size

    def solve(self,
              d: pd.DataFrame,
              anomaly_algo: AnomalyAlgo,
              s: list,
              time_limit_seconds: int,
              scorer: AfesMetric) -> tuple:
        start_time = time()
        # check what is the best solution
        is_row = True
        rows = []
        cols = []
        # run until the time is over
        for index in range(self.d_tag_size + self.f_diff_size):
            # just to break over time
            if (time() - start_time) > time_limit_seconds:
                break
            if is_row:
                best_row_index = 0
                best_row_score = 0
                for row_index, row in d.iterrows():
                    this_rows = rows.copy()
                    this_rows.append(row_index)
                    score = scorer.compute_all_features(d.iloc[rows, cols], s[cols])
                    if best_row_score < score:
                        best_row_score = score
                        best_row_index = row_index
                cols.append(best_row_index)
            else:
                best_col_index = 0
                best_col_score = 0
                for col_index in range(d.shape[1]):
                    this_rows = cols.copy()
                    this_rows.append(col_index)
                    score = scorer.compute_all_features(d.iloc[rows, cols], s[cols])
                    if best_col_score < score:
                        best_col_score = score
                        best_col_index = row_index
                cols.append(best_col_index)
            self.convert_process.append({
                "rows_indexes": rows,
                "cols_indexes": cols,
                "score": scorer.compute_all_features(d.iloc[rows, cols], s[cols])}
            )
        # return the best so far
        ans = d.iloc[rows, cols]
        return ans, scorer.compute_all_features(ans, s[cols])
