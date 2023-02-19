# library imports
import numpy as np
import pandas as pd
from time import time
from sklearn.neighbors import KNeighborsRegressor

# project imports
from solvers.solver import Solver
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class KnnSolver(Solver):
    """
    A KNN approach for the rows (D') and a Top-k approach for F_{diff} such that k is searched using a grid search
    """

    def __init__(self,
                 k: int = 3):
        Solver.__init__(self)
        if isinstance(k, int) and k > 0:
            self._k = k
        else:
            raise Exception("KnnSovler.__init__ error saying that 'k' is possitive integer")

    def solve(self,
              anomaly_algo: AnomalyAlgo,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: int,
              scorer: AfesMetric) -> tuple:
        start_time = time()
        # check what is the best solution
        best_ans = None
        best_ans_score = 99999
        f_diff_size = 1
        # run KNN on the D' for different samples of F_{diff} obtained from F_{diff}
        knn = KNeighborsRegressor(n_neighbors=self._k)
        knn.fit(d, list(range(d.shape[0])))  # the y is useless so we just put indexes, it can be any value
        rows_indexes = knn.kneighbors(X=[s],
                                      n_neighbors=self._k,
                                      return_distance=False)
        d_tag_full_f = d[rows_indexes, :]
        f_diff_dist_vector = np.abs(np.array(s) - np.array(d_tag_full_f.mean(axis=0)))
        # run until the time is over
        while ((time() - start_time) < time_limit_seconds or best_ans is None) and (f_diff_size < d.shape[1]):
            cols_indexes = (-f_diff_dist_vector).argsort()[:f_diff_size]
            # obtain the D' with F_{diff}
            ans = d.iloc[rows_indexes, cols_indexes]
            # score it
            score = scorer.compute_all_features(ans, s)
            # if best so far, replace and record
            if score < best_ans_score:
                best_ans_score = score
                best_ans = ans
                self.convert_process.append({
                    "rows_indexes": rows_indexes,
                    "cols_indexes": cols_indexes,
                    "score": score}
                )
            # count this try and try larger set
            f_diff_size += 1
            # return the best so far
        return best_ans, best_ans_score
