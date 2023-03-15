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
                 param: dict = None):
        if param is None:
            param = {"k": 3}
        Solver.__init__(self,
                        param=param)
        if isinstance(param["k"], int) and param["k"] > 0:
            self._k = param["k"]
        else:
            raise Exception("KnnSovler.__init__ error saying that 'k' is positive integer")

    def solve(self,
              anomaly_algo: AnomalyAlgo,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: int,
              scorer: AfesMetric) -> tuple:
        features = d.columns.values
        start_time = time()
        # check what is the best solution
        best_ans = None
        best_ans_score = 0
        f_diff_size = 1
        # run KNN on the D' for different samples of F_{diff} obtained from F_{diff}
        knn = KNeighborsRegressor(n_neighbors=self._k)
        knn.fit(X=d, y=list(range(d.shape[0])))  # the y is useless so we just put indexes, it can be any value
        rows_indexes = knn.kneighbors(X=[s],
                                      n_neighbors=self._k,
                                      return_distance=False)[0]
        d_tag_full_f = d.iloc[rows_indexes]
        f_diff_dist_vector = np.abs(np.array(s) - np.array(d_tag_full_f.mean(axis=0)))
        # run until the time is over
        while ((time() - start_time) < time_limit_seconds or best_ans is None) and (f_diff_size < d.shape[1]):
            cols_indexes = (-f_diff_dist_vector).argsort()[:f_diff_size]
            # obtain the D' with F_{diff}
            ans = d.iloc[rows_indexes]
            # score it
            score = scorer.compute(d=ans, s=s, f_sim=cols_indexes,
                                   f_diff=[feature for feature in features if feature not in cols_indexes])
            # if best so far, replace and record
            if score > best_ans_score:
                best_ans_score = score
                best_ans = ans

            self.convert_process["time"].append(time() - start_time)
            self.convert_process["rows_indexes"].append(rows_indexes)
            self.convert_process["cols_indexes"].append(cols_indexes)
            self.convert_process["shape"].append([len(rows_indexes), len(cols_indexes)])
            self.convert_process["score"].append(best_ans_score)

            # count this try and try larger set
            f_diff_size += 1

        if self.convert_process["time"][-1] < 60:
            self.convert_process["time"].append(60.0)
            self.convert_process["rows_indexes"].append(self.convert_process["rows_indexes"][-1])
            self.convert_process["cols_indexes"].append(self.convert_process["cols_indexes"][-1])
            self.convert_process["shape"].append(self.convert_process["shape"][-1])
            self.convert_process["score"].append(best_ans_score)

        # return the best so far
        return best_ans, best_ans_score
