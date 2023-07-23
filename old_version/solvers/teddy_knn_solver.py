# library imports
import numpy as np
import pandas as pd
from time import time
from sklearn.neighbors import KNeighborsRegressor

# project imports
from old_version.solvers.solver import Solver
from old_version.anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from old_version.explanation_analysis.score_function.score_function import ScoreFunction


class TeddyKnnSolver(Solver):
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
              scorer: ScoreFunction,
              save_conv=False) -> tuple:
        features = d.columns.values
        start_time = time()

        # check what is the best solution
        best_ans = None
        best_score = float('-inf')
        solution = {}
        f_diff_size = 1

        # run KNN on the D' for different samples of F_{diff} obtained from F_{diff}
        knn = KNeighborsRegressor(n_neighbors=self._k)
        knn.fit(X=d, y=list(range(d.shape[0])))  # the y is useless so we just put indexes, it can be any value
        rows_indexes = knn.kneighbors(X=[s], n_neighbors=self._k, return_distance=False)[0]
        d_tag_full_f = d.iloc[rows_indexes]
        f_diff_dist_vector = np.abs(np.array(s) - np.array(d_tag_full_f.mean(axis=0)))

        # run until the time is over
        while ((time() - start_time) < time_limit_seconds or best_ans is None) and (f_diff_size < d.shape[1]):
            f_sim = [features[i] for i in (-f_diff_dist_vector).argsort()[:f_diff_size]]
            f_sim.sort()
            f_diff = [feature for feature in features if feature not in f_sim]

            # obtain the D' with F_{diff}
            ans = d.iloc[rows_indexes]

            # score it
            current_score, scores = scorer.compute(d=ans, s=s, f_sim=f_sim, f_diff=f_diff, overall_size=len(d))

            # if best so far, replace and record
            if current_score > best_score:
                best_score = current_score
                solution = {'d_tag': ans,
                            'shape': (len(rows_indexes), len(f_diff)),
                            'f_diff': f_diff,
                            'f_sim': f_sim,
                            'best_score': current_score,
                            'self_sim': scores['self_sim'],
                            'local_sim': scores['local_sim'],
                            'sim_cluster': scores['sim_cluster_score'],
                            'local_diff': scores['local_diff'],
                            'diff_cluster': scores['sim_cluster_score'],
                            'coverage': scores['coverage'],
                            'conciseness': scores['conciseness']
                            }

            # count this try and try larger set
            f_diff_size += 1

        assoc = np.zeros(len(d), dtype=int)
        assoc[rows_indexes] = 1

        # return the best so far
        return solution, list(assoc)
