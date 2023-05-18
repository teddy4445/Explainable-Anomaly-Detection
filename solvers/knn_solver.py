# library imports
import itertools

import numpy as np
import pandas as pd
from time import time
from sklearn.neighbors import KNeighborsRegressor

# project imports
from tqdm import tqdm

from solvers.solver import Solver
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class KnnSolver(Solver):
    """
    A KNN approach
    """

    def __init__(self,
                 param: dict = None):
        Solver.__init__(self,
                        param=param)
        self._k = param.get('k', 3)
        self.f_diff = param.get('f_diff', None)
        self.f_diff_size = param.get('f_diff_size', None)

    def solve(self,
              anomaly_algo: AnomalyAlgo,
              d: pd.DataFrame,
              s: list,
              time_limit_seconds: int,
              scorer: AfesMetric,
              save_conv=False) -> tuple:
        features = d.columns.values

        # run KNN and find top-k
        knn = KNeighborsRegressor(n_neighbors=self._k)
        knn.fit(X=d, y=list(range(d.shape[0])))  # the y is useless so we just put indexes, it can be any value
        rows_indexes = knn.kneighbors(X=[s], n_neighbors=self._k, return_distance=False)[0]
        d_tag_full_f = d.iloc[rows_indexes]
        start_time = time()

        if self.f_diff:
            ans_fdiff = self.f_diff
            f_sim = [feature for feature in features if feature not in ans_fdiff]

            best_ans_score = scorer.compute(d=d_tag_full_f, s=s, f_sim=f_sim, f_diff=ans_fdiff, overall_size=len(d))
            global_sim, local_sim, local_diff, coverage = scorer.compute_parts(d=d_tag_full_f, s=s,
                                                                               f_sim=f_sim, f_diff=ans_fdiff,
                                                                               overall_size=len(d))

            solution = {'d_tag': d_tag_full_f,
                        'shape': (len(d_tag_full_f), len(self.f_diff)),
                        'f_diff': self.f_diff,
                        'f_sim': f_sim,
                        'best_score': best_ans_score,
                        'best_gs': global_sim,
                        'best_ls': local_sim,
                        'best_ld': local_diff,
                        'best_cov': coverage}

            # save conversion process
            if save_conv:
                self.convert_process["time"].append(time() - start_time)
                self.convert_process["rows_indexes"].append(rows_indexes)
                self.convert_process["cols_indexes"].append(self.f_diff)
                self.convert_process["shape"].append([len(rows_indexes), len(self.f_diff)])
                self.convert_process["score"].append(best_ans_score)
                self.convert_process["global_sim"].append(global_sim)
                self.convert_process["local_sim"].append(local_sim)
                self.convert_process["local_diff"].append(local_diff)
                self.convert_process["coverage"].append(coverage)

        else:
            best_ans_score = -99999999

            # run until the time is over
            # while (time() - start_time) < time_limit_seconds or best_ans is None:
            for f_diff_size in range(1, d.shape[1] + 1):
                subsets_cols = list(map(set, itertools.combinations(list(d.columns.values), f_diff_size)))
                for cols_indexes in subsets_cols:
                    f_diff = list(cols_indexes)
                    f_sim = [feature for feature in features if feature not in f_diff]
                    # score it
                    score = scorer.compute(d=d_tag_full_f, s=s, f_sim=f_sim, f_diff=f_diff, overall_size=len(d))
                    global_sim, local_sim, local_diff, coverage = scorer.compute_parts(d=d_tag_full_f, s=s,
                                                                                       f_sim=f_sim, f_diff=f_diff,
                                                                                       overall_size=len(d))

                    # if best so far, replace and record
                    if score > best_ans_score:
                        best_ans_score = score
                        solution = {'d_tag': d_tag_full_f,
                                    'shape': (len(d_tag_full_f), len(f_diff)),
                                    'f_diff': f_diff,
                                    'f_sim': f_sim,
                                    'best_score': score,
                                    'best_gs': global_sim,
                                    'best_ls': local_sim,
                                    'best_ld': local_diff,
                                    'best_cov': coverage}

                    # save conversion process
                    if save_conv:
                        self.convert_process["time"].append(time() - start_time)
                        self.convert_process["rows_indexes"].append(rows_indexes)
                        self.convert_process["cols_indexes"].append(cols_indexes)
                        self.convert_process["shape"].append([len(rows_indexes), len(cols_indexes)])
                        self.convert_process["score"].append(score)
                        self.convert_process["global_sim"].append(global_sim)
                        self.convert_process["local_sim"].append(local_sim)
                        self.convert_process["local_diff"].append(local_diff)
                        self.convert_process["coverage"].append(coverage)

        if save_conv:
            self.close_convergence_process(time_limit_seconds=time_limit_seconds)
        assoc = np.zeros(len(d), dtype=int)
        assoc[rows_indexes] = 1

        # return the best so far
        return solution, list(assoc)
