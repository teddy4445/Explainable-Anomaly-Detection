# library imports
import random
import itertools
import numpy as np
import pandas as pd
from time import time
from itertools import combinations, chain

# project imports
from solvers.solver import Solver
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class BruteForceSolver(Solver):
    """
    A Brute Force approach for the rows (D') and columns (F_{diff})
    """

    def __init__(self,
                 param: dict = None):
        Solver.__init__(self,
                        param=param)

    def solve(self,
              d: pd.DataFrame,
              anomaly_algo: AnomalyAlgo,
              s: list,
              time_limit_seconds: int,
              scorer: AfesMetric) -> tuple:
        features = d.columns.values
        start_time = time()
        # check what is the best solution
        best_ans = None
        best_ans_score = -99999999
        # run until the time is over
        for d_tag_size in range(1, d.shape[0]-1):
            for f_diff_size in range(1, d.shape[1]-1):
                subsets_rows = list(map(set, itertools.combinations(list(range(d.shape[0])), d_tag_size)))
                subsets_cols = list(map(set, itertools.combinations(list(range(d.shape[1])), f_diff_size)))
                for rows_indexes in subsets_rows:
                    for cols_indexes in subsets_cols:
                        rows_indexes = list(rows_indexes)
                        cols_indexes = list(cols_indexes)
                        ans = d.loc[rows_indexes]
                        # score it
                        score = scorer.compute(d=ans, s=s, f_sim=cols_indexes,
                                               f_diff=[feature for feature in features if feature not in cols_indexes],
                                               overall_size=len(d))
                        global_sim, local_sim, local_diff, coverage = scorer.compute_parts(d=ans, s=s, f_sim=cols_indexes,
                                                                                           f_diff=[feature for feature in features
                                                                                                   if feature not in cols_indexes],
                                                                                           overall_size=len(d))

                        # if best so far, replace and record
                        if score > best_ans_score:
                            # best_score = score
                            best_ans = ans
                            scores = {'best_score': score,
                                      'best_gs': global_sim,
                                      'best_ls': local_sim,
                                      'best_ld': local_diff,
                                      'best_cov': coverage}

                        self.convert_process["time"].append(time() - start_time)
                        self.convert_process["rows_indexes"].append(rows_indexes)
                        self.convert_process["cols_indexes"].append(cols_indexes)
                        self.convert_process["shape"].append([len(rows_indexes), len(cols_indexes)])
                        self.convert_process["score"].append(best_ans_score)
                        self.convert_process["global_sim"].append(global_sim)
                        self.convert_process["local_sim"].append(local_sim)
                        self.convert_process["local_diff"].append(local_diff)
                        self.convert_process["coverage"].append(coverage)

        assoc = np.zeros(len(d), dtype=int)
        assoc[rows_indexes] = 1

        # return the best so far
        return best_ans, scores, list(assoc)
