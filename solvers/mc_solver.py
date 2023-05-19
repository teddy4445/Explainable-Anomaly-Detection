# library imports
import random
import numpy as np
import pandas as pd
from time import time

# project imports
from solvers.solver import Solver
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.afes.afes_metric import AfesMetric


class MonteCarloSolver(Solver):
    """
    A Monte Carlo approach for the rows (D') and columns (F_{diff})
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
        while (time() - start_time) < time_limit_seconds or best_ans is None:
            # pick number of rows
            row_count = random.randint(2, d.shape[0])
            # pick number of cols
            col_count = random.randint(1, d.shape[1])
            # pick rows for D' at random
            rows_indexes = list(d.index)
            random.shuffle(rows_indexes)
            rows_indexes = rows_indexes[:row_count]
            # pick cols for F_{diff} at random
            cols_indexes = d.columns.values  # list(range(d.shape[1]))
            random.shuffle(cols_indexes)
            cols_indexes = cols_indexes[:col_count]
            # obtain the D' with F_{diff}
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
                best_ans_score = score
                best_ans = ans[cols_indexes]
                scores = {'best_score': score,
                          'best_ss': global_sim,
                          'best_ls': local_sim,
                          'best_ld': local_diff,
                          'best_cov': coverage}

            # self.convert_process["time"].append(time() - start_time)
            # self.convert_process["rows_indexes"].append(rows_indexes)
            # self.convert_process["cols_indexes"].append(cols_indexes)
            # self.convert_process["shape"].append([len(rows_indexes), len(cols_indexes)])
            # self.convert_process["score"].append(best_ans_score)
            # self.convert_process["global_sim"].append(global_sim)
            # self.convert_process["local_sim"].append(local_sim)
            # self.convert_process["local_diff"].append(local_diff)
            # self.convert_process["coverage"].append(coverage)

        # if self.convert_process["time"][-1] < time_limit_seconds:
        #     self.convert_process["time"].append(time_limit_seconds)
        #     self.convert_process["rows_indexes"].append(self.convert_process["rows_indexes"][-1])
        #     self.convert_process["cols_indexes"].append(self.convert_process["cols_indexes"][-1])
        #     self.convert_process["shape"].append(self.convert_process["shape"][-1])
        #     self.convert_process["score"].append(best_ans_score)
        #     self.convert_process["global_sim"].append(self.convert_process["global_sim"][-1])
        #     self.convert_process["local_sim"].append(self.convert_process["local_sim"][-1])
        #     self.convert_process["local_diff"].append(self.convert_process["local_diff"][-1])
        #     self.convert_process["coverage"].append(self.convert_process["coverage"][-1])
        #     print()

        assoc = np.zeros(len(d), dtype=int)
        assoc[rows_indexes] = 1

        # return the best so far
        return best_ans, scores, list(assoc)
