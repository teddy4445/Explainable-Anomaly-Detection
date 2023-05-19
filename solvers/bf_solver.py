# library imports
import random
import itertools
import numpy as np
import pandas as pd
from time import time
from itertools import combinations, chain

# project imports
from tqdm import tqdm

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
        self.columns = param.get('columns', None)
        self.rows = param.get('rows', None)
        self.columns_num = param.get('columns_num', None)
        self.rows_num = param.get('rows_num', None)

    def solve(self,
              d: pd.DataFrame,
              anomaly_algo: AnomalyAlgo,
              s: list,
              time_limit_seconds: int,
              scorer: AfesMetric,
              save_conv=False) -> tuple:
        features = d.columns.values

        # check what is the best solution
        best_ans = None
        best_ans_score = -99999999

        # run until the time is over
        if self.columns and self.rows_num:
            f_sim = [feature for feature in features if feature not in self.columns]
            subsets_rows = list(map(set, itertools.combinations(list(range(d.shape[0])), self.rows_num)))
            # print('tqdm over subsets_rows')
            # start_time = time()
            for rows_indexes in subsets_rows:
                rows_indexes = list(rows_indexes)
                ans = d.loc[rows_indexes]
                # score it
                score = scorer.compute(d=ans, s=s, f_sim=f_sim, f_diff=self.columns, overall_size=len(d))
                self_sim, local_sim, local_diff, coverage = scorer.compute_parts(d=ans, s=s, f_sim=f_sim,
                                                                                 f_diff=self.columns,
                                                                                 overall_size=len(d))

                # if best so far, replace and record
                if score > best_ans_score:
                    best_ans_score = score
                    solution = {'d_tag': ans,
                                'shape': (len(ans), len(self.columns)),
                                'f_diff': self.columns,
                                'f_sim': f_sim,
                                'best_score': score,
                                'best_ss': self_sim,
                                'best_ls': local_sim,
                                'best_ld': local_diff,
                                'best_cov': coverage}

                # print(time() - start_time)
                # print()

        # elif self.columns_num and self.rows:
        #     subsets_cols = list(map(set, itertools.combinations(list(d.columns.values), self.columns_num)))
        #     ans = d.loc[self.rows]
        #     for cols_indexes in subsets_cols:
        #         # rows_indexes = self.rows
        #         cols_indexes = list(cols_indexes)
        #         # score it
        #         score = scorer.compute(d=ans, s=s, f_sim=[feature for feature in features if
        #                                                   feature not in cols_indexes],
        #                                f_diff=cols_indexes,
        #                                overall_size=len(d))
        #         self_sim, local_sim, local_diff, coverage = scorer.compute_parts(d=ans, s=s,
        #                                                                            f_sim=[feature for
        #                                                                                   feature in
        #                                                                                   features
        #                                                                                   if
        #                                                                                   feature not in cols_indexes],
        #                                                                            f_diff=cols_indexes,
        #                                                                            overall_size=len(d))
        #
        #         # if best so far, replace and record
        #         if score > best_ans_score:
        #             best_ans_score = score
        #             best_ans = ans[cols_indexes]
        #             scores = {'best_score': score,
        #                       'best_ss': self_sim,
        #                       'best_ls': local_sim,
        #                       'best_ld': local_diff,
        #                       'best_cov': coverage}
        #             rows_indexes = self.rows
        #
        # elif self.columns:
        #     print('tqdm over d_tag_size')
        #     for d_tag_size in tqdm(range(1, d.shape[0] + 1)):
        #         subsets_rows = list(map(set, itertools.combinations(list(range(d.shape[0])), d_tag_size)))
        #         for rows_indexes in subsets_rows:
        #             rows_indexes = list(rows_indexes)
        #             ans = d.loc[rows_indexes]
        #             # score it
        #             score = scorer.compute(d=ans, s=s, f_sim=self.columns,
        #                                    f_diff=[feature for feature in features if
        #                                            feature not in self.columns],
        #                                    overall_size=len(d))
        #             self_sim, local_sim, local_diff, coverage = scorer.compute_parts(d=ans, s=s,
        #                                                                                f_sim=self.columns,
        #                                                                                f_diff=[feature for feature in
        #                                                                                        features
        #                                                                                        if
        #                                                                                        feature not in self.columns],
        #                                                                                overall_size=len(d))
        #
        #             # if best so far, replace and record
        #             if score > best_ans_score:
        #                 # best_score = score
        #                 best_ans = ans
        #                 scores = {'best_score': score,
        #                           'best_ss': self_sim,
        #                           'best_ls': local_sim,
        #                           'best_ld': local_diff,
        #                           'best_cov': coverage}
        #
        # else:
        #     for d_tag_size in tqdm(range(1, d.shape[0] + 1)):
        #         for f_diff_size in range(1, d.shape[1] + 1):
        #             subsets_rows = list(map(set, itertools.combinations(list(range(d.shape[0])), d_tag_size)))
        #             subsets_cols = list(map(set, itertools.combinations(list(d.columns.values), f_diff_size)))
        #             for rows_indexes in subsets_rows:
        #                 for cols_indexes in subsets_cols:
        #                     rows_indexes = list(rows_indexes)
        #                     cols_indexes = list(cols_indexes)
        #                     ans = d.loc[rows_indexes]
        #                     # score it
        #                     score = scorer.compute(d=ans, s=s, f_sim=cols_indexes,
        #                                            f_diff=[feature for feature in features if
        #                                                    feature not in cols_indexes],
        #                                            overall_size=len(d))
        #                     self_sim, local_sim, local_diff, coverage = scorer.compute_parts(d=ans, s=s,
        #                                                                                        f_sim=cols_indexes,
        #                                                                                        f_diff=[feature for
        #                                                                                                feature in
        #                                                                                                features
        #                                                                                                if
        #                                                                                                feature not in cols_indexes],
        #                                                                                        overall_size=len(d))
        #
        #                     # if best so far, replace and record
        #                     if score > best_ans_score:
        #                         # best_score = score
        #                         best_ans = ans
        #                         scores = {'best_score': score,
        #                                   'best_ss': self_sim,
        #                                   'best_ls': local_sim,
        #                                   'best_ld': local_diff,
        #                                   'best_cov': coverage}
        #
        #                     # self.convert_process["time"].append(time() - start_time)
        #                     # self.convert_process["rows_indexes"].append(rows_indexes)
        #                     # self.convert_process["cols_indexes"].append(cols_indexes)
        #                     # self.convert_process["shape"].append([len(rows_indexes), len(cols_indexes)])
        #                     # self.convert_process["score"].append(best_ans_score)
        #                     # self.convert_process["self_sim"].append(self_sim)
        #                     # self.convert_process["local_sim"].append(local_sim)
        #                     # self.convert_process["local_diff"].append(local_diff)
        #                     # self.convert_process["coverage"].append(coverage)

        if save_conv:
            self.close_convergence_process(time_limit_seconds=time_limit_seconds)
        assoc = np.zeros(len(d), dtype=int)
        assoc[rows_indexes] = 1

        # return the best so far
        return solution, list(assoc)
