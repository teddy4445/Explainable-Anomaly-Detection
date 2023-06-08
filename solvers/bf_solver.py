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
from explanation_analysis.score_function.score_function import ScoreFunction


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
              scorer: ScoreFunction,
              save_conv=False) -> tuple:
        # check what is the best solution
        features = d.columns.values
        assoc = np.zeros(len(d), dtype=int)
        best_ans_score = -99999999
        best_row_indexes = []
        solution = {}

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
                current_score, scores = scorer.compute(d=ans, s=s, f_sim=f_sim, f_diff=self.columns,
                                                       overall_size=len(d))

                # if best so far, replace and record
                if current_score > best_ans_score:
                    best_ans_score = current_score
                    best_row_indexes = rows_indexes
                    solution = {'d_tag': ans,
                                'shape': (self.rows_num, len(self.columns)),
                                'f_diff': self.columns,
                                'f_sim': f_sim,
                                'best_score': best_ans_score,
                                'self_sim': scores['self_sim'],
                                'local_sim': scores['local_sim'],
                                'sim_cluster': scores['sim_cluster_score'],
                                'local_diff': scores['local_diff'],
                                'diff_cluster': scores['sim_cluster_score'],
                                'coverage': scores['coverage']}

                # print(time() - start_time)
                # print()
            assoc[best_row_indexes] = 1

        elif self.columns_num and self.rows:
            subsets_cols = list(map(set, itertools.combinations(list(d.columns.values), self.columns_num)))
            ans = d.loc[self.rows]
            for cols_indexes in subsets_cols:
                # rows_indexes = self.rows
                cols_indexes = list(cols_indexes)
                # score it
                current_score, scores = scorer.compute(d=ans, s=s, f_diff=cols_indexes, overall_size=len(d),
                                                       f_sim=[feature for feature in features if
                                                              feature not in cols_indexes])

                # if best so far, replace and record
                if current_score > best_ans_score:
                    best_ans_score = current_score
                    solution = {'d_tag': ans,
                                'shape': (len(self.rows), len(self.columns_num)),
                                'f_diff': cols_indexes,
                                'f_sim': [feature for feature in features if feature not in cols_indexes],
                                'best_score': best_ans_score,
                                'self_sim': scores['self_sim'],
                                'local_sim': scores['local_sim'],
                                'sim_cluster': scores['sim_cluster_score'],
                                'local_diff': scores['local_diff'],
                                'diff_cluster': scores['sim_cluster_score'],
                                'coverage': scores['coverage']}
            assoc[self.rows] = 1

        elif self.columns:
            print('tqdm over d_tag_size')
            for d_tag_size in tqdm(range(1, d.shape[0] + 1)):
                subsets_rows = list(map(set, itertools.combinations(list(range(d.shape[0])), d_tag_size)))
                for rows_indexes in subsets_rows:
                    rows_indexes = list(rows_indexes)
                    ans = d.loc[rows_indexes]
                    # score it
                    current_score, scores = scorer.compute(d=ans, s=s, f_diff=self.columns, overall_size=len(d),
                                                           f_sim=[feature for feature in features if
                                                                  feature not in self.columns])

                    # if best so far, replace and record
                    if current_score > best_ans_score:
                        best_ans_score = current_score
                        best_row_indexes = rows_indexes
                        solution = {'d_tag': ans,
                                    'shape': (len(rows_indexes), len(self.columns)),
                                    'f_diff': self.columns,
                                    'f_sim': [feature for feature in features if feature not in self.columns],
                                    'best_score': best_ans_score,
                                    'self_sim': scores['self_sim'],
                                    'local_sim': scores['local_sim'],
                                    'sim_cluster': scores['sim_cluster_score'],
                                    'local_diff': scores['local_diff'],
                                    'diff_cluster': scores['sim_cluster_score'],
                                    'coverage': scores['coverage']}
            assoc[best_row_indexes] = 1

        if self.rows_num:
            subsets_rows = list(itertools.combinations(list(range(d.shape[0])), self.rows_num))
            for rows_indexes in subsets_rows:
                rows_indexes = list(rows_indexes)
                ans = d.loc[rows_indexes]
                subsets_cols = list(chain.from_iterable(combinations(features, r) for r in range(1, d.shape[1] + 1)))
                for cols_indexes in subsets_cols:
                    cols_indexes = list(cols_indexes)
                    # score it
                    current_score, scores = scorer.compute(d=ans, s=s, f_diff=cols_indexes, overall_size=len(d),
                                                           f_sim=[feature for feature in features if
                                                                  feature not in cols_indexes])

                    # if best so far, replace and record
                    if current_score > best_ans_score:
                        best_ans_score = current_score
                        best_row_indexes = rows_indexes
                        solution = {'d_tag': ans,
                                    'shape': (self.rows_num, len(cols_indexes)),
                                    'f_diff': cols_indexes,
                                    'f_sim': [feature for feature in features if feature not in cols_indexes],
                                    'best_score': best_ans_score,
                                    'self_sim': scores['self_sim'],
                                    'local_sim': scores['local_sim'],
                                    'sim_cluster': scores['sim_cluster_score'],
                                    'local_diff': scores['local_diff'],
                                    'diff_cluster': scores['sim_cluster_score'],
                                    'coverage': scores['coverage']}
            assoc[best_row_indexes] = 1

        else:
            for d_tag_size in tqdm(range(1, d.shape[0] + 1)):
                for f_diff_size in range(1, d.shape[1] + 1):
                    subsets_rows = list(map(set, itertools.combinations(list(range(d.shape[0])), d_tag_size)))
                    subsets_cols = list(map(set, itertools.combinations(list(d.columns.values), f_diff_size)))
                    for rows_indexes in subsets_rows:
                        for cols_indexes in subsets_cols:
                            rows_indexes = list(rows_indexes)
                            cols_indexes = list(cols_indexes)
                            ans = d.loc[rows_indexes]
                            # score it
                            current_score, scores = scorer.compute(d=ans, s=s, f_diff=cols_indexes, overall_size=len(d),
                                                                   f_sim=[feature for feature in features if
                                                                          feature not in cols_indexes])

                            # if best so far, replace and record
                            if current_score > best_ans_score:
                                best_ans_score = current_score
                                best_row_indexes = rows_indexes
                                solution = {'d_tag': ans,
                                            'shape': (len(rows_indexes), len(cols_indexes)),
                                            'f_diff': cols_indexes,
                                            'f_sim': [feature for feature in features if feature not in cols_indexes],
                                            'best_score': best_ans_score,
                                            'self_sim': scores['self_sim'],
                                            'local_sim': scores['local_sim'],
                                            'sim_cluster': scores['sim_cluster_score'],
                                            'local_diff': scores['local_diff'],
                                            'diff_cluster': scores['sim_cluster_score'],
                                            'coverage': scores['coverage']}

                            # self.convert_process["time"].append(time() - start_time)
                            # self.convert_process["rows_indexes"].append(rows_indexes)
                            # self.convert_process["cols_indexes"].append(cols_indexes)
                            # self.convert_process["shape"].append([len(rows_indexes), len(cols_indexes)])
                            # self.convert_process["score"].append(best_ans_score)
                            # self.convert_process["self_sim"].append(self_sim)
                            # self.convert_process["local_sim"].append(local_sim)
                            # self.convert_process["local_diff"].append(local_diff)
                            # self.convert_process["coverage"].append(coverage)
            assoc[best_row_indexes] = 1

        if save_conv:
            self.close_convergence_process(time_limit_seconds=time_limit_seconds)

        # return the best so far
        return solution, list(assoc)
