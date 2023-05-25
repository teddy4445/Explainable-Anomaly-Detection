# library imports
import random
import numpy as np
import pandas as pd
from time import time

# project imports
from solvers.solver import Solver
from anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from explanation_analysis.score_function.score_function import ScoreFunction


class OneOneSolver(Solver):
    """
    A solver that tries to get the best row and column each time
    """

    def __init__(self,
                 param: dict = None):
        Solver.__init__(self,
                        param=param)
        self.d_tag_size = param["d_tag_size"]
        self.f_diff_size = param["f_diff_size"]

    def solve(self,
              d: pd.DataFrame,
              anomaly_algo: AnomalyAlgo,
              s: list,
              time_limit_seconds: int,
              scorer: ScoreFunction) -> tuple:
        # check what is the best solution
        features = list(d.columns.values)
        start_feature = features[random.randrange(len(features))]
        iter_features = list(set(features) - set([start_feature]))
        rows_num = len(d)
        iter_rows = list(range(rows_num))

        cols = [start_feature]
        rows = []
        is_row = True
        stop_switch = False

        # run until the time is over
        best_score = 0
        start_time = time()
        for index in range(self.d_tag_size + self.f_diff_size - 1):  # self.d_tag_size + self.f_diff_size - 1
            # just to break over time
            if (time() - start_time) > time_limit_seconds:
                break
            if is_row:
                best_row_score = -99999999
                best_row_index = [iter_rows[random.randrange(len(iter_rows))]]  # iter_rows[0]
                for row_index in iter_rows:
                    if row_index in rows:
                        pass
                    this_rows = rows.copy()  # why?
                    this_rows.append(row_index)
                    score = scorer.compute(d=d.iloc[this_rows], s=s, f_sim=cols,
                                           f_diff=[feature for feature in features if feature not in cols])
                    global_sim, local_sim, local_diff = scorer.compute_parts(d=d.iloc[this_rows], s=s, f_sim=cols,
                                                                             f_diff=[feature for feature in features if
                                                                                     feature not in cols])
                    if best_row_score < score:
                        best_row_score = score
                        best_row_index = row_index

                if best_row_index not in rows:
                    rows.append(best_row_index)
                    best_score = best_row_score
                iter_rows = list(set(iter_rows) - set(rows))

                if len(rows) == self.d_tag_size:  # rows_num
                    stop_switch = True
                    is_row = False
                elif stop_switch:
                    is_row = True
                else:
                    is_row = bool(random.randint(0, 1))

            else:
                best_col_score = -99999999
                best_col_index = [iter_features[random.randrange(len(iter_features))]]  # iter_features[0]
                for col_index in iter_features:
                    if col_index in cols:
                        pass
                    this_cols = cols.copy()
                    this_cols.append(col_index)
                    score = scorer.compute(d=d.iloc[rows], s=s, f_sim=this_cols,
                                           f_diff=[feature for feature in features if feature not in this_cols])
                    global_sim, local_sim, local_diff = scorer.compute_parts(d=d.iloc[rows], s=s, f_sim=this_cols,
                                                                             f_diff=[feature for feature in features if
                                                                                     feature not in this_cols])
                    if best_col_score < score:
                        best_col_score = score
                        best_col_index = col_index

                if best_col_index not in cols:
                    cols.append(best_col_index)
                    best_score = best_col_score
                iter_features = list(set(iter_features) - set(cols))

                if len(cols) == self.f_diff_size:  # len(features)
                    stop_switch = True
                    is_row = True
                elif stop_switch:
                    is_row = False
                else:
                    is_row = bool(random.randint(0, 1))

            self.convert_process["time"].append(time() - start_time)
            self.convert_process["rows_indexes"].append(rows)
            self.convert_process["cols_indexes"].append(cols)
            self.convert_process["shape"].append([len(rows), len(cols)])
            self.convert_process["score"].append(best_score)
            self.convert_process["global_sim"].append(global_sim)
            self.convert_process["local_sim"].append(local_sim)
            self.convert_process["local_diff"].append(local_diff)

        if self.convert_process["time"][-1] < 60:
            self.convert_process["time"].append(60.0)
            self.convert_process["rows_indexes"].append(self.convert_process["rows_indexes"][-1])
            self.convert_process["cols_indexes"].append(self.convert_process["cols_indexes"][-1])
            self.convert_process["shape"].append(self.convert_process["shape"][-1])
            self.convert_process["score"].append(self.convert_process["score"][-1])
            self.convert_process["global_sim"].append(self.convert_process["global_sim"][-1])
            self.convert_process["local_sim"].append(self.convert_process["local_sim"][-1])
            self.convert_process["local_diff"].append(self.convert_process["local_diff"][-1])

        # return the best so far
        ans = d.loc[rows, cols]

        assoc = np.zeros(len(d), dtype=int)
        assoc[rows] = 1

        return ans, self.convert_process["score"][-1], list(assoc)
