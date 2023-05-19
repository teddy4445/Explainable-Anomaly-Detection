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


class GreedySolver(Solver):
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
              scorer: AfesMetric) -> tuple:
        features = d.columns.values
        best_ans = None
        selected_features = {features[random.randrange(len(features))]}  # Set to store the selected feature names

        num_rows, num_features = d.shape
        best_score = float('-inf')
        best_subset = None

        # Iterate through each feature as a starting point
        for start_feature in d.columns:
            selected_rows = set()  # Set to store the selected row indexes
            selected_features = {start_feature}  # Set to store the selected feature names

            # Iterate until all rows or features are selected
            while len(selected_rows) < num_rows and len(selected_features) < num_features:
                best_row_score = float('-inf')
                best_row = None
                best_feature_score = float('-inf')
                best_feature = None

                # Find the best row to add
                for row_index, row in d.iterrows():
                    if row_index not in selected_rows:
                        current_rows = selected_rows.union({row_index})
                        current_score = scorer.compute(d=d.iloc[current_rows], s=s,
                                                       f_sim=[feature for feature in features if
                                                              feature not in selected_features],
                                                       f_diff=selected_features)

                        if current_score > best_row_score:
                            best_row_score = current_score
                            best_row = row_index

                # Find the best feature to add
                for feature in d.columns:
                    if feature not in selected_features:
                        current_features = selected_features.union({feature})
                        current_score = scorer.compute(d=d.iloc[selected_rows], s=s,
                                                       f_sim=[feature for feature in features if
                                                              feature not in current_features],
                                                       f_diff=current_features)

                        if current_score > best_feature_score:
                            best_feature_score = current_score
                            best_feature = feature

                # Add the best row or feature to the selected subsets
                if best_row_score > best_feature_score:
                    selected_rows.add(best_row)
                else:
                    selected_features.add(best_feature)

            # Check if the current subset is better than the previous best subset
            current_score = scorer.compute(d=d.iloc[selected_rows], s=s,
                                           f_sim=[feature for feature in features if
                                                  feature not in selected_features],
                                           f_diff=selected_features)
            if current_score > best_score:
                best_score = current_score
                best_subset = (selected_rows, selected_features)

        return best_subset

        # return selected_rows, selected_features
        #
        # assoc = np.zeros(len(d), dtype=int)
        # assoc[rows_indexes] = 1
        #
        # return the best so far
        # return best_ans, scores, list(assoc)
