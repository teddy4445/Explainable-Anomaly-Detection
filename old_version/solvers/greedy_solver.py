# library imports
import numpy as np
import pandas as pd

# project imports
from tqdm import tqdm

from old_version.solvers.solver import Solver
from old_version.anomaly_detection_algos.anomaly_algo import AnomalyAlgo
from old_version.explanation_analysis.score_function.score_function import ScoreFunction


class GreedySolver(Solver):
    """
    A Brute Force approach for the rows (D') and columns (F_{diff})
    """

    def __init__(self,
                 param: dict = None):
        Solver.__init__(self,
                        param=param)
        self._depth = param.get('depth', -1)

    def solve(self,
              d: pd.DataFrame,
              anomaly_algo: AnomalyAlgo,
              s: list,
              time_limit_seconds: int,
              scorer: ScoreFunction,
              save_conv=False) -> tuple:
        if self._depth == -1:
            self._depth = len(d)

        features = d.columns.values
        num_rows, num_features = d.shape
        best_score = float('-inf')
        solution = {}
        selected_rows = {}

        # Iterate through each feature as a starting point
        total_iterations = len(d) * len(features)
        pbar = tqdm(total=total_iterations, desc=f"Processing combinations")
        for start_feature in d.columns:
            for start_row in range(len(d)):
                selected_rows = {start_row}  # Set to store the selected row indexes
                selected_features = {start_feature}  # Set to store the selected feature names

                # Iterate until all rows or features are selected
                best_row_score = float('-inf')  # best_score, float('-inf')
                best_feature_score = float('-inf')  # best_score, float('-inf')
                while len(selected_rows) < self._depth and len(selected_features) < num_features:
                    best_row = None
                    best_feature = None

                    # Find the best row to add
                    for row_index, row in d.iterrows():
                        if row_index not in selected_rows:
                            current_rows = selected_rows.union({row_index})
                            current_score, _ = scorer.compute(d=d.iloc[list(current_rows)], s=s,
                                                              f_sim=[feature for feature in features if
                                                                     feature not in selected_features],
                                                              f_diff=list(selected_features),
                                                              overall_size=len(d))

                            if current_score > best_row_score:
                                best_row_score = current_score
                                best_row = row_index

                    # Find the best feature to add
                    for feature in d.columns:
                        if feature not in selected_features:
                            current_features = selected_features.union({feature})
                            current_score, _ = scorer.compute(d=d.iloc[list(selected_rows)], s=s,
                                                              f_sim=[feature for feature in features if
                                                                     feature not in current_features],
                                                              f_diff=list(current_features),
                                                              overall_size=len(d))

                            if current_score > best_feature_score:
                                best_feature_score = current_score
                                best_feature = feature

                    # Add the best row or feature to the selected subsets
                    if not best_row and not best_feature:
                        break
                    elif best_row_score > best_feature_score:
                        selected_rows.add(best_row)
                        best_feature_score = best_row_score
                    else:
                        selected_features.add(best_feature)
                        best_row_score = best_feature_score

                # Check if the current subset is better than the previous best subset
                current_score, scores = scorer.compute(d=d.iloc[list(selected_rows)], s=s,
                                                       f_sim=[feature for feature in features if
                                                              feature not in selected_features],
                                                       f_diff=list(selected_features),
                                                       overall_size=len(d))

                if current_score > best_score:
                    best_score = current_score
                    solution = {'d_tag': d.iloc[list(selected_rows)],
                                'shape': (len(selected_rows), len(selected_features)),
                                'f_diff': list(selected_features),
                                'f_sim': [feature for feature in features if feature not in selected_features],
                                'best_score': current_score,
                                'self_sim': scores['self_sim'],
                                'local_sim': scores['local_sim'],
                                'sim_cluster': scores['sim_cluster_score'],
                                'local_diff': scores['local_diff'],
                                'diff_cluster': scores['sim_cluster_score'],
                                'coverage': scores['coverage'],
                                'conciseness': scores['conciseness']
                                }
                    print()
                    print(solution['shape'])

                # Update the progress bar
                pbar.update()

        # Close the progress bar
        pbar.close()

        if save_conv:
            self.close_convergence_process(time_limit_seconds=time_limit_seconds)
        assoc = np.zeros(len(d), dtype=int)
        assoc[list(selected_rows)] = 1

        # return the best so far
        return solution, list(assoc)
