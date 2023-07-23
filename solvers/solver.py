# library imports
import itertools
import pandas as pd

# project imports
from scorers.score_function.score_function import ScoreFunction


class Solver:
    """
    An abstract class for the project's solver classes
    """

    def __init__(self, scorer: ScoreFunction, data, anomaly, params: dict = None):
        self.params = params
        self.scorer = scorer
        self.data = data
        self.anomaly = anomaly
        self.dataset_size = len(self.data)
        self.features = list(self.data.columns.values)
        self.features_num = len(self.features)
        self.best_score = float('-inf')
        self.best_row_indexes = None
        self.solution = None

    @staticmethod
    def _get_subsets_of_size(subset_size, father_set):
        return [list(combination) for combination in itertools.combinations(father_set, subset_size)]

    @staticmethod
    def _get_solution(best_score, scores, rows_indexes, f_diff, f_sim):
        solution = {
            'shape': (len(rows_indexes), len(f_diff)),
            'f_diff': f_diff,
            'f_sim': f_sim,
            'rows_indexes': rows_indexes,
            'best_score': best_score,
            'sub_scores': scores,
            'd_inf': None
        }
        return solution

    def evaluate(self, rows_indexes, f_diff, update_globals=True, current_best_score=None):  # , f_sim
        f_sim = [feature for feature in self.features if feature not in f_diff]

        # score it
        current_score, scores = self.scorer.compute(d=self.data.loc[rows_indexes],
                                                    s=self.anomaly,
                                                    f_diff=f_diff, f_sim=f_sim,
                                                    overall_size=self.dataset_size)

        current_best_score = current_best_score or self.best_score
        if current_score > current_best_score:
            solution = self._get_solution(best_score=current_score, scores=scores,
                                          rows_indexes=rows_indexes, f_diff=f_diff, f_sim=f_sim)
            if update_globals:
                self.best_score = current_score
                self.best_row_indexes = rows_indexes
                self.solution = solution
            return solution

    def _add_informed_dataframe(self):
        d_inf = pd.concat([self.data, pd.DataFrame(self.anomaly).T], ignore_index=True)
        d_inf['assoc'] = [1 if (i in self.best_row_indexes) else 0 for i in range(self.dataset_size)] + [2]
        self.solution['d_inf'] = d_inf

    def solve(self) -> tuple:
        pass
