# library imports
from time import time
from sklearn.neighbors import KNeighborsRegressor

# project imports
from tqdm import tqdm

from solvers.solver import Solver
from scorers.score_function.score_function import ScoreFunction


class KnnSolver(Solver):
    """
    A KNN approach
    """

    def __init__(self, scorer, data, anomaly, k=3, f_diff=None):
        super().__init__(scorer=scorer, data=data, anomaly=anomaly)
        self.k = k
        self.f_diff = f_diff
        # self.f_diff_size = self.params.get('f_diff_size', None)

    def solve(self) -> tuple:
        # Run KNN and find top-k
        knn = KNeighborsRegressor(n_neighbors=self.k)
        knn.fit(X=self.data, y=list(range(self.dataset_size)))  # y is useless so we just put indexes, can be any value
        rows_indexes = knn.kneighbors(X=[self.anomaly], n_neighbors=self.k, return_distance=False)[0]

        if self.f_diff:
            self.evaluate(rows_indexes=rows_indexes, f_diff=self.f_diff)

        else:
            # run until the time is over
            # while (time() - start_time) < time_limit_seconds or best_ans is None:
            for f_diff_size in tqdm(range(1, self.features_num + 1), position=0, desc=f"{'f_diff_size':<20}"):
                subsets_cols = self._get_subsets_of_size(subset_size=f_diff_size, father_set=self.features)
                for cols_indexes in tqdm(subsets_cols, position=1, leave=False, desc=f"{'cols_indexes':<20}"):
                    self.evaluate(rows_indexes=rows_indexes, f_diff=cols_indexes)

        self._add_informed_dataframe()
        return self.solution
