# library imports
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

# project imports
from solvers.solver import Solver


class TeddyKnnSolver(Solver):
    """
    A KNN approach for the rows (D') and a Top-k approach for F_{diff} such that k is searched using a grid search
    """

    def __init__(self, scorer, data, anomaly, k=3):
        super().__init__(scorer=scorer, data=data, anomaly=anomaly)
        self.k = k

    def solve(self) -> tuple:
        # run KNN on the D' for different samples of F_{diff} obtained from F_{diff}
        knn = KNeighborsRegressor(n_neighbors=self.k)
        knn.fit(X=self.data, y=list(range(self.dataset_size)))  # y is useless, can be any value
        rows_indexes = knn.kneighbors(X=[self.anomaly], n_neighbors=self.k, return_distance=False)[0]

        d_tag_full_f = self.data.iloc[rows_indexes]
        f_diff_dist_vector = np.abs(np.array(self.anomaly) - np.array(d_tag_full_f.mean(axis=0)))
        f_diff_candidates = [self.features[i] for i in (-f_diff_dist_vector).argsort()]

        for f_diff_size in tqdm(range(1, self.features_num + 1), position=0, desc=f"{'f_diff_size':<20}"):
            f_diff = f_diff_candidates[:f_diff_size]
            self.evaluate(rows_indexes=rows_indexes, f_diff=f_diff)

        self._add_informed_dataframe()
        return self.solution
