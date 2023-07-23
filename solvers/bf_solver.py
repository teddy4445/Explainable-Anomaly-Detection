# library imports
from time import time
from tqdm import tqdm

# project imports
from solvers.solver import Solver


class BruteForceSolver(Solver):
    """
    A Brute Force approach for the rows (D') and columns (F_{diff})
    """

    def __init__(self, scorer, data, anomaly,
                 columns=None, columns_num=None, rows=None, rows_num=None):
        super().__init__(scorer=scorer, data=data, anomaly=anomaly)
        self.columns = columns
        self.rows = rows
        self.columns_num = columns_num
        self.rows_num = rows_num

    def solve(self) -> tuple:
        # TODO: add tqdm progress bar
        # TODO: Check on all cases
        d_tag_sizes = [self.rows_num] if (self.rows_num or self.rows) else range(1, self.dataset_size + 1)
        f_diff_sizes = [self.columns_num] if (self.columns_num or self.columns) else range(1, self.data.shape[1] + 1)
        for d_tag_size in tqdm(d_tag_sizes, position=0, desc=f"{'d_tag_size':<20}"):
            for f_diff_size in tqdm(f_diff_sizes, position=1, leave=False, desc=f"{'f_diff_size':<20}"):
                subsets_rows = [self.rows] if self.rows else self._get_subsets_of_size(subset_size=d_tag_size, father_set=range(self.dataset_size))
                subsets_cols = [self.columns] if self.columns else self._get_subsets_of_size(subset_size=f_diff_size, father_set=self.features)
                for selected_features in tqdm(subsets_cols, position=2, leave=False, desc=f"{'selected_features':<20}"):
                    for selected_rows in tqdm(subsets_rows, position=3, leave=False, desc=f"{'selected_rows':<20}"):
                        self.evaluate(rows_indexes=selected_rows, f_diff=selected_features)

        self._add_informed_dataframe()
        return self.solution
