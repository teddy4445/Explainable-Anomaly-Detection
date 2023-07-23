# library imports
from tqdm import tqdm

# project imports
from solvers.solver import Solver


class GreedySolver(Solver):
    """
    A Brute Force approach for the rows (D') and columns (F_{diff})
    """

    def __init__(self, scorer, data, anomaly, depth=-1):
        super().__init__(scorer=scorer, data=data, anomaly=anomaly)
        self._depth = depth

    def _find_better_next_step(self, selected_rows, selected_features, best_path_score):
        best_solution = None
        for row_index, row in self.data.iterrows():
            if row_index not in selected_rows:
                better_solution = self.evaluate(rows_indexes=selected_rows + [row_index], f_diff=selected_features,
                                                update_globals=False, current_best_score=best_path_score)
                if better_solution:
                    best_path_score = better_solution["best_score"]
                    best_solution = better_solution

        for feature in self.features:
            if feature not in selected_features:
                better_solution = self.evaluate(rows_indexes=selected_rows, f_diff=selected_features + [feature],
                                                update_globals=False, current_best_score=best_path_score)
                if better_solution:
                    best_path_score = better_solution["best_score"]
                    best_solution = better_solution

        return best_solution

    def _build_path(self, start_row, start_feature):
        selected_rows = [start_row]  # Set to store the selected row indexes
        selected_features = [start_feature]  # Set to store the selected feature names

        # Iterate until all rows or features are selected
        best_path_score = float('-inf')  # best_score, float('-inf')
        best_path_solution = None
        while len(selected_rows) < self._depth:
            better_path_solution = self._find_better_next_step(selected_rows, selected_features, best_path_score)
            if better_path_solution:
                best_path_solution = better_path_solution
                best_path_score = better_path_solution["best_score"]
                selected_rows = better_path_solution["rows_indexes"]
                selected_features = better_path_solution["f_diff"]
            else:
                break

        return best_path_solution

    def solve(self) -> tuple:
        if self._depth == -1:
            self._depth = len(self.data)

        # Iterate through each feature as a starting point
        # total_iterations = self.dataset_size * len(self.features)
        # pbar = tqdm(total=total_iterations, desc=f"Processing combinations")
        for start_feature in tqdm(self.features, position=0, desc=f"{'start_feature':<20}"):
            for start_row in tqdm(range(self.dataset_size), position=1, leave=False, desc=f"{'start_row':<20}"):
                solution = self._build_path(start_row, start_feature)
                self.evaluate(rows_indexes=solution["rows_indexes"], f_diff=solution["f_diff"])

        self._add_informed_dataframe()
        return self.solution
