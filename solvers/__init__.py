from .bf_solver import BruteForceSolver
from .greedy_solver import GreedySolver
from .knn_solver import KnnSolver
from .mc_solver import MonteCarloSolver
from .teddy_knn_solver import TeddyKnnSolver

SOLVERS = {
    "BruteForce": BruteForceSolver,
    "Greedy": GreedySolver,
    "KNN": KnnSolver,
    "MonteCarlo": MonteCarloSolver,
    "TeddyKNN": TeddyKnnSolver
}
