from .sim_cosine import CosineSim
from .sim_euclidean_inverse import InverseEuclideanSim
from .sim_mean_entropy import MeanEntropySim

SIMILARITY_FUNCTIONS = {
    "cosine": CosineSim,
    "inverse_euclidean": InverseEuclideanSim,
    "mean_entropy": MeanEntropySim
}
