# library imports
import numpy as np

# project imports
from experiments.experiment_properties.feature_distribution import FeatureDistribution


class FeatureDistributionNormal(FeatureDistribution):
    """
    A class responsible for a normally dist. feature
    """

    def __init__(self,
                 mean: float,
                 std: float):
        FeatureDistribution.__init__(self)
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean,
                                self.std,
                                1)[0]

    def sample_col(self,
                   count: int):
        return np.random.normal(self.mean,
                                self.std,
                                count)
