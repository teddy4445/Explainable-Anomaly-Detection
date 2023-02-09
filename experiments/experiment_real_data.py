# library imports
import pandas as pd

# project imports
from experiments.experiment_properties import Experiment


class ExperimentRealData(Experiment):
    """
    A class for real-world data experiments
    """

    def __init__(self):
        Experiment.__init__(self)

    def run(self,
            algo,
            dataset: pd.DataFrame,
            anomaly_sample: list,
            f_diff: list):
        """
        This method runs an algorithm on the experiment_properties's data and stores the results needed for this experiment_properties
        """
        pass

    def test_results(self):
        """
        This method analyze the results of the algorithm, even more than one run and produces a result data structure
        """
        pass
