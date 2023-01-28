# library imports
import os

# project imports
from consts import *
from experiment import Experiment


class ExperimentSynthetic(Experiment):
    """
    A class for synthetic data experiments
    """

    def __init__(self):
        Experiment.__init__(self)

    def run(self):
        """
        This method runs an algorithm on the experiment's data and stores the results needed for this experiment
        """
        pass

    def test_results(self):
        """
        This method analyze the results of the algorithm, even more than one run and produces a result data structure
        """
        pass
