# library imports
import os

# project imports
from consts import *


class Experiment:
    """
    A virtual class of an experiment
    """

    def __init__(self):
        self.results = {}

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
