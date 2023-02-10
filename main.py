# library imports

# project imports
from consts import *
from anomaly_detection_algos.z_score import Zscore
from experiments.experiment_real_data import ExperimentRealData
from experiments.experiment_synthetic import ExperimentSynthetic
from solvers.mc_solver import MonteCarloSolver


class Main:
    """
    The main class of the project, allow other developers to run all the experiments in the paper
    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        """
        Single entry point
        """

        # 1) prepare IO
        for path in SETUP_FOLDERS:
            os.makedirs(path, exist_ok=True)

        # 2) experiments run #
        # load experiments
        synthetic_data_exp = ExperimentSynthetic()
        real_data_exp = ExperimentRealData()
        # load algorithms
        algos = {
            Zscore.NAME: Zscore
        }
        algo_properties = {
            Zscore.NAME: {Zscore.PROPERTY_NAME: 1.0}
        }
        sovler = MonteCarloSolver()
        # run them
        # check results
        # TODO: add later
        # save results
        # TODO: add later

        # 3) prepare plots for the paper
        # TODO: add later

        if __name__ == '__main__':
            Main.run()
