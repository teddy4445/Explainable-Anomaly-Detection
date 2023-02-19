# library imports
import os
import numpy as np

# project imports
from consts import *
from experiments.experiment import Experiment
from solvers.mc_solver import MonteCarloSolver
from anomaly_detection_algos.z_score import Zscore
from explanation_analysis.afes.afes_sum import AfesSum
from anomaly_detection_algos.DBSCAN import DBSCANwrapper
from explanation_analysis.similarity_metrices.sim_prob import ProbSim
from experiments.experiment_properties.feature_distribution_normal import FeatureDistributionNormal
from experiments.experiment_properties.synthetic_dataset_generation import SyntheticDatasetGeneration


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
        # generate datasets
        synthetic_data_exp = Experiment(time_limit_seconds=30)
        # generate dataset
        f_diff = [0, 1]
        d_tag_size = 10
        dataset, d_tag = SyntheticDatasetGeneration.generate_one(anomaly_detection_algorithm=DBSCANwrapper(),
                                                                 row_count=50,
                                                                 cols_dist_functions={
                                                                     "a": FeatureDistributionNormal(mean=1, std=0.5),
                                                                     "b": FeatureDistributionNormal(mean=0.8, std=0.7),
                                                                     "c": FeatureDistributionNormal(mean=5, std=1)
                                                                 },
                                                                 f_diff=f_diff,
                                                                 d_tag_size=d_tag_size,
                                                                 save_csv=os.path.join(RESULTS_FOLDER_PATH,
                                                                                       "main_synt_example.csv")
                                                                 )

        # run experiment
        synthetic_data_exp.run(anomaly_algo=DBSCANwrapper(),
                               solver=MonteCarloSolver(),
                               scorer=AfesSum(sim=ProbSim,
                                              w_gsim=1,
                                              w_ldiff=1,
                                              w_lsim=1),
                               d_tags=[d_tag],
                               f_diff_list=[f_diff],
                               anomaly_sample=[-1],
                               dataset=dataset
                               )
        # check results
        print(synthetic_data_exp.test_results_report())


if __name__ == '__main__':
    Main.run()
