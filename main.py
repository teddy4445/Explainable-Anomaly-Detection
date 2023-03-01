# library imports
import os
import numpy as np
import pandas as pd

# project imports
from consts import *
from experiments.experiment import Experiment
from solvers.knn_solver import KnnSolver
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
        iterations = 10
        best_ans_knn, best_ans_mc = [], []
        ans_knn_shape, ans_mc_shape = [], []
        best_ans_score_knn, best_ans_score_mc = [], []
        knn_solving_time, mc_solving_time = [], []

        for iteration in range(iterations):
            # Set-up experiment
            print(f"Set-up experiment {iteration}")
            synthetic_data_knn_exp = Experiment(time_limit_seconds=60)
            synthetic_data_mc_exp = Experiment(time_limit_seconds=60)

            # generate dataset
            print(f"generate dataset {iteration}")
            f_diff = [0, 1]
            d_tag_size = 10
            dataset, d_tag, _ = SyntheticDatasetGeneration.generate_one(anomaly_detection_algorithm=DBSCANwrapper(),
                                                                        row_count=50,
                                                                        cols_dist_functions={
                                                                            "a": FeatureDistributionNormal(mean=1,
                                                                                                           std=0.5),
                                                                            "b": FeatureDistributionNormal(mean=0.8,
                                                                                                           std=0.7),
                                                                            "c": FeatureDistributionNormal(mean=5,
                                                                                                           std=1)
                                                                        },
                                                                        f_diff=f_diff,
                                                                        d_tag_size=d_tag_size,
                                                                        save_csv=os.path.join(RESULTS_FOLDER_PATH,
                                                                                              "main_synt_example")
                                                                        )

            # run experiments
            print(f"run experiment {iteration}")
            synthetic_data_knn_exp.run(anomaly_algo=DBSCANwrapper(),
                                       solver=KnnSolver(),
                                       scorer=AfesSum(sim_module=ProbSim(),
                                                      w_gsim=1,
                                                      w_ldiff=1,
                                                      w_lsim=1),
                                       d_tags=[d_tag],
                                       f_diff_list=[f_diff],
                                       anomaly_sample=dataset.iloc[-1].values,  # [-1], .reshape(1, -1)
                                       dataset=dataset
                                       )
            synthetic_data_mc_exp.run(anomaly_algo=DBSCANwrapper(),
                                      solver=MonteCarloSolver(),
                                      scorer=AfesSum(sim_module=ProbSim(),
                                                     w_gsim=1,
                                                     w_ldiff=1,
                                                     w_lsim=1),
                                      d_tags=[d_tag],
                                      f_diff_list=[f_diff],
                                      anomaly_sample=dataset.iloc[-1].values,  # [-1], .reshape(1, -1)
                                      dataset=dataset
                                      )

            # check results
            print("check results")
            best_ans_knn.append(synthetic_data_knn_exp.results['best_ans'])
            best_ans_mc.append(synthetic_data_mc_exp.results['best_ans'])
            ans_knn_shape.append(synthetic_data_knn_exp.results['best_ans'].shape)
            ans_mc_shape.append(synthetic_data_mc_exp.results['best_ans'].shape)
            best_ans_score_knn.append(synthetic_data_knn_exp.results['best_ans_score'])
            best_ans_score_mc.append(synthetic_data_mc_exp.results['best_ans_score'])
            knn_solving_time.append(synthetic_data_knn_exp.results['solving_time'])
            mc_solving_time.append(synthetic_data_mc_exp.results['solving_time'])

            # print(synthetic_data_exp.test_results_report())

        analysis_df = pd.DataFrame({'best_ans_score_knn': best_ans_score_knn, 'best_ans_score_mc': best_ans_score_mc,
                                    'ans_knn_shape': ans_knn_shape, 'ans_mc_shape': ans_mc_shape,
                                    'knn_solving_time': knn_solving_time, 'mc_solving_time': mc_solving_time})
        analysis_df.to_csv(os.path.join(RESULTS_FOLDER_PATH, "meta_analysis.csv"), index=False)


if __name__ == '__main__':
    Main.run()
