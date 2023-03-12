# library imports
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
from tqdm import tqdm

# project imports
from consts import *
from anomaly_detection_algos.z_score import Zscore
from anomaly_detection_algos.DBSCAN import DBSCANwrapper
from explanation_analysis.afes.afes_sum import AfesSum
from explanation_analysis.similarity_metrices.sim_cosine import CosineSim
from explanation_analysis.similarity_metrices.sim_euclidean import EuclideanSim
from explanation_analysis.similarity_metrices.sim_mean_entropy import MeanEntropySim
from explanation_analysis.similarity_metrices.sim_prob import ProbSim
from experiments.experiment import Experiment
from experiments.experiment_properties.feature_distribution_normal import FeatureDistributionNormal
from experiments.experiment_properties.synthetic_dataset_generation import SyntheticDatasetGeneration
from solvers.knn_solver import KnnSolver
from solvers.mc_solver import MonteCarloSolver
from solvers.one_ones_solver import OneOneSolver


class Main:
    """
    The main class of the project, allow other developers to run all the experiments in the paper
    """

    def __init__(self):
        pass

    @staticmethod
    def run(create_corpus=False, iterations=50, features_num=5, row_count=50, f_diff=[0, 1], d_tag_size=10,
            run_experiments=True, corpus_name="DBSCAN_rc50_pmNone"):
        """
        Single entry point
        """

        # 1) prepare IO
        for path in SETUP_FOLDERS:
            os.makedirs(path, exist_ok=True)

        # 2) generate dataset
        if create_corpus:
            anom_algo_name = DBSCANwrapper.NAME
            performance_metric = None  # EuclideanSim()
            performance_metric_name = None  # performance_metric.NAME
            performance_metric_attempts = 0  # 2

            meta_data = {"iterations": iterations,
                         "features_num": features_num,
                         "row_count": row_count,
                         "f_diff": f_diff,
                         "d_tag_size": d_tag_size,
                         "anom_algo_name": anom_algo_name,
                         "performance_metric_name": performance_metric_name,
                         "performance_metric_attempts": performance_metric_attempts,
                         "cols_dist": {}}
            folder_path = os.path.join(DATA_FOLDER_PATH,
                                       f"{DBSCANwrapper.NAME}_rc{row_count}_pm{performance_metric_name}")
            os.makedirs(folder_path, exist_ok=True)

            for iteration in tqdm(range(iterations)):
                # print(f"generate dataset {iteration}")

                cols_dist_functions = {}
                for feature in range(features_num):
                    mean = random.uniform(0, 1)
                    std = random.uniform(0, 1)
                    cols_dist_functions[feature] = FeatureDistributionNormal(mean=mean, std=std)
                meta_data["cols_dist"][f"iter{iteration}"] = cols_dist_functions

                dataset, d_tag, d_inf = SyntheticDatasetGeneration.generate_one(
                    anomaly_detection_algorithm=DBSCANwrapper(),
                    row_count=row_count,
                    cols_dist_functions=cols_dist_functions,
                    f_diff=f_diff,
                    d_tag_size=d_tag_size,
                    performance_metric=performance_metric,
                    performance_metric_attempts=performance_metric_attempts,
                    save_csv=os.path.join(folder_path, f"synt_iter{iteration}")
                )

            # save dictionary to person_data.pkl file
            with open(os.path.join(folder_path, "meta_data.pkl"), 'wb') as fp:
                pickle.dump(meta_data, fp)

        # 3) run experiments
        elif run_experiments:
            results_path = os.path.join(RESULTS_FOLDER_PATH, corpus_name)
            os.makedirs(results_path, exist_ok=True)

            d_tag_size = 10

            # Set-up experiment
            print(f"Set-up experiment")
            # best_ans_knn = []
            # ans_knn_shape = []
            # best_ans_score_knn = []
            # knn_solving_time = []
            print()

            # run experiments
            print(f"run experiments")
            for filename in tqdm(os.scandir(os.path.join(DATA_FOLDER_PATH, corpus_name))):
                if filename.is_file():
                    d_inf = pd.read_csv(filename)
                    dataset = d_inf[[feature for feature in d_inf.columns.values if feature != 'assoc']]
                    d_tag = dataset.loc[(d_inf['assoc'] == 1) | (d_inf['assoc'] == 2)]
                    anomaly_sample = dataset.loc[d_inf['assoc'] == 2].iloc[-1]

                    knn_exp = Experiment(time_limit_seconds=60)
                    knn_exp.run(anomaly_algo=DBSCANwrapper(),
                                solver=KnnSolver(),
                                scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                                d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample, dataset=dataset)
                    mc_exp = Experiment(time_limit_seconds=60)
                    mc_exp.run(anomaly_algo=DBSCANwrapper(),
                               solver=MonteCarloSolver(),
                               scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                               d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample, dataset=dataset)
                    # obo_exp = Experiment(time_limit_seconds=60)
                    # obo_exp.run(anomaly_algo=DBSCANwrapper(),
                    #             solver=OneOneSolver(d_tag_size=d_tag_size, f_diff_size=len(f_diff)),
                    #             scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                    #             d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample, dataset=dataset)
                    print()

                    # check results
                    # print("check results")
                    # best_ans_knn.append(knn_exp.results['best_ans'])
                    # ans_knn_shape.append(knn_exp.results['best_ans'].shape)
                    # best_ans_score_knn.append(knn_exp.results['best_ans_score'])
                    # knn_solving_time.append(knn_exp.results['solving_time'])

                    # print convergence
                    print("print convergence")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(np.array(knn_exp.convert_process["time"]), np.array(knn_exp.convert_process["score"]),
                            'o-', label='KNN Solver')
                    ax.plot(np.array(mc_exp.convert_process["time"]), np.array(mc_exp.convert_process["score"]),
                            'o-', label='Monte-Carlo Solver')
                    # ax.plot(np.array(obo_exp.convert_process["time"]), np.array(obo_exp.convert_process["score"]),
                    #         'o-', label='1-by-1 Solver')

                    fig.suptitle(f"Solvers Convergence Over Time", fontsize=20)
                    plt.xlabel("Time [sec]", fontsize=16)
                    plt.ylabel("AFEX Score", fontsize=16)

                    ax.legend()
                    plt.savefig(os.path.join(results_path, f"{os.path.basename(filename).split('.')[0]}_conv.png"))
                    # plt.show()

            # 4) Save experiments metadata
            # analysis_df = pd.DataFrame({'best_ans_score_knn': best_ans_score_knn, 'best_ans_score_mc': best_ans_score_mc,
            #                             'ans_knn_shape': ans_knn_shape, 'ans_mc_shape': ans_mc_shape,
            #                             'knn_solving_time': knn_solving_time, 'mc_solving_time': mc_solving_time})
            # analysis_df.to_csv(os.path.join(RESULTS_FOLDER_PATH, "meta_analysis.csv"), index=False)


if __name__ == '__main__':
    Main.run(create_corpus=False,
             corpus_name="DBSCAN_rc50_pmNone")
