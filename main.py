# library imports
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
from tqdm import tqdm
from memory_profiler import profile

# project imports
from consts import *
from plotter import Plotter
from anomaly_detection_algos.DBSCAN import DBSCANwrapper
from experiments.experiment import Experiment
from experiments.experiment_properties.feature_distribution_normal import FeatureDistributionNormal
from experiments.synthetic_dataset_generation import SyntheticDatasetGeneration
from solvers.knn_solver import KnnSolver
from solvers.mc_solver import MonteCarloSolver
from solvers.one_ones_solver import OneOneSolver
from explanation_analysis.afes.afes_sum import AfesSum
from explanation_analysis.similarity_metrices.sim_euclidean import EuclideanSim
class Main:
    """
    The main class of the project, allow other developers to run all the experiments in the paper
    """

    def __init__(self):
        pass

    # @profile
    @staticmethod
    def run(create_corpus: bool = False,
            run_experiments: bool = True,
            iterations: int = 50,
            features_num: int = 5,
            row_count: int = 50,
            f_diff: list = None,
            d_tag_size: int = 10,
            time_limit_seconds: int = 120,
            corpus_name: str = "DBSCAN_rc50_pmNone"):
        """
        Single entry point - running all the experiments logic
        """

        # 0) default value setting
        if f_diff is None or not isinstance(f_diff, list):
            f_diff = [0, 1]

        # 1) prepare IO
        for path in SETUP_FOLDERS:
            os.makedirs(path, exist_ok=True)

        # 2) generate dataset
        if create_corpus:
            anom_algo_name = DBSCANwrapper.NAME
            performance_metric = None  # InverseEuclideanSim, EuclideanSim()
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

                SyntheticDatasetGeneration.generate_one(
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
        if run_experiments:
            results_path = os.path.join(RESULTS_FOLDER_PATH, corpus_name)
            os.makedirs(results_path,
                        exist_ok=True)

            # Set-up experiment
            print(f"Set-up experiment")
            file_names = []
            # best_ans_knn, ans_knn_shape, best_ans_score_knn, knn_solving_time = [], [], [], []
            best_ans_knn5, ans_knn5_shape, best_ans_score_knn5, knn5_solving_time = [], [], [], []
            best_ans_knn10, ans_knn10_shape, best_ans_score_knn10, knn10_solving_time = [], [], [], []
            best_ans_knn15, ans_knn15_shape, best_ans_score_knn15, knn15_solving_time = [], [], [], []
            best_ans_mc, ans_mc_shape, best_ans_score_mc, mc_solving_time = [], [], [], []
            best_ans_obo, ans_obo_shape, best_ans_score_obo, obo_solving_time = [], [], [], []

            # run experiments
            print(f"run experiments")
            dict_main_key = 'assoc'  # the key for the dict of the meta-data, declare once so will be the same everywhere
            for filename in tqdm(os.scandir(os.path.join(DATA_FOLDER_PATH, corpus_name))):
                print(filename)
                if filename.is_file():
                    file_names.append(os.path.basename(filename).split('.')[0])
                    d_inf = pd.read_csv(filename)
                    dataset = d_inf[[feature for feature in d_inf.columns.values if feature != dict_main_key]]
                    d_tag = dataset.loc[(d_inf[dict_main_key] == 1) | (d_inf[dict_main_key] == 2)]
                    anomaly_sample = dataset.loc[d_inf[dict_main_key] == 2].iloc[-1]
                    dataset_wo_anomaly = dataset.loc[d_inf[dict_main_key] != 2].reset_index(drop=True)

                    print('knn')
                    knn5_exp = Experiment(time_limit_seconds=time_limit_seconds)
                    knn5_exp.run(anomaly_algo=DBSCANwrapper(),
                                 solver=KnnSolver(param={"k": 5}),
                                 scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                                 d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample,
                                 dataset=dataset_wo_anomaly)
                    knn10_exp = Experiment(time_limit_seconds=time_limit_seconds)
                    knn10_exp.run(anomaly_algo=DBSCANwrapper(),
                                  solver=KnnSolver(param={"k": 10}),
                                  scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                                  d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample,
                                  dataset=dataset_wo_anomaly)
                    knn15_exp = Experiment(time_limit_seconds=time_limit_seconds)
                    knn15_exp.run(anomaly_algo=DBSCANwrapper(),
                                  solver=KnnSolver(param={"k": 15}),
                                  scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                                  d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample,
                                  dataset=dataset_wo_anomaly)
                    print('mc')
                    mc_exp = Experiment(time_limit_seconds=time_limit_seconds)
                    mc_exp.run(anomaly_algo=DBSCANwrapper(),
                               solver=MonteCarloSolver(),
                               scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                               d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample,
                               dataset=dataset_wo_anomaly)
                    print('obo')
                    obo_exp = Experiment(time_limit_seconds=time_limit_seconds)
                    obo_exp.run(anomaly_algo=DBSCANwrapper(),
                                solver=OneOneSolver(param={"d_tag_size": d_tag_size, "f_diff_size": len(f_diff)}),
                                scorer=AfesSum(sim_module=EuclideanSim(), w_gsim=1, w_ldiff=1, w_lsim=1),
                                d_tags=[d_tag], f_diff_list=[f_diff], anomaly_sample=anomaly_sample,
                                dataset=dataset_wo_anomaly)
                    print()

                    # check results
                    print("check results")
                    best_ans_knn5.append(knn5_exp.results['best_ans'])
                    # ans_knn5_shape.append(knn5_exp.results['best_ans'].shape)
                    best_ans_score_knn5.append(knn5_exp.results['best_ans_score'])
                    knn5_solving_time.append(knn5_exp.results['solving_time'])
                    knn5_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                                  f"{os.path.basename(filename).split('.')[0]}"
                                                                  f"_knn5_inf.csv"), index=False)

                    best_ans_knn10.append(knn10_exp.results['best_ans'])
                    # ans_knn10_shape.append(knn10_exp.results['best_ans'].shape)
                    best_ans_score_knn10.append(knn10_exp.results['best_ans_score'])
                    knn10_solving_time.append(knn10_exp.results['solving_time'])
                    knn10_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                                   f"{os.path.basename(filename).split('.')[0]}"
                                                                   f"_knn10_inf.csv"), index=False)

                    best_ans_knn15.append(knn15_exp.results['best_ans'])
                    # ans_knn15_shape.append(knn15_exp.results['best_ans'].shape)
                    best_ans_score_knn15.append(knn15_exp.results['best_ans_score'])
                    knn15_solving_time.append(knn15_exp.results['solving_time'])
                    knn15_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                                   f"{os.path.basename(filename).split('.')[0]}"
                                                                   f"_knn15_inf.csv"), index=False)

                    best_ans_mc.append(mc_exp.results['best_ans'])
                    # ans_mc_shape.append(mc_exp.results['best_ans'].shape)
                    best_ans_score_mc.append(mc_exp.results['best_ans_score'])
                    mc_solving_time.append(mc_exp.results['solving_time'])
                    mc_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                                f"{os.path.basename(filename).split('.')[0]}"
                                                                f"_mc_inf.csv"), index=False)

                    best_ans_obo.append(obo_exp.results['best_ans'])
                    # ans_obo_shape.append(obo_exp.results['best_ans'].shape)
                    best_ans_score_obo.append(obo_exp.results['best_ans_score'])
                    obo_solving_time.append(obo_exp.results['solving_time'])
                    obo_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                                 f"{os.path.basename(filename).split('.')[0]}"
                                                                 f"_obo_inf.csv"), index=False)

                    # print convergence
                    print("print score convergence")
                    # Plotter.score_converge(exp_dict_list=[knn_exp, mc_exp, obo_exp],
                    #                        exp_names=["KNN", "MC", "OBO"],
                    #                        save_path=os.path.join(results_path,
                    #                                               f"{os.path.basename(filename).split('.')[0]}"
                    #                                               f"_score_conv.png"))
                    Plotter.score_converge(exp_dict_list=[knn5_exp, knn10_exp, knn15_exp, mc_exp, obo_exp],
                                           exp_names=["KNN5", "KNN10", "KNN15", "MC", "OBO"],
                                           save_path=os.path.join(results_path,
                                                                  f"{os.path.basename(filename).split('.')[0]}"
                                                                  f"_score_conv.png"))

                    print("print shape convergence")
                    # Plotter.shape_converge(exp_dict_list=[knn_exp, mc_exp, obo_exp],
                    #                        exp_names=["KNN", "MC", "OBO"],
                    #                        save_path=os.path.join(results_path,
                    #                                               f"{os.path.basename(filename).split('.')[0]}"
                    #                                               f"_shape_conv.png"))
                    Plotter.shape_converge(exp_dict_list=[knn5_exp, knn10_exp, knn15_exp, mc_exp, obo_exp],
                                           exp_names=["KNN5", "KNN10", "KNN15", "MC", "OBO"],
                                           save_path=os.path.join(results_path,
                                                                  f"{os.path.basename(filename).split('.')[0]}"
                                                                  f"_shape_conv.png"))

                    print()
                    # break

            # 4) Save experiments metadata
            analysis_df = pd.DataFrame({'dataset': file_names,
                                        'knn5_score': best_ans_score_knn5,
                                        'knn10_score': best_ans_score_knn10,
                                        'knn15_score': best_ans_score_knn15,
                                        'mc_score': best_ans_score_mc,
                                        'obo_score': best_ans_score_obo,
                                        'knn5_features': [ans.columns.values for ans in best_ans_knn5],
                                        'knn10_features': [ans.columns.values for ans in best_ans_knn10],
                                        'knn15_features': [ans.columns.values for ans in best_ans_knn15],
                                        'mc_features': [ans.columns.values for ans in best_ans_mc],
                                        'obo_features': [ans.columns.values for ans in best_ans_obo],
                                        'knn5_shape': [ans.shape for ans in best_ans_knn5],
                                        'knn10_shape': [ans.shape for ans in best_ans_knn10],
                                        'knn15_shape': [ans.shape for ans in best_ans_knn15],
                                        'mc_shape': [ans.shape for ans in best_ans_mc],
                                        'obo_shape': [ans.shape for ans in best_ans_obo],
                                        'knn5_solving_time': knn5_solving_time,
                                        'knn10_solving_time': knn10_solving_time,
                                        'knn15_solving_time': knn15_solving_time,
                                        'mc_solving_time': mc_solving_time,
                                        'obo_solving_time': obo_solving_time})
            # analysis_df = pd.DataFrame({'best_ans_score_knn': best_ans_score_knn,
            #                             'best_ans_score_mc': best_ans_score_mc,
            #                             'best_ans_score_obo': best_ans_score_obo,
            #                             'ans_knn_shape': ans_knn_shape[-1],
            #                             'ans_mc_shape': ans_mc_shape[-1],
            #                             'ans_obo_shape': ans_mc_shape[-1],
            #                             'knn_solving_time': knn_solving_time,
            #                             'mc_solving_time': mc_solving_time,
            #                             'obo_solving_time': obo_solving_time})
            analysis_df.T.to_csv(os.path.join(results_path, "meta_analysis.csv"), index=True)


if __name__ == '__main__':
    Main.run(create_corpus=False,
             run_experiments=True,
             iterations=50,
             features_num=5,
             row_count=50,
             f_diff=None,
             d_tag_size=10,
             time_limit_seconds=300,
             corpus_name="DBSCAN_rc50_pmNone")


from explanation_analysis.similarity_metrices.sim_euclidean_inverse import InverseEuclideanSim
