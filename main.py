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
from anomaly_detection_algos.IsolationForest import IsolationForestwrapper
from experiments.experiment import Experiment, TRACKED_METRICS
from experiments.experiment_properties.feature_distribution_normal import FeatureDistributionNormal
from experiments.synthetic_dataset_generation import SyntheticDatasetGeneration
from solvers.bf_solver import BruteForceSolver
from solvers.knn_solver import KnnSolver
from solvers.mc_solver import MonteCarloSolver
from solvers.one_ones_solver import OneOneSolver
from explanation_analysis.afes.afes_sum import AfesSum
from explanation_analysis.similarity_metrices.sim_euclidean import EuclideanSim
from explanation_analysis.similarity_metrices.sim_euclidean_inverse import InverseEuclideanSim
from explanation_analysis.similarity_metrices.sim_min_inverse import InverseMinSim


class Main:
    """
    The main class of the project, allow other developers to run all the experiments in the paper
    """

    def __init__(self):
        pass

    @staticmethod
    def create_corpus(iterations, features_num, row_count, f_diff, d_tag_size):
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
        return

        # @profile

    @staticmethod
    def solve_supervised_dataset(filename,
                                 exp_dict,
                                 dict_main_key='assoc',
                                 time_limit_seconds=60,
                                 save_d_inf=False,
                                 results_path=None):

        d_inf = pd.read_csv(filename)
        dataset = d_inf[[feature for feature in d_inf.columns.values if feature != dict_main_key]]
        d_tag = dataset.loc[(d_inf[dict_main_key] == 1) | (d_inf[dict_main_key] == 2)]
        anomaly_sample = dataset.loc[d_inf[dict_main_key] == 2].iloc[-1]
        dataset_wo_anomaly = dataset.loc[d_inf[dict_main_key] != 2].reset_index(drop=True)

        for exp_name, exp_data in exp_dict.items():
            print(exp_name)
            curr_exp = Experiment(time_limit_seconds=time_limit_seconds)
            curr_exp.run(anomaly_algo=DBSCANwrapper(),
                         solver=exp_data['solver'](param=exp_data['params']),
                         scorer=AfesSum(sim_module=InverseMinSim(), w_gsim=1, w_ldiff=1, w_lsim=1, w_cov=1),
                         anomaly_sample=anomaly_sample, dataset=dataset_wo_anomaly)

            # check results
            print("check results")
            # exp_dict[exp_name]['data_shape'].append(dataset.shape)
            for metric in TRACKED_METRICS:
                exp_dict[exp_name][metric].append(curr_exp.results[metric])

            if save_d_inf:
                curr_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                              f"{os.path.basename(filename).split('.')[0]}"
                                                              f"_{exp_name}_inf.csv"), index=False)

            # print convergence
            print("print score convergence")
            # Plotter.score_converge(exp_dict_list=exp_dict_list,
            #                        exp_names=exp_names,
            #                        save_path=os.path.join(results_path,
            #                                               f"{os.path.basename(filename).split('.')[0]}"
            #                                               f"_score_conv.png"))
            #
            # print("print shape convergence")
            # Plotter.shape_converge(exp_dict_list=exp_dict_list,
            #                        exp_names=exp_names,
            #                        save_path=os.path.join(results_path,
            #                                               f"{os.path.basename(filename).split('.')[0]}"
            #                                               f"_shape_conv.png"))

        return exp_dict

    @staticmethod
    def solve_unsupervised_dataset(filename,
                                   exp_dict,
                                   time_limit_seconds=60,
                                   save_d_inf=False,
                                   results_path=None):
        dataset = pd.read_csv(filename)
        print()

        # find an anomaly
        ad_algo = IsolationForestwrapper()
        ad_algo.fit(dataset)
        labels = ad_algo.predict(dataset)
        anomaly_indexes = [i for i in range(len(labels)) if labels[i] == 1]

        anomaly_sample = dataset.iloc[anomaly_indexes[0]]  # first anomaly
        dataset_wo_anomaly = pd.concat([dataset.iloc[:anomaly_indexes[0]], dataset.iloc[anomaly_indexes[0] + 1:]],
                                       ignore_index=True)

        for exp_name, exp_data in exp_dict.items():
            print(exp_name)
            curr_exp = Experiment(time_limit_seconds=time_limit_seconds)
            curr_exp.run(anomaly_algo=DBSCANwrapper(),
                         solver=exp_data['solver'](param=exp_data['params']),
                         scorer=AfesSum(sim_module=InverseMinSim(), w_gsim=1, w_ldiff=0.1, w_lsim=10, w_cov=1),
                         anomaly_sample=anomaly_sample, dataset=dataset_wo_anomaly)

            # check results
            print("check results")
            # exp_dict[exp_name]['data_shape'].append(dataset.shape)
            for metric in TRACKED_METRICS:
                exp_dict[exp_name][metric].append(curr_exp.results[metric])

            if save_d_inf:
                curr_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                              f"{os.path.basename(filename).split('.')[0]}"
                                                              f"_{exp_name}_inf.csv"), index=False)

            # print convergence
            print("print score convergence")
            # Plotter.score_converge(exp_dict_list=exp_dict_list,
            #                        exp_names=exp_names,
            #                        save_path=os.path.join(results_path,
            #                                               f"{os.path.basename(filename).split('.')[0]}"
            #                                               f"_score_conv.png"))
            #
            # print("print shape convergence")
            # Plotter.shape_converge(exp_dict_list=exp_dict_list,
            #                        exp_names=exp_names,
            #                        save_path=os.path.join(results_path,
            #                                               f"{os.path.basename(filename).split('.')[0]}"
            #                                               f"_shape_conv.png"))

        return exp_dict

    @staticmethod
    def save_metadata(analysis_dict_2df, exp_dict):
        for metric in TRACKED_METRICS:
            if metric != 'd_tag':
                for exp_name, exp_data in exp_dict.items():
                    analysis_dict_2df[f"{exp_name}_{metric}"] = exp_data[metric]
        return pd.DataFrame(analysis_dict_2df)

    @staticmethod
    def run(create_corpus: bool = False,
            run_experiments: bool = True,
            iterations: int = 50,
            features_num: int = 5,
            row_count: int = 50,
            f_diff: list = None,
            d_tag_size: int = 10,
            time_limit_seconds: int = 120,
            corpus_name: str = "DBSCAN_rc50_pmNone",
            supervised: bool = True):
        """
        Single entry point - running all the experiments logic
        """

        # 0) default value setting
        if f_diff is None or not isinstance(f_diff, list):
            f_diff = ['0', '1']

        # 1) prepare IO
        for path in SETUP_FOLDERS:
            os.makedirs(path, exist_ok=True)

        # 2) generate dataset
        if create_corpus:
            Main.create_corpus(iterations=iterations, features_num=features_num, row_count=row_count, f_diff=f_diff,
                               d_tag_size=d_tag_size)

        # 3) run experiments
        if run_experiments:
            # the key for the dict of the meta-data, declare once so will be the same everywhere
            # dict_main_key = 'assoc'
            results_path = os.path.join(RESULTS_FOLDER_PATH, corpus_name)
            os.makedirs(results_path, exist_ok=True)

            # Set-up experiment
            print(f"Set-up experiment")
            file_names = []

            exp_dict = {'knn5_fdiff': {'solver': KnnSolver, 'params': {"k": 5, "f_diff": f_diff}},
                        'knn5': {'solver': KnnSolver, 'params': {"k": 5}},
                        # 'knn10_fdiff': {'solver': KnnSolver, 'params': {"k": 10, "f_diff": f_diff}},
                        # 'knn10': {'solver': KnnSolver, 'params': {"k": 10}},
                        # 'knn15_fdiff': {'solver': KnnSolver, 'params': {"k": 15, "f_diff": f_diff}},
                        # 'knn15': {'solver': KnnSolver, 'params': {"k": 15}},
                        # 'mc': {'solver': MonteCarloSolver, 'params': {}},
                        # 'bf1': {'solver': BruteForceSolver, 'params': {'columns': ['0', '1'], 'rows_num': 1}},
                        # 'bf2': {'solver': BruteForceSolver, 'params': {'columns': ['0', '1'], 'rows_num': 2}},
                        # 'bf3': {'solver': BruteForceSolver, 'params': {'columns': ['0', '1'], 'rows_num': 3}},
                        # 'bf4': {'solver': BruteForceSolver, 'params': {'columns': ['0', '1'], 'rows_num': 4}},
                        # 'bf5': {'solver': BruteForceSolver, 'params': {'columns': ['0', '1'], 'rows_num': 5}},
                        }

            for exp_name, exp_data in exp_dict.items():
                for metric in TRACKED_METRICS:
                    exp_data[metric] = []

            print()

            # run experiments
            print(f"run experiments")
            for filename in tqdm(os.scandir(os.path.join(DATA_FOLDER_PATH, corpus_name))):
                print(filename)
                if filename.is_file():
                    file_names.append(os.path.basename(filename).split('.')[0])
                    if supervised:
                        exp_dict = Main.solve_supervised_dataset(filename=filename,
                                                                 exp_dict=exp_dict,
                                                                 dict_main_key='assoc',
                                                                 time_limit_seconds=time_limit_seconds,
                                                                 save_d_inf=True,
                                                                 results_path=results_path)

                    else:
                        exp_dict = Main.solve_unsupervised_dataset(filename=filename,
                                                                   exp_dict=exp_dict,
                                                                   time_limit_seconds=time_limit_seconds,
                                                                   save_d_inf=True,
                                                                   results_path=results_path)
                    print()

            # 4) Save experiments metadata
            print('Save experiments metadata')
            analysis_dict_2df = {'dataset': file_names}
            analysis_df = Main.save_metadata(analysis_dict_2df=analysis_dict_2df, exp_dict=exp_dict)
            analysis_df.T.to_csv(os.path.join(results_path, "meta_analysis_knn.csv"), index=True)


if __name__ == '__main__':
    # Main.run(create_corpus=False,
    #          iterations=50,
    #          features_num=5,
    #          row_count=50,
    #          d_tag_size=10,
    #          f_diff=None,
    #          run_experiments=True,
    #          supervised=False,
    #          time_limit_seconds=60,
    #          corpus_name="T_Corpus")

    Main.run(create_corpus=False,
             f_diff=None,
             run_experiments=True,
             supervised=True,
             time_limit_seconds=60,
             corpus_name="partial_synthetic")  # DBSCAN_rc50_pmNone
