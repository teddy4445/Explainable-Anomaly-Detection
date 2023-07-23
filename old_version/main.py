# library imports
import pandas as pd

# project imports
from consts import *
from old_version.anomaly_detection_algos.DBSCAN import DBSCANwrapper
from old_version.anomaly_detection_algos.IsolationForest import IsolationForestwrapper
from old_version.experiments import Experiment, TRACKED_METRICS
from old_version.solvers import GreedySolver
from old_version.explanation_analysis.score_function import LinearScore
from old_version.explanation_analysis.similarity_metrices.sim_euclidean_inverse import InverseEuclideanSim


class Main:
    """
    The main class of the project, allow other developers to run all the experiments in the paper
    """

    def __init__(self):
        pass

    @staticmethod
    def get_anomaly(filename, supervised=True):
        df = pd.read_csv(filename)
        if supervised:
            dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
            anomaly_sample = dataset.loc[df['assoc'] == 2].iloc[-1]
            dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)
        else:
            # find an anomaly
            ad_algo = IsolationForestwrapper()
            ad_algo.fit(df)
            labels = ad_algo.predict(df)
            anomaly_indexes = [i for i in range(len(labels)) if labels[i] == 1]

            anomaly_sample = df.iloc[anomaly_indexes[0]]  # first anomaly
            dataset_wo_anomaly = pd.concat(
                [df.iloc[:anomaly_indexes[0]], df.iloc[anomaly_indexes[0] + 1:]],
                ignore_index=True)

        return anomaly_sample, dataset_wo_anomaly

    @staticmethod
    def solve_dataset(filename, supervised, exp_dict, scorer,
                      time_limit_seconds=60,
                      save_d_inf=False,
                      results_path=None):

        anomaly_sample, dataset_wo_anomaly = Main.get_anomaly(filename=filename, supervised=supervised)

        for exp_name, exp_data in exp_dict.items():
            print(exp_name)
            curr_exp = Experiment(time_limit_seconds=time_limit_seconds)
            curr_exp.run(anomaly_algo=DBSCANwrapper(),
                         solver=exp_data['solver'](param=exp_data['params']),
                         scorer=scorer,
                         anomaly_sample=anomaly_sample, dataset=dataset_wo_anomaly)

            for metric in TRACKED_METRICS:
                exp_dict[exp_name][metric].append(curr_exp.results[metric])

            if save_d_inf:
                curr_exp.results['d_inf'].to_csv(os.path.join(results_path,
                                                              f"{os.path.basename(filename).split('.')[0]}"
                                                              f"_{exp_name}_inf.csv"), index=False)

        return exp_dict

    @staticmethod
    def save_metadata(analysis_dict_2df, exp_dict):
        for metric in TRACKED_METRICS:
            if metric != 'd_tag':
                for exp_name, exp_data in exp_dict.items():
                    analysis_dict_2df[f"{exp_name}_{metric}"] = exp_data[metric]
        return pd.DataFrame(analysis_dict_2df)

    @staticmethod
    def run(exp_dict,
            scorer,
            time_limit_seconds: int = 120,
            corpus_name: str = "DBSCAN_rc50_pmNone",
            supervised: bool = True):
        """
        Single entry point - running all the experiments logic
        """
        # prepare IO
        for path in SETUP_FOLDERS:
            os.makedirs(path, exist_ok=True)

        # the key for the dict of the meta-data, declare once so will be the same everywhere
        results_path = os.path.join(RESULTS_FOLDER_PATH, corpus_name)
        os.makedirs(results_path, exist_ok=True)

        # Set-up experiment
        print(f"Set-up experiment")
        file_names = []

        for exp_name, exp_data in exp_dict.items():
            for metric in TRACKED_METRICS:
                exp_data[metric] = []

        # run experiments
        print(f"run experiments")
        for filename in os.scandir(os.path.join(DATA_FOLDER_PATH, corpus_name)):
            print(filename)
            if filename.is_file():
                file_names.append(os.path.basename(filename).split('.')[0])

                exp_dict = Main.solve_dataset(filename=filename,
                                              supervised=supervised,
                                              exp_dict=exp_dict,
                                              scorer=scorer,
                                              time_limit_seconds=time_limit_seconds,
                                              save_d_inf=True,
                                              results_path=results_path)

        # Save experiments metadata
        print('Save experiments metadata')
        analysis_dict_2df = {'dataset': file_names}
        analysis_df = Main.save_metadata(analysis_dict_2df=analysis_dict_2df, exp_dict=exp_dict)
        analysis_df.T.to_csv(os.path.join(results_path, "meta_analysis.csv"), index=True)


if __name__ == '__main__':
    scorer = LinearScore(sim_module=InverseEuclideanSim(),
                         w_self_sim=1,
                         w_local_sim=1, w_cluster_sim=0,
                         w_local_diff=1, w_cluster_diff=0,
                         w_cov=0, w_conc=0.08)
    exp_dict = {
        # 'knn5_fdiff': {'solver': KnnSolver, 'params': {"k": 5, "f_diff": f_diff}},
        # 'knn5': {'solver': KnnSolver, 'params': {"k": 5}},
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
        # 'bf_full': {'solver': BruteForceSolver, 'params': {}},
        'greedy4': {'solver': GreedySolver, 'params': {'depth': 4}},
        # 'greedy5': {'solver': GreedySolver, 'params': {'depth': 5}},
        # 'greedy': {'solver': GreedySolver, 'params': {'depth': -1}},
        # 'TeddyKNN2': {'solver': TeddyKnnSolver, 'params': {'k': 2}},
        # 'TeddyKNN3': {'solver': TeddyKnnSolver, 'params': {'k': 3}},
        # 'TeddyKNN4': {'solver': TeddyKnnSolver, 'params': {'k': 4}},
        # 'TeddyKNN5': {'solver': TeddyKnnSolver, 'params': {'k': 5}},
    }

    Main.run(exp_dict=exp_dict,
             scorer=scorer,
             supervised=True,
             time_limit_seconds=600,
             corpus_name="g")  # DBSCAN_rc50_pmNone, partial_synthetic, T_Corpus_fixed, partial_T_Corpus_fixed
