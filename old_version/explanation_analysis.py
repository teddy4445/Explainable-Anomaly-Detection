# library imports
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

# project imports
from old_version.explanation_analysis.similarity_metrices.sim_euclidean_inverse import InverseEuclideanSim
from old_version.explanation_analysis.score_function import LinearScore
from visualization import project_fdiff


def calculate_distance(df):
    dist_matrix = np.zeros((df.shape[0], df.shape[0]))

    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(df.iloc[i] - df.iloc[j])
            else:
                dist_matrix[i, j] = np.nan

    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)


def plot_distance_matrices(d_dist, features, f_diff, save=False, save_filename=None, plot=False):
    dist_df = calculate_distance(d_dist[features])
    dist_df_diff = calculate_distance(d_dist[f_diff])
    dist_df_sim = calculate_distance(d_dist[[f for f in features if f not in f_diff]])

    if save:
        dist_df.to_csv(os.path.join(save_filename, "dist_mat.csv"), index=True)
        dist_df_diff.to_csv(os.path.join(save_filename, "dist_diff_mat.csv"), index=True)
        dist_df_sim.to_csv(os.path.join(save_filename, "dist_sim_mat.csv"), index=True)

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Calculate the color scale normalization based on the first matrix
        norm = Normalize(vmin=dist_df.min().min(), vmax=dist_df.max().max())

        sns.heatmap(dist_df, ax=axs[0], cmap='RdYlGn', norm=norm, annot=True)
        axs[0].set_title('Distance Matrix')

        sns.heatmap(dist_df_sim, ax=axs[1], cmap='RdYlGn', norm=norm, annot=True)  # Green to Red, RdYlGn_r
        axs[1].set_title('Distance Matrix - Over F_sim')

        sns.heatmap(dist_df_diff, ax=axs[2], cmap='RdYlGn', norm=norm, annot=True)  # Red to Green, RdYlGn
        axs[2].set_title('Distance Matrix - Over F_diff')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dict_main_key = "assoc"
    # filename = 'data/partial_synthetic/synt_iter0_inf_aug.csv'
    # filename = 'results/partial_synthetic/old/synt_iter0_inf_aug_knn5_fdiff_inf.csv'
    # filename = 'results/partial_synthetic/greedy45/synt_iter0_inf_aug_greedy4_inf.csv'
    # filename = 'results/partial_synthetic/TeddyKNN/synt_iter0_inf_aug_TeddyKNN2_inf.csv'
    filename = 'results/partial_T_Corpus_fixed/arcene/arcene_greedy4_inf.csv'

    d_inf = pd.read_csv(filename)
    dataset = d_inf[[feature for feature in d_inf.columns.values if feature != dict_main_key]]
    d_tag = dataset.loc[(d_inf[dict_main_key] == 1)]  # | (d_inf[dict_main_key] == 2)
    anomaly_sample = dataset.loc[d_inf[dict_main_key] == 2].iloc[-1]
    dataset_wo_anomaly = dataset.loc[d_inf[dict_main_key] != 2].reset_index(drop=True)

    scorer = LinearScore(sim_module=InverseEuclideanSim(),
                         w_self_sim=1,
                         w_local_sim=1, w_cluster_sim=0,
                         w_local_diff=1, w_cluster_diff=0,
                         w_cov=0, w_conc=0.08)

    # features = ['0', '1', '2', '3', '4']
    features = list(dataset.columns.values)
    # f_diff = ['0', '1', '4']
    # f_diff = ['3']
    f_diff = ['V9949']
    # f_diff = ['V9907', 'V9908', 'V9943', 'V9944', 'V9945', 'V9969', 'V9972', 'V9973', 'V9976', 'V9981', 'V9983']
    # f_diff = ['Adiponectin']  # ['Adiponectin'] , ['Glucose', 'HOMA'] , ['MCP.1'] , ['Glucose', 'HOMA']
    # f_diff = ['Leptin']  #
    # f_diff = ['Whether_of_not_the_TA_is_a_native_English_speaker', 'Course_instructor', 'Summer_or_regular_semester', 'Class_size']
    # f_diff = ['Course', 'Course_instructor', 'Class_size']

    f_sim = [f for f in features if f not in f_diff]
    ans_score, scores = scorer.compute(d=d_tag, s=anomaly_sample,
                                       f_sim=f_sim, f_diff=f_diff, overall_size=len(dataset))
    print("ans_score: ", ans_score)
    print(scores)

    project_fdiff(d_inf=d_inf, f_diff=f_diff, method='tsne', plot=True, annotate=False, save=False)

    d_dist = dataset.loc[(d_inf[dict_main_key] == 1) | (d_inf[dict_main_key] == 2)]
    # dist_df = calculate_distance(d_dist[features])
    # dist_df_diff = calculate_distance(d_dist[f_diff])
    # dist_df_sim = calculate_distance(d_dist[f_sim])
    # dist_df.to_csv(os.path.join(RESULTS_FOLDER_PATH, "partial_synthetic", "dist_mat_syn0_greedy4.csv"), index=True)
    # dist_df_diff.to_csv(os.path.join(RESULTS_FOLDER_PATH, "partial_synthetic", "dist_diff_syn0_greedy4.csv"), index=True)
    # dist_df_sim.to_csv(os.path.join(RESULTS_FOLDER_PATH, "partial_synthetic", "dist_sim_syn0_greedy4.csv"), index=True)
    plot_distance_matrices(d_dist=d_dist, features=features, f_diff=f_diff,
                           save=False, save_filename=None, plot=True)
