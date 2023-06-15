# library imports
import pandas as pd
from typing import Callable

# project imports
from explanation_analysis.similarity_metrices.sim_euclidean_inverse import InverseEuclideanSim
from explanation_analysis.score_function.linear_score import LinearScore
from visualization import project_fdiff


if __name__ == '__main__':
    dict_main_key = "assoc"
    # filename = 'data/partial_synthetic/synt_iter0_inf_aug.csv'
    # filename = 'results/partial_synthetic/old/synt_iter0_inf_aug_knn5_fdiff_inf.csv'
    filename = 'results/partial_synthetic/old/synt_iter0_inf_aug_knn5_inf.csv'
    # filename = 'results/partial_synthetic/old/synt_iter0_inf_aug_bf5_inf.csv'

    d_inf = pd.read_csv(filename)
    dataset = d_inf[[feature for feature in d_inf.columns.values if feature != dict_main_key]]
    d_tag = dataset.loc[(d_inf[dict_main_key] == 1) | (d_inf[dict_main_key] == 2)]
    anomaly_sample = dataset.loc[d_inf[dict_main_key] == 2].iloc[-1]
    dataset_wo_anomaly = dataset.loc[d_inf[dict_main_key] != 2].reset_index(drop=True)

    scorer = LinearScore(sim_module=InverseEuclideanSim(),
                         w_self_sim=1,
                         w_local_sim=1, w_cluster_sim=1,
                         w_local_diff=1, w_cluster_diff=1,
                         w_cov=0)

    features = ['0', '1', '2', '3', '4']
    f_diff = ['0', '1', '2', '4']
    f_sim = [f for f in features if f not in f_diff]
    ans_score, scores = scorer.compute(d=d_tag, s=anomaly_sample,
                                       f_sim=f_sim, f_diff=f_diff, overall_size=len(dataset))
    print("ans_score: ", ans_score)
    print(scores)

    project_fdiff(d_inf=d_inf, f_diff=f_diff, method='tsne', plot=True, annotate=False, save=False)

