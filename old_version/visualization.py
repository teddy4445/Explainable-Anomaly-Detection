import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# project imports
from consts import *


def scale_df(df):
    return (df - df.min()) / (df.max() - df.min())


def project_2d(d_inf, method='tsne', plot=False):
    # reading the CSV file
    d = d_inf[[feature for feature in d_inf.columns.values if feature != 'assoc']]
    d_tag_c = d.loc[d_inf['assoc'] == 0]
    d_tag = d.loc[d_inf['assoc'] == 1]
    anomaly = d.loc[d_inf['assoc'] == 2]

    # create the 2D scatter plot
    # fig, ax = plt.subplots()

    if len(d.columns) == 1:
        x_dtc = d_tag_c.values.reshape(1, -1)[0]
        y_dtc = 0.5 * np.ones_like(x_dtc)
        x_dt = d_tag.values.reshape(1, -1)[0]
        y_dt = 0.5 * np.ones_like(x_dt)
        x_anom = anomaly.values.reshape(1, -1)[0]
        y_anom = 0.5 * np.ones_like(x_anom)

    # transform
    elif method == "pca":
        pca = PCA(n_components=2)
        pca.fit(d)
        reduced_d_tag_c = pca.transform(d_tag_c)
        reduced_d_tag = pca.transform(d_tag)
        reduced_anomaly = pca.transform(anomaly)

        x_dtc, y_dtc = reduced_d_tag_c[:, 0], reduced_d_tag_c[:, 1]
        x_dt, y_dt = reduced_d_tag[:, 0], reduced_d_tag[:, 1]
        x_anom, y_anom = reduced_anomaly[:, 0], reduced_anomaly[:, 1]

    elif method == "tsne":
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0)
        d_tsne = pd.DataFrame(tsne.fit_transform(d))
        d_tsne = scale_df(d_tsne)
        reduced_d_tag_c = d_tsne.loc[d_inf['assoc'] == 0]
        reduced_d_tag = d_tsne.loc[d_inf['assoc'] == 1]
        reduced_anomaly = d_tsne.loc[d_inf['assoc'] == 2]

        x_dtc, y_dtc = reduced_d_tag_c[0], reduced_d_tag_c[1]
        x_dt, y_dt = reduced_d_tag[0], reduced_d_tag[1]
        x_anom, y_anom = reduced_anomaly[0], reduced_anomaly[1]

    else:
        return None

    if plot:
        plt.show()
    proj_data = {"x_dtc": x_dtc, "y_dtc": y_dtc,
                 "x_dt": x_dt, "y_dt": y_dt,
                 "x_anom": x_anom, "y_anom": y_anom}
    return proj_data


def project_fdiff(d_inf, f_diff, method='tsne', plot=True, annotate=False, save=False):
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
    fig.suptitle('Dataset Projection', fontsize=20)

    proj_full = project_2d(d_inf, method=method)
    axs[0].scatter(proj_full["x_dtc"], proj_full["y_dtc"], c='y', label='D')
    axs[0].scatter(proj_full["x_dt"], proj_full["y_dt"], c='r', label='D Tag')
    axs[0].scatter(proj_full["x_anom"], proj_full["y_anom"], c='k', label='s')
    axs[0].set_xlabel(f'{method} 1', fontsize=12)
    axs[0].set_ylabel(f'{method} 2', fontsize=12)
    if annotate:
        for i in range(len(proj_full["x_dtc"])):
            if isinstance(proj_full["x_dtc"], np.ndarray):
                axs[0].annotate(f"{i}", (proj_full["x_dtc"][i], proj_full["y_dtc"][i]),
                                xycoords="axes fraction")
            else:
                axs[0].annotate(f"{i}", (proj_full["x_dtc"].values[i], proj_full["y_dtc"].values[i]),
                                xycoords="axes fraction")
        for i in range(len(proj_full["x_dt"])):
            if isinstance(proj_full["x_dt"], np.ndarray):
                axs[0].annotate(f"{i + len(proj_full['x_dtc'])}",
                                (proj_full["x_dt"][i], proj_full["y_dt"][i]),
                                xycoords="axes fraction")
            else:
                axs[0].annotate(f"{i + len(proj_full['x_dtc'])}",
                                (proj_full["x_dt"].values[i], proj_full["y_dt"].values[i]),
                                xycoords="axes fraction")
        axs[0].annotate(f"{len(d_inf) - 1}", (proj_full["x_anom"], proj_full["y_anom"]), xycoords="axes fraction")
    else:
        axs[0].legend()
    axs[0].set_title(f"D - {method} projection", fontsize=16)

    proj_fsim = project_2d(d_inf[[f for f in d_inf.columns.values if f not in f_diff]], method=method)
    axs[1].scatter(proj_fsim["x_dtc"], proj_fsim["y_dtc"], c='y', label='D')
    axs[1].scatter(proj_fsim["x_dt"], proj_fsim["y_dt"], c='r', label='D Tag')
    axs[1].scatter(proj_fsim["x_anom"], proj_fsim["y_anom"], c='k', label='s')
    axs[1].set_xlabel(f'{method} 1', fontsize=12)
    axs[1].set_ylabel(f'{method} 2', fontsize=12)
    if annotate:
        for i in range(len(proj_fsim["x_dtc"])):
            if isinstance(proj_fsim["x_dtc"], np.ndarray):
                axs[1].annotate(f"{i}", (proj_fsim["x_dtc"][i], proj_fsim["y_dtc"][i]),
                                xycoords="axes fraction")
            else:
                axs[1].annotate(f"{i}", (proj_fsim["x_dtc"].values[i], proj_fsim["y_dtc"].values[i]),
                                xycoords="axes fraction")
        for i in range(len(proj_fsim["x_dt"])):
            if isinstance(proj_fsim["x_dt"], np.ndarray):
                axs[1].annotate(f"{i + len(proj_fsim['x_dtc'])}",
                                (proj_fsim["x_dt"][i], proj_fsim["y_dt"][i]),
                                xycoords="axes fraction")
            else:
                axs[1].annotate(f"{i + len(proj_fsim['x_dtc'])}",
                                (proj_fsim["x_dt"].values[i], proj_fsim["y_dt"].values[i]),
                                xycoords="axes fraction")
        axs[1].annotate(f"{len(d_inf) - 1}", (proj_fsim["x_anom"], proj_fsim["y_anom"]), xycoords="axes fraction")
    else:
        axs[1].legend()
    axs[1].set_title(f"D[f_sim] = D{[f for f in d_inf.columns.values if f not in f_diff + ['assoc']]} "
                     f"- {method} projection", fontsize=16)

    proj_fdiff = project_2d(d_inf[f_diff + ['assoc']], method=method)
    axs[2].scatter(proj_fdiff["x_dtc"], proj_fdiff["y_dtc"], c='y', label='D')
    axs[2].scatter(proj_fdiff["x_dt"], proj_fdiff["y_dt"], c='r', label='D Tag')
    axs[2].scatter(proj_fdiff["x_anom"], proj_fdiff["y_anom"], c='k', label='s')
    axs[2].set_xlabel(f'{method} 1', fontsize=12)
    axs[2].set_ylabel(f'{method} 2', fontsize=12)
    if annotate:
        for i in range(len(proj_fdiff["x_dtc"])):
            if isinstance(proj_fdiff["x_dtc"], np.ndarray):
                axs[2].annotate(f"{i}", (proj_fdiff["x_dtc"][i], proj_fdiff["y_dtc"][i]),
                                xycoords="axes fraction")
            else:
                axs[2].annotate(f"{i}", (proj_fdiff["x_dtc"].values[i], proj_fdiff["y_dtc"].values[i]),
                                xycoords="axes fraction")
        for i in range(len(proj_fdiff["x_dt"])):
            if isinstance(proj_fdiff["x_dt"], np.ndarray):
                axs[2].annotate(f"{i + len(proj_fdiff['x_dtc'])}",
                                (proj_fdiff["x_dt"][i], proj_fdiff["y_dt"][i]),
                                xycoords="axes fraction")
            else:
                axs[2].annotate(f"{i + len(proj_fdiff['x_dtc'])}",
                                (proj_fdiff["x_dt"].values[i], proj_fdiff["y_dt"].values[i]),
                                xycoords="axes fraction")
        axs[2].annotate(f"{len(d_inf) - 1}", (proj_fdiff["x_anom"], proj_fdiff["y_anom"]), xycoords="axes fraction")
    else:
        axs[2].legend()
    axs[2].set_title(f"D[f_diff] = D{f_diff} - {method} projection", fontsize=16)

    if plot:
        plt.show()
    if save:
        plt.savefig(os.path.join(RESULTS_FOLDER_PATH, "Projections", f"{os.path.basename(file_name)}_proj.png"))
    return


if __name__ == '__main__':
    # file_name = 'data/DBSCAN_rc50_pmNone/synt_iter0_inf.csv'
    # file_name = 'data/DBSCAN_rc50_pmNone_aug/synt_iter0_inf.csv'
    # file_name = 'data/partial_synthetic/synt_iter0_inf_aug.csv'
    file_name = '../data/partial_T_Corpus_fixed/dataset_48_tae.csv'
    # file_name = 'results/partial_synthetic/old/synt_iter0_inf_aug_knn5_fdiff_inf.csv'
    # file_name = 'results/partial_synthetic/synt_iter0_inf_aug_bf2_inf.csv'
    # file_name = 'results/partial_synthetic/synt_iter6_inf_aug_greedy_inf.csv'
    # file_name = 'results/partial_T_Corpus_fixed/arcene/arcene_greedy5_inf.csv'
    print()

    # reading the CSV file
    dataset = pd.read_csv(file_name)

    # scatter_3d(d_inf)
    # f_diff = ['0', '1']
    # f_diff = ['0']

    # f_diff = ['V9902']
    # f_diff = ['Age', 'BMI']
    f_diff = ['Whether_of_not_the_TA_is_a_native_English_speaker', 'Course_instructor']

    project_fdiff(d_inf=dataset, f_diff=f_diff, method='tsne', plot=True, annotate=False, save=False)
    # project_fdiff(d_inf=dataset, f_diff=f_diff, method='pca', plot=True, annotate=False, save=False)