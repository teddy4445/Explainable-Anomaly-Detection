import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# project imports
from consts import *


def scale_df(df):
    return (df - df.min()) / (df.max() - df.min())


def scatter_3d(d_inf):
    d = d_inf[[feature for feature in d_inf.columns.values if feature != 'assoc']]
    d_tag_c = d.loc[d_inf['assoc'] == 0]
    d_tag = d.loc[d_inf['assoc'] == 1]
    anomaly = d.loc[d_inf['assoc'] == 2]

    # generate 3D data
    x, y, z = d_tag_c['0'].values, d_tag_c['1'].values, d_tag_c['2'].values
    x_tag, y_tag, z_tag = d_tag['0'].values, d_tag['1'].values, d_tag['2'].values
    x_anom, y_anom, z_anom = anomaly['0'], anomaly['1'], anomaly['2']

    # create the 3D plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z, c='y', label='D')
    ax.scatter3D(x_tag, y_tag, z_tag, c='r', label='D_tag')
    ax.scatter3D(x_anom, y_anom, z_anom, c='k', label='s')
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.set_zlabel('feature 3')
    plt.legend()

    # enable interactive 3D navigation
    ax.view_init(elev=20, azim=30)
    ax.dist = 10

    plt.show()
    # plt.savefig(os.path.join(RESULTS_FOLDER_PATH, "convergence.jpg"))


def project_2d(d_inf, method='tsne', plot=False):
    # reading the CSV file
    d = d_inf[[feature for feature in d_inf.columns.values if feature != 'assoc']]
    d_tag_c = d.loc[d_inf['assoc'] == 0]
    d_tag = d.loc[d_inf['assoc'] == 1]
    anomaly = d.loc[d_inf['assoc'] == 2]

    # create the 2D scatter plot
    # fig, ax = plt.subplots()

    # transform
    if method == "pca":
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


def project_fdiff(file_name, f_diff, method='tsne', plot=True, save=False):
    # reading the CSV file
    d_inf = pd.read_csv(file_name)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
    fig.suptitle('Dataset Projection', fontsize=20)
    # axs[0].set_ylabel('AFEX Score')
    # axs[0].plot(synthetic_data_knn_exp.convert_process["time"],
    #             synthetic_data_knn_exp.convert_process["score"], '-o')
    # axs[0].set_title('Solvers Convergence Over Time')
    # axs[0].set_xlabel('Time [sec]')
    proj_full = project_2d(d_inf, method=method)
    axs[0].scatter(proj_full["x_dtc"], proj_full["y_dtc"], c='y', label='D')
    axs[0].scatter(proj_full["x_dt"], proj_full["y_dt"], c='r', label='D Tag')
    axs[0].scatter(proj_full["x_anom"], proj_full["y_anom"], c='k', label='s')
    axs[0].set_xlabel(f'{method} 1', fontsize=12)
    axs[0].set_ylabel(f'{method} 2', fontsize=12)
    axs[0].legend()
    axs[0].set_title(f"D - {method} projection", fontsize=16)

    proj_fsim = project_2d(d_inf[[f for f in d_inf.columns.values if f not in f_diff]], method=method)
    axs[1].scatter(proj_fsim["x_dtc"], proj_fsim["y_dtc"], c='y', label='D')
    axs[1].scatter(proj_fsim["x_dt"], proj_fsim["y_dt"], c='r', label='D Tag')
    axs[1].scatter(proj_fsim["x_anom"], proj_fsim["y_anom"], c='k', label='s')
    axs[1].set_xlabel(f'{method} 1', fontsize=12)
    axs[1].set_ylabel(f'{method} 2', fontsize=12)
    axs[1].legend()
    axs[1].set_title(f"D[f_sim] = D{[f for f in d_inf.columns.values if f not in f_diff + ['assoc']]} "
                     f"- {method} projection", fontsize=16)

    proj_fdiff = project_2d(d_inf[f_diff + ['assoc']], method=method)
    axs[2].scatter(proj_fdiff["x_dtc"], proj_fdiff["y_dtc"], c='y', label='D')
    axs[2].scatter(proj_fdiff["x_dt"], proj_fdiff["y_dt"], c='r', label='D Tag')
    axs[2].scatter(proj_fdiff["x_anom"], proj_fdiff["y_anom"], c='k', label='s')
    axs[2].set_xlabel(f'{method} 1', fontsize=12)
    axs[2].set_ylabel(f'{method} 2', fontsize=12)
    axs[2].legend()
    axs[2].set_title(f"D[f_diff] = D{f_diff} - {method} projection", fontsize=16)

    if plot:
        plt.show()
    if save:
        plt.savefig(os.path.join(RESULTS_FOLDER_PATH, "Projections", f"{os.path.basename(file_name)}_proj.png"))
    return


if __name__ == '__main__':
    # file_name = 'data/DBSCAN_rc50_pmNone/synt_iter4_inf.csv'
    # file_name = 'results/DBSCAN_rc50_pmNone/synt_iter0_inf_obo_inf.csv'
    # file_name = 'data/T_Corpus/arcene.csv'
    # file_name = 'data/T_Corpus/clinical_1.csv'
    file_name = 'results/partial_synthetic/synt_iter4_inf_bf_inf.csv'
    # file_name = 'results/partial_synthetic/synt_iter0_inf_mc_inf.csv'

    # reading the CSV file
    dataset = pd.read_csv(file_name)

    # scatter_3d(d_inf)
    f_diff = ['0', '3']

    project_fdiff(file_name, f_diff=f_diff, method='tsne', plot=True, save=False)
