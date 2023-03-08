import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def project_fdiff(d_inf, f_diff, method='tsne', plot=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
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

    proj_fdiff = project_2d(d_inf[f_diff + ['assoc']], method=method)
    axs[1].scatter(proj_fdiff["x_dtc"], proj_fdiff["y_dtc"], c='y', label='D')
    axs[1].scatter(proj_fdiff["x_dt"], proj_fdiff["y_dt"], c='r', label='D Tag')
    axs[1].scatter(proj_fdiff["x_anom"], proj_fdiff["y_anom"], c='k', label='s')
    axs[1].set_xlabel(f'{method} 1', fontsize=12)
    axs[1].set_ylabel(f'{method} 2', fontsize=12)
    axs[1].legend()
    axs[1].set_title(f"D[f_diff] - {method} projection", fontsize=16)

    if plot:
        plt.show()
    return


if __name__ == '__main__':
    file_name = 'results/synt_example_iter0_inf.csv'

    # reading the CSV file
    d_inf = pd.read_csv(file_name)

    # scatter_3d(d_inf)
    project_fdiff(d_inf, f_diff=['0', '1'], method='tsne', plot=True)
