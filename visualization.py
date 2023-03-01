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


def scatter_2d(d_inf, method='tsne'):
    # reading the CSV file
    d = d_inf[[feature for feature in d_inf.columns.values if feature != 'assoc']]
    d_tag_c = d.loc[d_inf['assoc'] == 0]
    d_tag = d.loc[d_inf['assoc'] == 1]
    anomaly = d.loc[d_inf['assoc'] == 2]

    # create the 2D scatter plot
    fig, ax = plt.subplots()

    # transform
    if method == "pca":
        pca = PCA(n_components=2)
        pca.fit(d)
        reduced_d_tag_c = pca.transform(d_tag_c)
        reduced_d_tag = pca.transform(d_tag)
        reduced_anomaly = pca.transform(anomaly)

        ax.scatter(reduced_d_tag_c[:, 0], reduced_d_tag_c[:, 1], c='y', label='D')
        ax.scatter(reduced_d_tag[:, 0], reduced_d_tag[:, 1], c='r', label='D Tag')
        ax.scatter(reduced_anomaly[:, 0], reduced_anomaly[:, 1], c='k', label='s')

    elif method == "tsne":
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0)
        d_tsne = pd.DataFrame(tsne.fit_transform(d))
        reduced_d_tag_c = d_tsne.loc[d_inf['assoc'] == 0]
        reduced_d_tag = d_tsne.loc[d_inf['assoc'] == 1]
        reduced_anomaly = d_tsne.loc[d_inf['assoc'] == 2]

        ax.scatter(reduced_d_tag_c[0], reduced_d_tag_c[1], c='y', label='D')
        ax.scatter(reduced_d_tag[0], reduced_d_tag[1], c='r', label='D Tag')
        ax.scatter(reduced_anomaly[0], reduced_anomaly[1], c='k', label='s')

    else:
        return



    # ax.set_xlabel('PC 1')
    # ax.set_ylabel('PC 2')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    file_name = 'results/synt_example_iter0_inf.csv'

    # reading the CSV file
    d_inf = pd.read_csv(file_name)

    # scatter_3d(d_inf)
    # scatter_2d(d_inf, method='pca')
    scatter_2d(d_inf, method='tsne')
