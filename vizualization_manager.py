import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class VisualizationManager:
    @staticmethod
    def scale_df(df):
        return (df - df.min()) / (df.max() - df.min())

    @staticmethod
    def calculate_distance(df):
        dist_matrix = np.zeros((df.shape[0], df.shape[0]))

        for i in range(df.shape[0]):
            for j in range(df.shape[0]):
                if i != j:
                    dist_matrix[i, j] = np.linalg.norm(df.iloc[i] - df.iloc[j])
                else:
                    dist_matrix[i, j] = np.nan

        return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

    def plot_distance_matrices(self, d_inf, f_diff, f_sim, dist_axs):
        d_tag_anomaly = d_inf.loc[d_inf['assoc'] != 0]
        d_wo_assoc = d_tag_anomaly[[feature for feature in d_inf.columns.values if feature != 'assoc']]
        dist_df = self.calculate_distance(d_wo_assoc)
        dist_df_diff = self.calculate_distance(d_wo_assoc[f_diff])
        dist_df_sim = self.calculate_distance(d_wo_assoc[f_sim])

        # Calculate the color scale normalization based on the first matrix
        norm = Normalize(vmin=dist_df.min().min(), vmax=dist_df.max().max())

        sns.heatmap(dist_df, ax=dist_axs[0], cmap='RdYlGn', norm=norm, annot=True)
        dist_axs[0].set_title('Distance Matrix')

        sns.heatmap(dist_df_sim, ax=dist_axs[1], cmap='RdYlGn', norm=norm, annot=True)  # Green to Red, RdYlGn_r
        dist_axs[1].set_title('Distance Matrix - Over F_sim')

        sns.heatmap(dist_df_diff, ax=dist_axs[2], cmap='RdYlGn', norm=norm, annot=True)  # Red to Green, RdYlGn
        dist_axs[2].set_title('Distance Matrix - Over F_diff')

    def project_2d(self, d_partial, method='tsne'):
        d_wo_assoc = d_partial[[feature for feature in d_partial.columns.values if feature != 'assoc']]
        index_dtc = list(d_partial.index[d_partial['assoc'] == 0])
        index_dt = list(d_partial.index[d_partial['assoc'] == 1])
        index_anom = list(d_partial.index[d_partial['assoc'] == 2])

        proj_data = {
            'index_dtc': index_dtc,
            'index_dt': index_dt,
            'index_anom': index_anom
        }
        if len(d_wo_assoc.columns) == 1:
            x_dtc = d_wo_assoc.loc[index_dtc].values.reshape(1, -1)[0]
            y_dtc = 0.5 * np.ones_like(x_dtc)
            x_dt = d_wo_assoc.loc[index_dt].values.reshape(1, -1)[0]
            y_dt = 0.5 * np.ones_like(x_dt)
            x_anom = d_wo_assoc.loc[index_anom].values.reshape(1, -1)[0]
            y_anom = 0.5 * np.ones_like(x_anom)

            proj_data.update({"x_dtc": x_dtc, "y_dtc": y_dtc,
                              "x_dt": x_dt, "y_dt": y_dt,
                              "x_anom": x_anom, "y_anom": y_anom})

        # transform
        elif method == "pca":
            pca = PCA(n_components=2)
            pca.fit(d_wo_assoc)
            reduced_d_tag_c = pca.transform(d_wo_assoc.loc[d_partial['assoc'] == 0])
            reduced_d_tag = pca.transform(d_wo_assoc.loc[d_partial['assoc'] == 1])
            reduced_anomaly = pca.transform(d_wo_assoc.loc[d_partial['assoc'] == 2])

            proj_data.update({"x_dtc": reduced_d_tag_c[:, 0], "y_dtc": reduced_d_tag_c[:, 1],
                              "x_dt": reduced_d_tag[:, 0], "y_dt": reduced_d_tag[:, 1],
                              "x_anom": reduced_anomaly[:, 0], "y_anom": reduced_anomaly[:, 1]})

        elif method == "tsne":
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0)
            d_tsne = pd.DataFrame(tsne.fit_transform(d_wo_assoc))
            d_tsne = self.scale_df(d_tsne)
            reduced_d_tag_c = d_tsne.loc[d_partial['assoc'] == 0]
            reduced_d_tag = d_tsne.loc[d_partial['assoc'] == 1]
            reduced_anomaly = d_tsne.loc[d_partial['assoc'] == 2]

            proj_data.update({"x_dtc": reduced_d_tag_c[0], "y_dtc": reduced_d_tag_c[1],
                              "x_dt": reduced_d_tag[0], "y_dt": reduced_d_tag[1],
                              "x_anom": reduced_anomaly[0], "y_anom": reduced_anomaly[1]})

        return proj_data

    def scatter(self, graph, X, Y, color, label, annotations=None):
        graph.scatter(X, Y, c=color, label=label)
        if annotations:
            for x, y, index in zip(X, Y, annotations):
                graph.annotate(index, (x+0.01, y))  # , xycoords="axes fraction"

    def prepare_one_scatter_graph(self, axs, graph_index, d_partial, method, annotate, title):
        proj_data = self.project_2d(d_partial=d_partial, method=method)
        graph = axs[graph_index]
        self.scatter(graph=graph, X=proj_data["x_dtc"], Y=proj_data["y_dtc"], color='y', label='D',
                     annotations=proj_data["index_dtc"] if annotate else None)
        self.scatter(graph=graph, X=proj_data["x_dt"], Y=proj_data["y_dt"], color='b', label='D Tag',
                     annotations=proj_data["index_dt"] if annotate else None)
        self.scatter(graph=graph, X=proj_data["x_anom"], Y=proj_data["y_anom"], color='k', label='s',
                     annotations=proj_data["index_anom"] if annotate else None)
        graph.set_xlabel(f'{method} 1', fontsize=12)
        graph.set_ylabel(f'{method} 2', fontsize=12)

        graph.legend()
        graph.set_title(title, fontsize=16)

    def plot_scatters(self, d_inf, f_diff, f_sim, scatter_axs, method='tsne', annotate=False):
        self.prepare_one_scatter_graph(axs=scatter_axs, graph_index=0, d_partial=d_inf, method=method,
                                       annotate=annotate, title="D (full)")
        self.prepare_one_scatter_graph(axs=scatter_axs, graph_index=1, d_partial=d_inf[f_sim + ['assoc']],
                                       method=method,
                                       annotate=annotate, title="D[f_sim]")
        self.prepare_one_scatter_graph(axs=scatter_axs, graph_index=2, d_partial=d_inf[f_diff + ['assoc']],
                                       method=method,
                                       annotate=annotate, title="D[f_diff]")

    def visualize(self, d_inf, f_diff, f_sim, method='tsne', plot=True, annotate=False, save_filename=None):
        fig, axs = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle(f'Dataset Projection - {method} method', fontsize=20)
        self.plot_scatters(d_inf=d_inf, f_diff=f_diff, f_sim=f_sim,
                           scatter_axs=axs[0], method=method, annotate=annotate)
        self.plot_distance_matrices(d_inf=d_inf, f_diff=f_diff, f_sim=f_sim, dist_axs=axs[1])

        if save_filename:
            plt.savefig(f"{save_filename}_visualization.png")
        if plot:
            plt.show()


if __name__ == '__main__':
    vz_manager = VisualizationManager()
    file_name = './results/synt0.csv_BruteForce_linear_inverse_euclidean_29759353-2a3b-4740-b745-c05b6f50fbf0_d_inf.csv'
    dataset = pd.read_csv(file_name)
    f_diff = ['0', '4']  #
    f_sim = ['1', '2', '3']

    vz_manager.visualize(d_inf=dataset, f_diff=f_diff, f_sim=f_sim,
                         method='tsne', plot=False, annotate=True, save_filename="./results/check3")
