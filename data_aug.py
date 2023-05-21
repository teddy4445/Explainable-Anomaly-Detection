import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# project imports
from consts import *
from visualization import *


def move_target_point(anchor, target, features, factor):
    for feature in features:
        target[feature] = anchor[feature] + factor * (target[feature] - anchor[feature])
    return target


def move_cluster(dataset, target_assoc, features, factor):
    d = dataset[[feature for feature in dataset.columns.values if feature != 'assoc']]

    if target_assoc == 1:
        anchor = d.loc[dataset['assoc'] == 1].mean(axis=0)
    elif target_assoc == 2:
        anchor = d.loc[dataset['assoc'] == 2]

    for index, row in d.loc[dataset['assoc'] == 1].iterrows():
        # move_target_point(anchor=anchor, target=row, features=features, factor=factor)
        dataset.loc[index, features] = move_target_point(anchor=anchor, target=row, features=features, factor=factor)

    return dataset


if __name__ == '__main__':
    dataset_name = 'synt_iter6_inf'
    file_name = f'data/DBSCAN_rc50_pmNone_aug/{dataset_name}.csv'

    # reading the CSV file
    dataset = pd.read_csv(file_name)

    # scatter_3d(d_inf)
    f_diff = ['0', '1']
    f_sim = ['2', '3', '4']

    # first view
    project_fdiff(d_inf=dataset, f_diff=f_diff, method='tsne', plot=True, annotate=False, save=False)

    # augment dataset
    dataset_aug = move_cluster(dataset=dataset, target_assoc=2, features=f_diff, factor=1)
    dataset_aug = move_cluster(dataset=dataset_aug, target_assoc=1, features=f_diff, factor=0.3)
    dataset_aug = move_cluster(dataset=dataset_aug, target_assoc=2, features=f_sim, factor=0.3)
    dataset_aug = move_cluster(dataset=dataset_aug, target_assoc=1, features=f_sim, factor=1)
    project_fdiff(d_inf=dataset_aug, f_diff=f_diff, method='tsne', plot=True, annotate=False, save=False)

    # save new dataset
    saving_directory = 'partial_synthetic'
    # dataset_aug.to_csv(f'data/{saving_directory}/{dataset_name}_aug.csv', index=False)
