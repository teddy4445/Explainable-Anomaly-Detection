import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# project imports
from consts import *

# file_name = 'T_Corpus_fixed/arcene.csv'

# file_name = 'T_Corpus/arcene.csv'
# file_name = 'T_Corpus/clinical_1.csv'
# file_name = 'T_Corpus/cosmetics.csv'
# file_name = 'T_Corpus/Crop_Agriculture_Data_2.csv'
# file_name = 'T_Corpus/Customer_Behaviour_Survey_responses.csv'
# file_name = 'T_Corpus/dataR2.csv'
# file_name = 'T_Corpus/dataset_10_lymph.csv'
# file_name = 'T_Corpus/dataset_13_breast-cancer.csv'
# file_name = 'T_Corpus/dataset_17_bridges_version1.csv'
# file_name = 'T_Corpus/dataset_27_colic.csv'
file_name = 'T_Corpus/dataset_48_tae.csv'

# reading the CSV file
dataset = pd.read_csv(file_name)
dataset = dataset.drop(columns=['Unnamed: 0', 'target'])
# dataset = dataset.drop(columns=['Unnamed: 0', 'target'])
print(dataset.head())
print(dataset.columns.values)
print()

dataset.to_csv(os.path.join('T_Corpus_fixed', f"{file_name.split('/')[1]}"), index=False)
