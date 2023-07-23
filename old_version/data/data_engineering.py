import pandas as pd

# project imports

# file_name = 'T_Corpus_fixed/arcene.csv'

# file_name = 'T_Corpus/arcene.csv'  # (200,99)
# file_name = 'T_Corpus/clinical_1.csv'  # (2000,14)
# file_name = 'T_Corpus/cosmetics.csv'  # (1472,7)
# file_name = 'T_Corpus/Crop_Agriculture_Data_2.csv'  # (2000,8)
# file_name = 'T_Corpus/Customer_Behaviour_Survey_responses.csv'  # (254,12)
# file_name = 'T_Corpus/dataR2.csv'  # (116,9)
# file_name = 'T_Corpus/dataset_10_lymph.csv'  # (148,17)
# file_name = 'T_Corpus/dataset_13_breast-cancer.csv'  # (286,9)
# file_name = 'T_Corpus/dataset_17_bridges_version1.csv'  # (107,9)
# file_name = 'T_Corpus/dataset_27_colic.csv'  # (386,15)
file_name = 'T_Corpus/dataset_48_tae.csv'  # (151,5)

# reading the CSV file
dataset = pd.read_csv(file_name)
dataset = dataset.drop(columns=['Unnamed: 0', 'target'])
# dataset = dataset.drop(columns=['Unnamed: 0', 'target'])
print(dataset.head())
print(dataset.columns.values)
print(dataset.shape)
print()

# dataset.to_csv(os.path.join('T_Corpus_fixed', f"{file_name.split('/')[1]}"), index=False)
