# library imports
import json
import random
import jsbeautifier
import os
import itertools
import pandas as pd
from typing import Callable

# project imports
from explanation_analysis.similarity_metrices.sim_euclidean_inverse import InverseEuclideanSim
from explanation_analysis.score_function.linear_score import LinearScore
from visualization import project_fdiff


class RemoveRowsDegradator:
    def degragate(self, chain):
        rows = [row for row in chain[-1]["rows"] if row in chain[0]["rows"]]
        new_rows = sorted(random.sample(rows, len(rows) // 2))

        new_level = {
            "rows": new_rows,
            "columns": list(chain[-1]["columns"])
        }
        return new_level


class AddRowsDegradator:
    def __init__(self, all_rows):
        self.all_rows = all_rows

    def degragate(self, chain):
        valid_rows = set(self.all_rows) - set(chain[0]["rows"])
        new_rows = sorted(chain[-1]["rows"] + random.sample(list(valid_rows), random.randint(1, len(valid_rows) // 2)))

        new_level = {
            "rows": new_rows,
            "columns": list(chain[-1]["columns"])
        }
        return new_level


class SwitchRowsDegradator:
    def __init__(self, all_rows):
        self.all_rows = all_rows

    def degragate(self, chain):
        rows = [row for row in chain[-1]["rows"] if row in chain[0]["rows"]]
        valid_rows = set(self.all_rows) - set(chain[0]["rows"])
        new_rows = sorted(random.sample(rows, random.randint(1, len(rows)))
                          + random.sample(list(valid_rows), random.randint(1, len(valid_rows))))

        new_level = {
            "rows": new_rows,
            "columns": list(chain[-1]["columns"])
        }
        return new_level


class RemoveFeaturesDegradator:
    def degragate(self, chain):
        columns = [column for column in chain[-1]["columns"] if column in chain[0]["columns"]]
        new_columns = sorted(random.sample(columns, random.randint(1, len(columns)-1)))

        new_level = {
            "rows": list(chain[-1]["rows"]),
            "columns": list(new_columns)
        }
        return new_level


class AddFeaturesDegradator:
    def __init__(self, all_columns):
        self.all_columns = all_columns

    def degragate(self, chain):
        valid_columns = set(self.all_columns) - set(chain[0]["columns"])
        new_columns = sorted(chain[-1]["columns"] + random.sample(list(valid_columns), random.randint(1, len(valid_columns))))

        new_level = {
            "rows": list(chain[-1]["rows"]),
            "columns": list(new_columns)
        }
        return new_level


class SwitchFeaturesDegradator:
    def __init__(self, all_columns):
        self.all_columns = all_columns

    def degragate(self, chain):
        columns = [column for column in chain[-1]["columns"] if column in chain[0]["columns"]]
        valid_columns = set(self.all_columns) - set(chain[0]["columns"])
        new_columns = sorted(random.sample(columns, random.randint(1, len(columns)))
                             + random.sample(list(valid_columns), random.randint(1, len(valid_columns))))

        new_level = {
            "rows": list(chain[-1]["rows"]),
            "columns": list(new_columns)
        }
        return new_level


class ExplanationChainCreator:
    def __init__(self, filename, base_columns):
        self.d_inf = pd.read_csv(filename)
        self.base_columns = base_columns
        self.columns = list(self.d_inf.columns.values)
        self.columns.remove('assoc')
        self.rows = list(self.d_inf.index)[:-1]

        self.remove_rows_degragator = RemoveRowsDegradator()
        self.add_rows_degragator = AddRowsDegradator(all_rows=self.rows)
        self.switch_rows_degragator = SwitchRowsDegradator(all_rows=self.rows)
        self.remove_cols_degragator = RemoveFeaturesDegradator()
        self.add_cols_degragator = AddFeaturesDegradator(all_columns=self.columns)
        self.switch_cols_degragator = SwitchFeaturesDegradator(all_columns=self.columns)

        self.all_degragators = [
            self.remove_rows_degragator, self.add_rows_degragator, self.switch_rows_degragator,
            self.remove_cols_degragator, self.add_cols_degragator, self.switch_cols_degragator
        ]

    def exp2dict(self):
        exp_dict = {
            'rows': [index for index, row in self.d_inf.iterrows() if row['assoc'] == 1],
            'columns': self.base_columns
        }
        return exp_dict

    def create_chain_by_degragator(self, degragator):
        chain = list()
        chain.append(self.exp2dict())
        chain.append(self.remove_rows_degragator.degragate(chain))
        chain.append(degragator.degragate(chain))
        return chain

    def create_chains(self, n_chains=150):
        chain_list = list()
        for _ in range(n_chains // 6):
            for degragator in self.all_degragators:
                new_chain = None
                while not new_chain or new_chain in chain_list:
                    new_chain = self.create_chain_by_degragator(degragator)
                chain_list.append(new_chain)
        return chain_list


def print_dict(dict2print):
    options = jsbeautifier.default_options()
    options.indent_size = 2
    print(jsbeautifier.beautify(json.dumps(dict2print), options))


def vizualize_chain(d_inf, chain):
    for link in chain:
        # link = chain[0]
        d_inf['assoc'] = 0
        d_inf.loc[link['rows'], 'assoc'] = 1
        d_inf.loc[len(d_inf) - 1, 'assoc'] = 2
        project_fdiff(d_inf=d_inf, f_diff=link['columns'], method='tsne', plot=True, annotate=False, save=False)


def main(filename):
    # dict_main_key = "assoc"
    # d_inf = pd.read_csv(filename)
    #
    # dataset = d_inf[[feature for feature in d_inf.columns.values if feature != dict_main_key]]
    # d_tag = dataset.loc[(d_inf[dict_main_key] == 1) | (d_inf[dict_main_key] == 2)]
    # anomaly_sample = dataset.loc[d_inf[dict_main_key] == 2].iloc[-1]
    # dataset_wo_anomaly = dataset.loc[d_inf[dict_main_key] != 2].reset_index(drop=True)
    chain_creator = ExplanationChainCreator(filename=filename, base_columns=['0', '1'])
    # print_dict(chain_creator.create_chains())
    chain_list = chain_creator.create_chains()
    # chains_df = pd.DataFrame(chain_list)
    d_inf = pd.read_csv(filename)
    for chain in chain_list:
        vizualize_chain(d_inf=d_inf, chain=chain)
    # print(pd.DataFrame(chain_creator.create_chains()))
    # chains_df.to_csv("chains.csv", index=False)


if __name__ == '__main__':
    filename = 'data/partial_synthetic/synt_iter0_inf_aug.csv'
    # os.makedirs('data/expalnations_corpus/', exist_ok=True)
    main(filename=filename)
