import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import ast
from explanation_analysis.score_function.linear_score import LinearScore
from explanation_analysis.similarity_metrices.sim_euclidean_inverse import InverseEuclideanSim
from visualization import project_fdiff
import matplotlib.pyplot as plt
import seaborn as sns


def compute_scores(filename, output_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    s = df.iloc[-1]
    df = df.drop(df.index[-1])  # drop the last row as it is separated as s
    if 'assoc' in df.columns:
        df = df.drop(columns=['assoc'])

    rows = df.index.tolist()
    cols = df.columns.tolist()

    # Prepare Scorer
    scorer = LinearScore(sim_module=InverseEuclideanSim(),
                         w_self_sim=1,
                         w_local_sim=1, w_cluster_sim=0,
                         w_local_diff=1, w_cluster_diff=0,
                         w_cov=0, w_conc=0.05)  # 0.05

    # Prepare the output DataFrame
    results = []  # Will hold each row to be appended to the dataframe

    # Prepare tqdm progress bar
    total_iterations = sum(1 for _ in combinations(rows, 3)) * 31  # len(cols)
    pbar = tqdm(total=total_iterations, desc="Processing combinations")

    # Iterate over combinations of rows and columns
    # break_index = 0
    for c in range(1, len(cols) + 1):  # len(cols) + 1
        for col_comb in combinations(cols, c):
            for row_comb in combinations(rows, 3):
                # if break_index == 10:
                #     break
                # break_index += 1

                f_diff = sorted(list(col_comb))
                f_sim = sorted(list(set(cols) - set(f_diff)))

                overall_score, partial_scores = scorer.compute(
                    d=df.loc[list(row_comb)], s=s, f_sim=f_sim, f_diff=f_diff, overall_size=len(df)
                )
                result = {
                    'f_diff': f_diff,
                    'f_sim': f_sim,
                    'rows': list(row_comb),
                    'overall_score': overall_score,
                }
                result.update(partial_scores)
                results.append(result)

                # Update the progress bar
                pbar.update()

    # Close the progress bar
    pbar.close()

    # Convert list of dicts to DataFrame
    output_df = pd.DataFrame(results)
    output_df = output_df.sort_values(by=['overall_score'], ascending=False)

    # Write the output DataFrame to a CSV file
    output_df.to_csv(output_filename, index=False)


def vizualize_explanation(i_good, i_med, i_bad, scores_inf, print_good=True, print_med=True, print_bad=True):
    if print_good:
        print('\nIndex Good: ', i_good)
        print('Good Explanation Score: ', scores_inf.loc[i_good]['overall_score'])
        d_inf['assoc'] = 0
        d_inf.loc[ast.literal_eval(scores_inf.loc[i_good]['rows']), 'assoc'] = 1
        d_inf.loc[len(d_inf) - 1, 'assoc'] = 2
        project_fdiff(d_inf=d_inf, f_diff=ast.literal_eval(scores_inf.loc[i_good]['f_diff']),
                      method='tsne', plot=True, annotate=False, save=False)

    if print_med:
        print('\nIndex Med: ', i_med)
        print('Med Explanation Score: ', scores_inf.loc[i_med]['overall_score'])
        d_inf['assoc'] = 0
        d_inf.loc[ast.literal_eval(scores_inf.loc[i_med]['rows']), 'assoc'] = 1
        d_inf.loc[len(d_inf) - 1, 'assoc'] = 2
        project_fdiff(d_inf=d_inf, f_diff=ast.literal_eval(scores_inf.loc[i_med]['f_diff']),
                      method='tsne', plot=True, annotate=False, save=False)

    if print_bad:
        print('\nIndex Bad: ', i_bad)
        print('Bad Explanation Score: ', scores_inf.loc[i_bad]['overall_score'])
        d_inf['assoc'] = 0
        d_inf.loc[ast.literal_eval(scores_inf.loc[i_bad]['rows']), 'assoc'] = 1
        d_inf.loc[len(d_inf) - 1, 'assoc'] = 2
        project_fdiff(d_inf=d_inf, f_diff=ast.literal_eval(scores_inf.loc[i_bad]['f_diff']),
                      method='tsne', plot=True, annotate=False, save=False)


if __name__ == '__main__':
    filename = 'data/partial_synthetic/synt_iter0_inf_aug.csv'
    # compute_scores(filename=filename, output_filename='grade_bf_3_conc05_new.csv')

    scores_filename = 'grade_bf_3_conc05.csv'
    scores_inf = pd.read_csv(scores_filename)
    d_inf = pd.read_csv(filename)

    # i = 0
    # vizualize_explanation(i_good=len(scores_inf)//8,
    #                       i_med=3 * len(scores_inf) // 8,
    #                       i_bad=7 * len(scores_inf) // 8,
    #                       scores_inf=scores_inf, print_good=True, print_med=True, print_bad=True)

    # i = 9
    # vizualize_explanation(i_good=i, i_med=i+1, i_bad=i+2, scores_inf=scores_inf,
    #                       print_good=True, print_med=True, print_bad=True)

    print()

    # triplets = [
    #     {'f_diff': "['0']", 'rows': '[41, 43, 47]'},
    #     {'f_diff': "['0', '2']", 'rows': '[41, 43, 47]'},
    #     {'f_diff': "['2']", 'rows': '[41, 43, 47]'},
    # ]
    # triplets = [
    #     {'f_diff': "['0', '1']", 'rows': '[41, 43, 47]'},
    #     {'f_diff': "['0', '1']", 'rows': '[38, 43, 47]'},
    #     {'f_diff': "['0', '1']", 'rows': '[30, 36, 38]'},
    # ]
    # triplets = [
    #     {'f_diff': "['0', '2']", 'rows': '[40, 45, 47]'},
    #     {'f_diff': "['0', '2', '3']", 'rows': '[40, 45, 47]'},
    #     {'f_diff': "['0', '2', '3', '4']", 'rows': '[40, 45, 47]'},
    # ]
    # triplets = [
    #     {'f_diff': "['0', '3']", 'rows': '[40, 45, 47]'},
    #     {'f_diff': "['0', '3']", 'rows': '[24, 40, 45]'},
    #     {'f_diff': "['0', '3']", 'rows': '[12, 16, 36]'},
    # ]
    # i_good = scores_inf.loc[(scores_inf['rows'] == triplets[0]['rows'])
    #                         & (scores_inf['f_diff'] == triplets[0]['f_diff'])].index.to_list()[0]
    # i_med = scores_inf.loc[(scores_inf['rows'] == triplets[1]['rows'])
    #                         & (scores_inf['f_diff'] == triplets[1]['f_diff'])].index.to_list()[0]
    # i_bad = scores_inf.loc[(scores_inf['rows'] == triplets[2]['rows'])
    #                        & (scores_inf['f_diff'] == triplets[2]['f_diff'])].index.to_list()[0]
    # vizualize_explanation(i_good=i_good, i_med=i_med, i_bad=i_bad, scores_inf=scores_inf,
    #                       print_good=True, print_med=True, print_bad=True)
    print()

    # Create histogram on MEDV column (target column)
    column_name = 'overall_score'  # overall_score, self_sim, local_sim, local_diff
    # Set the width of each bin
    binwidth = 0.02

    # Calculate the bins
    bins = np.arange(scores_inf[column_name].min(), scores_inf[column_name].max() + binwidth, binwidth)

    # Plot the histogram
    sns.histplot(data=scores_inf, x=column_name, bins=bins, kde=False)

    plt.title(f"{column_name}'s Histogram")
    plt.show()

