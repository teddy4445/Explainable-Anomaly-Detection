import numpy as np
from acv_explainers import ACXplainer
from acv_explainers.utils import get_active_null_coalition_list
from sklearn.metrics import roc_auc_score

from baseline_explainers.baseline_explainer import BaselineExplainer


class ACVExplainer(BaselineExplainer):
    def __init__(self, data, model=None, mode=None):
        super().__init__(data=data, model=model, mode=mode)

    def get_explanation(self, anomaly):
        dataset_size = len(self.data)
        X_train = np.vstack([self.data, anomaly])
        y_train = np.array(dataset_size * [0] + [1])  # TODO: modify to use inherit model

        # It has the same params as a Random Forest, and it should be tuned to maximize the performance.
        acv_explainer = ACXplainer(classifier=True, n_estimators=10, max_depth=5)
        acv_explainer.fit(X=X_train, y=y_train)
        explanation = ''
        # roc = roc_auc_score(acv_explainer.predict(X_train), y_train)
        # explanation += f'The ROC AUC score of the explainer perdicator is {roc}\n'

        # Same Decision Probability (SDP)
        sdp_importance, sdp_index, size, sdp = acv_explainer.importance_sdp_rf(X_train, y_train.astype(np.double),
                                                                               X_train, y_train.astype(np.double))
        explanation += f'The Minimal Sufficient Explanation of the anomaly is = ' \
                       f'{sdp_index[dataset_size, :size[dataset_size]]} and it has a SDP = {sdp[dataset_size]}\n'

        # Compute the Sufficient Rules
        S_star, N_star = get_active_null_coalition_list(sdp_index, size)
        sdp, rules, sdp_all, rules_data, w = acv_explainer.compute_sdp_maxrules(X_train, y_train.astype(np.double),
                                                                                X_train, y_train.astype(np.double),
                                                                                S_star, verbose=True)
        rule = rules[dataset_size]  # dataset_size + 1 but starts by 0
        S = S_star[dataset_size]
        rule_string = ['{} <= {} <= {}'.format(rule[i, 0] if rule[i, 0] > -1e+10 else -np.inf, self.features[i],
                                               rule[i, 1] if rule[i, 1] < 1e+10 else +np.inf) for i in S]
        rule_string = ' and '.join(rule_string)
        explanation += f'The Minimal Sufficient Rule of the anomaly is: \n {rule_string}\n'

        return explanation


if __name__ == '__main__':
    import pandas as pd

    filename = '../datasets/supervised/synt0.csv'
    df = pd.read_csv(filename)

    dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
    anomaly_sample = dataset.loc[df['assoc'] == 2].iloc[-1]
    dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)

    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier
    explainer = ACVExplainer(data=dataset_wo_anomaly, model=classifier)  # classifier is unnecessary

    explanation = explainer.get_explanation(anomaly=anomaly_sample)
    print(explanation)