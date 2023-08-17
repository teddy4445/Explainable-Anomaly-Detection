from lime.lime_tabular import LimeTabularExplainer

from baseline_explainers.baseline_explainer import BaselineExplainer


class LimeExplainer(BaselineExplainer):
    """
    Lime-based approach
    """

    def __init__(self, data, classifier=None, mode='waterfall'):
        super().__init__(data=data, classifier=classifier)
        self.mode = mode

    def get_explanation(self, anomaly, threshold=0.95):
        # Use SHAP to explain the model's decisions
        pseudo_ad_model = self.trained_pseudo_ad_model(anomaly=anomaly)
        explainer = LimeTabularExplainer(self.data.values, feature_names=self.data.columns.values.tolist(),
                                         class_names=["non-anom", "anom"], discretize_continuous=True)
        exp = explainer.explain_instance(data_row=anomaly.values, predict_fn=pseudo_ad_model.predict_proba)

        return exp


if __name__ == '__main__':
    import pandas as pd

    filename = '../datasets/supervised/synt0.csv'
    df = pd.read_csv(filename)

    dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
    anomaly_sample = dataset.loc[df['assoc'] == 2].iloc[-1]
    dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)

    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier
    explainer = LimeExplainer(data=dataset_wo_anomaly, classifier=classifier)

    explanation = explainer.get_explanation(anomaly=anomaly_sample)
    print('\n'.join(map(str, explanation.as_list())))
    # explanation.as_pyplot_figure()
    # explanation.show_in_notebook()
