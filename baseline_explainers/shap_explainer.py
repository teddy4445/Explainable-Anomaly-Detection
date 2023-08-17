import shap
from baseline_explainers.baseline_explainer import BaselineExplainer


class ShapExplainer(BaselineExplainer):
    """
    A Shap-based approach
    """

    def __init__(self, data, classifier=None, mode='waterfall'):
        super().__init__(data=data, classifier=classifier)
        self.mode = mode

    def get_explanation(self, anomaly):
        # Use SHAP to explain the model's decisions
        pseudo_ad_model = self.trained_pseudo_ad_model(anomaly=anomaly)
        explainer = shap.Explainer(pseudo_ad_model)
        shap_values = explainer(anomaly)

        return shap_values


if __name__ == '__main__':
    import pandas as pd
    filename = '../datasets/supervised/synt0.csv'
    df = pd.read_csv(filename)

    dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
    anomaly_sample = dataset.loc[df['assoc'] == 2].iloc[-1]
    dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier
    explainer = ShapExplainer(data=dataset_wo_anomaly, classifier=classifier)

    explanation = explainer.get_explanation(anomaly=anomaly_sample)
    print(explanation)
