import shap
from baseline_explainers.baseline_explainer import BaselineExplainer


class ShapExplainer(BaselineExplainer):
    def __init__(self, data, model=None, mode=None):
        super().__init__(data=data, model=model, mode=mode)

    def get_explanation(self, anomaly):
        # pseudo_ad_model = self.trained_pseudo_ad_model(anomaly=anomaly)
        explainer = shap.Explainer(self.model)
        explanation = explainer(anomaly)

        if self.mode == 'clf':
            explanation.values = explanation.values[:, 1]
            explanation.base_values = explanation.base_values[0, 1]
        elif self.mode == 'ad':
            explanation.base_values = explanation.base_values[0, 0]
        shap.plots.bar(explanation, show=False)
        self.save_barplot_explanation(path='results', name='shap', show=False)

        return explainer(anomaly)


if __name__ == '__main__':
    import pandas as pd

    filename = '../datasets/supervised/synt0.csv'
    df = pd.read_csv(filename)

    dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
    anomaly_sample = dataset.loc[df['assoc'] == 2].iloc[-1]
    dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)

    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier
    explainer = ShapExplainer(data=dataset_wo_anomaly, model=classifier)

    explanation = explainer.get_explanation(anomaly=anomaly_sample)
    print(explanation)
