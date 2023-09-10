from lime.lime_tabular import LimeTabularExplainer

from baseline_explainers.baseline_explainer import BaselineExplainer


class LimeExplainer(BaselineExplainer):
    def __init__(self, data, model=None, mode=None):
        super().__init__(data=data, model=model, mode=mode)

    def get_explanation(self, anomaly, threshold=0.95):
        explainer = LimeTabularExplainer(self.data.values, feature_names=self.features,
                                         class_names=["non-anom", "anom"], discretize_continuous=True)
        explanation = explainer.explain_instance(data_row=anomaly.values, predict_fn=self.model.predict_proba)
        fig = explanation.as_pyplot_figure()
        self.save_barplot_explanation(path='results', name='lime', show=False)
        print(explanation.as_list())

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
    explainer = LimeExplainer(data=dataset_wo_anomaly, model=classifier)

    explanation = explainer.get_explanation(anomaly=anomaly_sample)
    print('\n'.join(map(str, explanation.as_list())))
    # explanation.as_pyplot_figure()
    # explanation.show_in_notebook()
