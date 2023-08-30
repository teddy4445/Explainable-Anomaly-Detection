import numpy as np
from anchor import anchor_tabular
from baseline_explainers.baseline_explainer import BaselineExplainer


class AnchorExplainer(BaselineExplainer):
    """
    Anchors explanations
    """

    def __init__(self, data, classifier=None, mode='waterfall', threshold=0.95):
        super().__init__(data=data, classifier=classifier)
        self.mode = mode
        self.threshold = threshold

    def get_explanation(self, anomaly):
        pseudo_ad_model = self.trained_pseudo_ad_model(anomaly=anomaly)
        explainer = anchor_tabular.AnchorTabularExplainer(class_names=np.array(["non-anom", "anom"]),
                                                          feature_names=self.data.columns.values.tolist(),
                                                          train_data=self.data.values)
        exp = explainer.explain_instance(data_row=anomaly.values,  # anomaly.values, np.array(anomaly.values),
                                         classifier_fn=pseudo_ad_model.predict,
                                         threshold=self.threshold)

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
    explainer = AnchorExplainer(data=dataset_wo_anomaly, classifier=classifier, threshold=0.95)

    explanation = explainer.get_explanation(anomaly=anomaly_sample)

    # Getting an anchor
    print('\nGetting an anchor')
    print('Anchor: %s' % (' AND '.join(explanation.names())))
    print('Precision: %.2f' % explanation.precision())
    print('Coverage: %.2f' % explanation.coverage())

    # Looking at a partial anchor
    print('\nLooking at a partial anchor')
    print('Partial anchor: %s' % (' AND '.join(explanation.names(1))))
    print('Partial precision: %.2f' % explanation.precision(1))
    print('Partial coverage: %.2f' % explanation.coverage(1))

    # explanation.show_in_notebook()
    print()
