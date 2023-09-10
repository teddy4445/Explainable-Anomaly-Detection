from ad_models import AD_MODELS
from class_algos import CLASS_ALGOS

class AnomalyManager:
    def get_pseudo_ad_model(self, dataset, anomaly, model_class):
        model = model_class()

        return model

    def get_anomaly_and_model(self, supervised, df, model_type):
        if supervised:
            dataset = df[[feature for feature in df.columns.values if feature != 'assoc']]
            anomaly = dataset.loc[df['assoc'] == 2].iloc[-1]
            dataset_wo_anomaly = dataset.loc[df['assoc'] != 2].reset_index(drop=True)

            model = self.get_pseudo_ad_model(dataset, anomaly, CLASS_ALGOS[model_type])
        else:
            model = AD_MODELS[model_type]()
            model.fit(df)
            anomaly = model.predict_scores(df).argmax()
        return anomaly, dataset_wo_anomaly, model

