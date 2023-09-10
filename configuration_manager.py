import json
import yaml


class ConfigurationError(Exception):
    pass


class ConfigurationManager:
    def __init__(self, config_filename=None, config_string=None, configuration=None, file_format="yaml"):
        self.configuration = configuration or self._load_configuration(config_filename, config_string, file_format)

    @staticmethod
    def _load_configuration(config_filename, config_string, file_format):
        try:
            if not config_string:
                with open(config_filename, "r") as conf_f:
                    config_string = conf_f.read()
        except IOError:
            raise ConfigurationError(f"Could not read config file {config_filename}")
        if file_format == "yaml":
            return yaml.load(config_string, Loader=yaml.Loader)
        elif file_format == "json":
            json.loads(config_string)
        else:
            raise ConfigurationError("Unknown configuration format")

    @staticmethod
    def _get_param_variations(conf_obj):
        all_variations = []
        for variation in conf_obj:
            for params in conf_obj[variation]:
                all_variations.append({"type": variation, "params": params})
        return all_variations

    def _get_experiments_conf(self, data_filename, supervised, model):
        all_solvers = self._get_param_variations(conf_obj=self.configuration["solvers"])
        all_scorers = self._get_param_variations(conf_obj=self.configuration["scorers"])

        experiments = []

        for solver in all_solvers:
            for scorer in all_scorers:
                for similarity_metric in self.configuration["similarity_metrics"]:
                    experiment_conf = {
                        "dataset_filename": data_filename,
                        "supervised": supervised,
                        "model": model,
                        "solver": solver,
                        "scorer": scorer,
                        "similarity_metric": similarity_metric
                    }
                    experiments.append(experiment_conf)

        return experiments

    def get_experiment_batches(self):
        all_classification_models = self._get_param_variations(conf_obj=self.configuration["clf_models"])
        all_ad_models = self._get_param_variations(conf_obj=self.configuration["ad_models"])

        batches = []
        supervised_batches = [(supervised_dataset, classification_model, True)
                              for supervised_dataset in self.configuration["supervised_datasets"] or []
                              for classification_model in all_classification_models]
        unsupervised_batches = [(unsupervised_dataset, ad_model, False)
                                for unsupervised_dataset in self.configuration["unsupervised_datasets"] or []
                                for ad_model in all_ad_models]

        for dataset, model, supervised in supervised_batches + unsupervised_batches:
            experiment_list = self._get_experiments_conf(data_filename=dataset,
                                                         supervised=supervised,
                                                         model=model)
            batches.append({"supervised": supervised,
                            "dataset": dataset,
                            "model": model,
                            "experiment_list": experiment_list,
                            "alternative_explainers": self.configuration["alternative_explainers"] or []})

        return batches
