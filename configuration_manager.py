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

    def _get_solver_combs(self):
        all_solvers = []
        for solver in self.configuration["solvers"]:
            for params in self.configuration["solvers"][solver]:
                all_solvers.append({"type": solver, "params": params})
        return all_solvers

    def _get_scorer_combs(self):
        all_scorers = []
        for scorer in self.configuration["scorers"]:
            for params in self.configuration["scorers"][scorer]:
                all_scorers.append({"type": scorer, "params": params})
        return all_scorers

    def get_experiments_conf(self):
        all_solvers = self._get_solver_combs()
        all_scorers = self._get_scorer_combs()

        experiments = []

        for data in self.configuration["datasets"]:
            for solver in all_solvers:
                for scorer in all_scorers:
                    for similarity_metric in self.configuration["similarity_metrics"]:
                        experiment_conf = {
                            "dataset_filename": data["filename"],
                            "supervised": data["supervised"],
                            "solver": solver,
                            "scorer": scorer,
                            "similarity_metric": similarity_metric
                        }
                        experiments.append(experiment_conf)

        return experiments
