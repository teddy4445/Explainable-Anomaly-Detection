import datetime
import traceback
import json

from configuration_manager import ConfigurationManager
from dataloader import Dataloader
from vizualization_manager import VisualizationManager
from experiment import Experiment
import solvers as solvers
import scorers.score_function as score_function
import scorers.similarity_metrics as similarity_metrics
from baseline_explainers import BASELINE_EXPLAINERS


class Main:
    def __init__(self, config_filename=None, file_format="yaml"):
        self.conf_manager = ConfigurationManager(config_filename=config_filename, file_format=file_format)
        self.dataloader = Dataloader()
        self.vm = VisualizationManager()
        self.experiment_batch = {}

    @staticmethod
    def dump_experiment(experiment, directory='results'):
        filename = f'{experiment.experiment_conf["dataset_filename"].replace(".", "_")}_' \
                   f'{experiment.experiment_conf["solver"]["type"]}_' \
                   f'{experiment.experiment_conf["scorer"]["type"]}_' \
                   f'{experiment.experiment_conf["similarity_metric"]}_' \
                   f'{experiment.uuid}'
        csv_filename = f"{directory}/{filename}_d_inf.csv"

        experiment.solution['d_inf'].to_csv(csv_filename, index=False)

        exp2dump = {
            'conf': experiment.experiment_conf,
            'solution': {key: experiment.solution[key] for key in experiment.solution if key != 'd_inf'},
            'd_inf_filename': csv_filename
        }
        with open(f"results/{filename}.json", "w") as outfile:
            json.dump(exp2dump, outfile, indent=4)

    def dump_batch_metadata(self):
        metadata = {
            "batch_conf": self.conf_manager.configuration,
            "experiments": {exp_id: self.experiment_batch[exp_id].experiment_conf for exp_id in self.experiment_batch}
        }
        filename = f"batch_metadata_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with open(f"results/{filename}.json", "w") as outfile:
            json.dump(metadata, outfile, indent=4)

    def visualize_experiment(self, experiment, method='tsne', plot=True, annotate=False, save=True):
        self.vm.visualize(d_inf=experiment.solution['d_inf'],
                          f_diff=experiment.solution['f_diff'], f_sim=experiment.solution['f_sim'],
                          method=method, plot=plot, annotate=annotate,
                          save_filename=f'results/{experiment.uuid}' if save else None)

    def setup_experiment(self, experiment, dataset):
        self.experiment_batch[experiment.uuid] = experiment
        conf = experiment.experiment_conf

        experiment.dataset = dataset

        similarity_metric = similarity_metrics.SIMILARITY_FUNCTIONS[conf["similarity_metric"]]()
        scorer_class = score_function.SCORERS[conf["scorer"]["type"]]
        scorer = scorer_class(sim_module=similarity_metric, **conf["scorer"]["params"])
        solver_class = solvers.SOLVERS[conf["solver"]["type"]]
        solver = solver_class(scorer=scorer, data=experiment.dataset.data_wo_anomaly,
                              anomaly=experiment.dataset.anomaly_row, **conf["solver"]["params"])
        return solver

    def run(self):
        for batch in self.conf_manager.get_experiment_batches():
            batch_dataset, batch_model = self.dataloader.load_dataset(data_filename=batch["dataset"],
                                                                      supervised=batch["supervised"],
                                                                      model_name=batch["model"])
            for experiment_conf in batch["experiment_list"]:
                curr_experiment = Experiment(experiment_conf=experiment_conf)
                solver = self.setup_experiment(experiment=curr_experiment, dataset=batch_dataset)

                try:
                    print(f'Solving dataset {experiment_conf["dataset_filename"]} '
                          f'using {experiment_conf["solver"]["type"]} solver '
                          f'and {experiment_conf["scorer"]["type"]} scorer')
                    curr_experiment.solution = solver.solve()
                    # self.visualize_experiment(experiment=curr_experiment, method='tsne', plot=True, annotate=False,
                    #                           save=False)
                    # self.visualize_experiment(experiment=curr_experiment, method='tsne', plot=True, annotate=False,
                    #                           save=True)
                    # self.dump_experiment(experiment=curr_experiment)

                except Exception as e:
                    print(f'Failed to solve dataset {experiment_conf["dataset_filename"]} '
                          f'using {experiment_conf["solver"]["type"]} solver and {experiment_conf["scorer"]["type"]} scorer'
                          f' - moving on to next experiment')
                    traceback.print_exc()

            for alternative_explainer in batch["alternative_explainers"]:
                explainer = BASELINE_EXPLAINERS[alternative_explainer](data=batch_dataset.data_wo_anomaly,
                                                                       model=batch_model,
                                                                       mode='clf' if batch["supervised"] else 'ad')
                explanation = explainer.get_explanation(anomaly=batch_dataset.anomaly_row)
                # pass
                print()

        self.dump_batch_metadata()


if __name__ == '__main__':
    main = Main(config_filename="template_conf.yaml", file_format="yaml")
    main.run()
