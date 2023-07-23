from uuid import uuid4


class Experiment:
    def __init__(self, experiment_conf):
        self.uuid = str(uuid4())
        self.experiment_conf = experiment_conf
        self.dataset = None
        self.solution = None
