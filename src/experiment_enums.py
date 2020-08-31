from enum import Enum
from src.models.generators import GeneratorNetDC
from src.models.discriminators import DiscriminatorNetDC
from torch import nn, optim
from src.experiment import Experiment


class ExperimentEnums(Enum):
    MNIST100PN = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNetDC,
        "discriminator": DiscriminatorNetDC,
        "dataset": "mnist",
        "batchSize": 100,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 10
    }

    MNIST35PN = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNetDC,
        "discriminator": DiscriminatorNetDC,
        "dataset": "mnist",
        "batchSize": 100,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 10
    }

    def __str__(self):
        return self.value


experimentsCurrent = [Experiment(experimentType=i) for i in [ExperimentEnums.MNIST100PN, ExperimentEnums.MNIST35PN]]
experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
