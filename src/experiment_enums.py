from enum import Enum
from src.models.generators import GeneratorNet
from src.models.discriminators import DiscriminatorNet
from torch import nn, optim
from src.experiment import Experiment


class ExperimentEnums(Enum):
    MNIST100PN = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 100,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    MNIST35PN = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 100,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST100PN = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 100,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST35PN = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 100,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    def __str__(self):
        return self.value


# experimentsCurrent = [Experiment(experimentType=i) for i in [ExperimentEnums.MNIST100PN]]
experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
