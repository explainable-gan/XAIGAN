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
        "epochs": 10
    }

    MNIST100PSHAP = {
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
        "epochs": 10
    }

    MNIST100PLIME = {
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
        "epochs": 10
    }

    MNIST100PSALIENCY = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 100,
        "percentage": 0.5,
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
        "epochs": 10
    }

    MNIST35PSHAP = {
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
        "epochs": 10
    }

    MNIST35PLIME = {
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
        "epochs": 10
    }

    MNIST35PSALIENCY = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 100,
        "percentage": 0.5,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 10
    }

    def __str__(self):
        return self.value


experimentsCurrent = [Experiment(experimentType=i, verbose=False, cuda=False) for i in [ExperimentEnums.MNIST100PN,
                                                                                        ExperimentEnums.MNIST35PN]]
experimentsAll = [Experiment(experimentType=i, verbose=False, cuda=False) for i in ExperimentEnums]
