from enum import Enum
from models.generators import GeneratorNet, GeneratorNetCifar10
from models.discriminators import DiscriminatorNet, DiscriminatorNetCifar10
from torch import nn, optim
from experiment import Experiment


class ExperimentEnums(Enum):

    MNIST35Shap = {
        "explainable": True,
        "explanationType": "shap",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 128,
        "percentage": 0.35,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    MNIST35Lime = {
        "explainable": True,
        "explanationType": "lime",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 128,
        "percentage": 0.35,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    MNIST35Saliency = {
        "explainable": True,
        "explanationType": "saliency",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "mnist",
        "batchSize": 128,
        "percentage": 0.35,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST35Shap = {
        "explainable": True,
        "explanationType": "shap",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 128,
        "percentage": 0.35,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST35Lime = {
        "explainable": True,
        "explanationType": "lime",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 128,
        "percentage": 0.35,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    FMNIST35Saliency = {
        "explainable": True,
        "explanationType": "saliency",
        "generator": GeneratorNet,
        "discriminator": DiscriminatorNet,
        "dataset": "fmnist",
        "batchSize": 128,
        "percentage": 0.35,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    # CIFAR20SaliencyDiff = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.2,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR20ShapDiff = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.2,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }

    # CIFAR20LimeDiff = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.2,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }

    # CIFAR100Saliency = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 1000
    # }

    # MNIST100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST50Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 0.5,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }
    #
    # MNIST5Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 1000
    # }

    # CIFAR100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }

    # CIFAR100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }

    # CIFAR100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }

    # CIFAR100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }


    # CIFAR100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR50Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.5,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR35Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR10Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR5Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR20Lime = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.2,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR20Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.2,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR20Perturb = {
    #     "explainable": True,
    #     "explanationType": "perturb",
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.2,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # CIFAR20Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.2,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 30
    # }
    #
    # DemoMNIST = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 0.01,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 2
    # }

    # MNIST100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # FMNIST50Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.5,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST10Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # FMNIST5Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST100Lime = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST35Lime = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Lime = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Lime = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST5Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST100Input = {
    #     "explainable": True,
    #     "explanationType": "inputXGradient",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 2
    # }
    #
    # MNIST35Input = {
    #     "explainable": True,
    #     "explanationType": "inputXGradient",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Input = {
    #     "explainable": True,
    #     "explanationType": "inputXGradient",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Input = {
    #     "explainable": True,
    #     "explanationType": "inputXGradient",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST100Perturb = {
    #     "explainable": True,
    #     "explanationType": "perturb",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST35Perturb = {
    #     "explainable": True,
    #     "explanationType": "perturb",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Perturb = {
    #     "explainable": True,
    #     "explanationType": "perturb",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Perturb = {
    #     "explainable": True,
    #     "explanationType": "perturb",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST100Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST35Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST100Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST35Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST5Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST5Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST5Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST100Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST35Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # FMNIST100Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # MNIST35Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST100Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST35Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST35Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 100,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    def __str__(self):
        return self.value


# experimentsCurrent = [Experiment(experimentType=i) for i in [ExperimentEnums.MNIST100PN]]
experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
