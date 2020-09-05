"""
This file contains the generator class

Date:
    November 6, 2019

Project:
    LogicGAN

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott, Vijay Ganesh
    contact: vineel.nagisetty@uwaterloo.ca
"""

from torch import nn, Tensor
import numpy as np


class GeneratorNet(nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.n_features = 100
        self.n_out = (1, 32, 32)

        self.input_layer = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1296),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1296, int(np.prod(self.n_out))),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        """ overrides the __call__ method of the generator """
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        x = x.view(x.size(0), *self.n_out)
        return x


class GeneratorNetCifar10(nn.Module):

    def __init__(self):
        super(GeneratorNetCifar10, self).__init__()
        self.n_features = 100
        self.n_out = (3, 32, 32)

        self.input_layer = nn.Sequential(
            nn.Linear(self.n_features, 128, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden4 = nn.Sequential(
            nn.Linear(1024, 2048, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Linear(2048, int(np.prod(self.n_out))),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
        x = x.view(x.size(0), *self.n_out)
        return x
