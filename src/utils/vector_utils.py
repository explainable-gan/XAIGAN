"""
This file is the util files needed for the experiments. Some functions from https://github.com/diegoalejogm/gans/

Date:
    November 6, 2019

Project:
    LogicGAN

"""

import numpy as np
import torch
from torch.autograd.variable import Variable


# functions to reshape images
def images_to_vectors(images: torch.Tensor) -> torch.Tensor:
    """ converts (Nx28x28) tensor to (Nx784) torch tensor """
    return images.view(images.size(0), 784)


def images_to_vectors_numpy(images: np.array) -> torch.Tensor:
    """ converts (Nx28x28) np array to (Nx784) torch tensor """
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2], images.shape[3])
    return torch.from_numpy(images[:, :, 0])


def images_to_vectors_numpy_multiclass(images: np.array) -> torch.Tensor:
    """ converts (Nx28x28) numpy array to (Nx784) tensor in multiclass setting"""
    images = images.reshape(images.shape[0], images.shape[2]*images.shape[3], images.shape[1])
    return torch.from_numpy(images[:, :, 0])


def vectors_to_images_numpy(vectors: np.array) -> np.array:
    """ converts (Nx784) tensor to (Nx28x28) numpy array """
    return vectors.reshape(vectors.shape[0], 1, 28, 28)


def vectors_to_images(vectors):
    """ converts (Nx784) tensor to (Nx28x28) tensor """
    return vectors.view(vectors.size(0), 1, 28, 28)


def noise(size: int) -> Variable:
    """ generates a 1-d vector of normal sampled random values of mean 0 and standard deviation 1 """
    return Variable(torch.randn(size, 100))


def ones_target(size: int) -> Variable:
    """ returns tensor filled with 1s of given size """
    return Variable(torch.ones(size))


def values_target(size: tuple, value: int) -> Variable:
    """ returns tensor filled with value of given size """
    return Variable(torch.full(size=size, fill_value=value, dtype=torch.long))


def zeros_target(size: int) -> Variable:
    """ returns tensor filled with 0s of given size """
    return Variable(torch.zeros(size))


def ema(l: list, decay=0.999) -> (list, float):
    """ get the moving average of a list. decay controls the smoothness """
    ret = [l[0]]
    for i in range(1, len(l)):
        ret.append(l[i] + ret[-1] * decay)
    val, last = 1.0, -1.0
    for i in range(len(l)):
        val = val * decay + 1.0
        ret[i] /= val
    return ret, round(val)
