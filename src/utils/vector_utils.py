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
    return images.view(images.size(0), 32 * 32)


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
    return vectors.reshape(vectors.shape[0], 32, 32)


def vectors_to_images(vectors):
    """ converts (Nx784) tensor to (Nx32x32) tensor """
    return vectors.view(vectors.size(0), 1, 32, 32)


def noise(size: int, cuda: False) -> Variable:
    """ generates a 1-d vector of normal sampled random values of mean 0 and standard deviation 1 """
    result = Variable(torch.randn(size, 100))
    if cuda:
        result = result.cuda()
    return result


def ones_target(size: int, cuda: False) -> Variable:
    """ returns tensor filled with 1s of given size """
    result = Variable(torch.ones(size))
    if cuda:
        result = result.cuda()
    return result


def values_target(size: tuple, value: int, cuda: False) -> Variable:
    """ returns tensor filled with value of given size """
    result = Variable(torch.full(size=size, fill_value=value))
    if cuda:
        result = result.cuda()
    return result


def zeros_target(size: int, cuda: False) -> Variable:
    """ returns tensor filled with 0s of given size """
    result = Variable(torch.zeros(size))
    if cuda:
        result = result.cuda()
    return result


def normal_init(m, mean, std):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
