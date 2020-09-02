"""
This file is the explanation util files needed for the experiments.

Date:
    November 14, 2019

Project:
    LogicGAN

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott, Vijay Ganesh
    contact: vineel.nagisetty@uwaterloo.ca
"""

import numpy as np
import torch
from captum.attr import GradientShap, DeepLift, IntegratedGradients, Saliency
from src.utils.vector_utils import vectors_to_images


# defining global variables
values = None


def get_explanation(generated_data, discriminator, prediction, XAItype="saliency", cuda=True) -> None:
    """
    This function calls the shap module, computes the mask and sets new gradient values
    :param background_selector: background selector that gives background data
    :param fake_data: the data to predict
    :param predictor: the classifier model
    :return:
    """
    # initialize temp values to all 1s
    temp = torch.ones(size=generated_data.size())

    # mask values with low prediction
    mask = (prediction < 0.5).squeeze()
    indices = (mask.nonzero()).cpu().numpy().flatten().tolist()

    data = generated_data[mask, :]

    if data.size(0) > 0:
        if XAItype == "saliency":
            for i in range(len(indices)):
                explainer = Saliency(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach())

    if cuda:
        temp = temp.cuda()
    set_values(temp)


def explanation_hook(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    # get stored mask
    temp = get_values()

    # multiply with mask
    new_grad = grad_input[0] + 0.2 * (grad_input[0] * temp)

    limit = 2e5

    # clamp result
    new_grad = torch.clamp(new_grad, min=grad_input[0].min()-limit,
                           max=grad_input[0].max()+limit)
    return (new_grad, )


def normalize_vector(vector: np.array) -> np.array:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    return np.float32(vector)


def get_values() -> np.array:
    """ get global values """
    global values
    return values


def set_values(x: np.array) -> None:
    """ set global values """
    global values
    values = x
