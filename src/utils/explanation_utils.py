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
from captum.attr import DeepLiftShap, Saliency, InputXGradient, FeaturePermutation

# defining global variables
global values


def get_explanation(generated_data, discriminator, prediction, XAItype="shap", cuda=True, trained_data=None) -> None:
    # initialize temp values to all 1s
    temp = torch.ones(size=generated_data.size())

    # mask values with low prediction
    mask = (prediction < 0.5).squeeze()
    indices = (mask.nonzero(as_tuple=False)).cpu().numpy().flatten().tolist()

    data = generated_data[mask, :]

    if data.size(0) > 0:
        if XAItype == "saliency":
            for i in range(len(indices)):
                explainer = Saliency(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach())

        elif XAItype == "shap":
            for i in range(len(indices)):
                explainer = DeepLiftShap(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0), trained_data, target=0)

        elif XAItype == "inputXGradient":
            for i in range(len(indices)):
                explainer = InputXGradient(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach())

        elif XAItype == "perturb":
            for i in range(len(indices)):
                explainer = FeaturePermutation(discriminator)
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

    # limit = 2e5

    # # clamp result
    # new_grad = torch.clamp(new_grad, min=grad_input[0].min()-limit,
    #                        max=grad_input[0].max()+limit)
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
