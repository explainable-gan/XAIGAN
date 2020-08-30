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
from src.utils.vector_utils import vectors_to_images


# defining global variables
values = None


def shap_explanation(background_selector, fake_data: torch.Tensor, predictor) -> None:
    """
    This function calls the shap module, computes the mask and sets new gradient values
    :param background_selector: background selector that gives background data
    :param fake_data: the data to predict
    :param predictor: the classifier model
    :return:
    """
    background_data = background_selector.get_background(vectors_to_images(fake_data))
    explainer = shap.DeepExplainer(predictor, background_data)
    temp = normalize_shap(np.absolute(explainer.shap_values(fake_data.detach())))
    set_values(temp)


def explanation_hook(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    temp = torch.from_numpy(get_values())
    new_grad = grad_output[0] * temp
    return new_grad,


def normalize_shap(vector: np.array) -> np.array:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector = ((vector-vector.min())/(vector.max()-vector.min()))
    return np.float32(vector)


def get_values() -> np.array:
    """ get global values """
    global values
    return values


def set_values(x: np.array) -> None:
    """ set global values """
    global values
    values = x
