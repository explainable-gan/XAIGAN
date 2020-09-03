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
from copy import deepcopy
import torch
from torch.nn import functional as F
from captum.attr import DeepLiftShap, Saliency, InputXGradient, FeaturePermutation
from lime import lime_image

# defining global variables
global values
global discriminatorLime


def get_explanation(generated_data, discriminator, prediction, XAItype="shap", cuda=True, trained_data=None) -> None:
    # initialize temp values to all 1s
    temp = torch.ones(size=generated_data.size())

    # mask values with low prediction
    mask = (prediction < 0.5).squeeze()
    indices = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()

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

        elif XAItype == "lime":
            explainer = lime_image.LimeImageExplainer()
            global discriminatorLime
            discriminatorLime = deepcopy(discriminator)
            discriminatorLime.eval()
            for i in range(len(indices)):
                exp = explainer.explain_instance(data[i, :].unsqueeze(0).detach(), batch_predict, num_samples=100)
                _, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, negative_only=False)
                temp[indices[i], :] = torch.tensor(mask.astype(np.float))
            del discriminatorLime

    if cuda:
        temp = temp.cuda()
    set_values(normalize_vector(temp))


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

    # # clamp result
    # limit = 2e5
    # new_grad = torch.clamp(new_grad, min=grad_input[0].min()-limit,
    #                        max=grad_input[0].max()+limit)
    return (new_grad, )


def normalize_vector(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)


def get_values() -> np.array:
    """ get global values """
    global values
    return values


def set_values(x: np.array) -> None:
    """ set global values """
    global values
    values = x


def batch_predict(images):
    # convert images to greyscale
    images = np.mean(images, axis=3)
    # stack up all images
    batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
    batch = batch.squeeze()
    logits = discriminatorLime(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().numpy()


