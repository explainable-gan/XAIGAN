"""
This file is the main function that parses user argument and runs experiments.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from evaluation.metrics.fid_score import calculate_fid_given_paths
from evaluation.metrics.kid_score import calculate_kid_given_paths
from utils.vector_utils import noise
import torch
import argparse
import numpy as np
from get_data import get_loader
from models.generators import GeneratorNet, GeneratorNetCifar10


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--file", default="./../results/FMNIST50Normal/generator.pt",
                        help="path of the file")
    parser.add_argument("-t", "--type", default="fmnist", help="type of the dataset")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=10000,
                        help="number of samples to generate")

    args = parser.parse_args()
    calculate_metrics(path=args.file, numberOfSamples=args.number_of_samples, datasetType=args.type)


def calculate_metrics(path, numberOfSamples=1000, datasetType="mnist"):
    path_real = "./real.npy"
    generate_real_data(number=numberOfSamples, path=path_real, datasetType=datasetType)

    path_generated = "./generated.npy"
    generate_samples(number=numberOfSamples, path_model=path, path_output=path_generated, datasetType=datasetType)

    paths = [path_generated] + [path_real]

    fid = calculate_fid_given_paths(paths)
    print(f"FID Score. Mean: {fid[0][0]}, Std: {fid[0][1]}")

    kid = calculate_kid_given_paths(paths)
    print(f"KID Score. Mean: {kid[0][0]}, Std: {kid[0][1]}")
    return fid, kid


def generate_real_data(number: int, path: str, datasetType="mnist") -> None:
    # get background data and save
    loader = get_loader(number, 1, datasetType)
    batch = next(iter(loader))[0].detach()
    if datasetType == "cifar":
        batch = torch.mean(batch, 1).unsqueeze(1)
    batch = batch.view(number, 1, 32, 32)
    np.save(path, batch)


def generate_samples(number, path_model, path_output, datasetType="mnist"):
    if datasetType == "cifar":
        generator = GeneratorNetCifar10()
    else:
        generator = GeneratorNet()
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))
    samples = generator(noise(number, False)).detach()

    if datasetType == "cifar":
        samples = torch.mean(samples, 1).unsqueeze(1)
    samples = samples.view(number, 1, 32, 32)
    np.save(path_output, samples)


if __name__ == "__main__":
    main()
