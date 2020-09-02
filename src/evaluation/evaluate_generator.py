import sys
from src.evaluation.metrics.fid_score import calculate_fid_given_paths
from src.evaluation.metrics.kid_score import calculate_kid_given_paths
from src.utils.vector_utils import noise, images_to_vectors
import torch
import argparse
import numpy as np
from src.get_data import get_loader
from src.models.generators import GeneratorNet


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--file", default="./../results/models/MNIST5Perturb/2020-09-02/generator.pt",
                        help="path of the file")
    parser.add_argument("-t", "--type", default="mnist", help="type of the dataset")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=10000,
                        help="number of samples to generate")

    args = parser.parse_args()
    calculate_metrics(path=args.file, numberOfSamples=args.number_of_samples, type=args.type)


def calculate_metrics(path, numberOfSamples=1000, type="mnist"):
    path_real = "./real.npy"
    generate_real_data(number=numberOfSamples, path=path_real, type=type)

    path_generated = "./generated.npy"
    generate_samples(number=numberOfSamples, path_model=path, path_output=path_generated)

    paths = [path_generated] + [path_real]

    fid_mean = calculate_fid_given_paths(paths)
    print(f"FID Score: {fid_mean}")

    kid_mean = calculate_kid_given_paths(paths)
    print(f"KID Score: {kid_mean}")


def generate_real_data(number: int,  path: str, type="mnist") -> None:
    # get background data and save
    loader = get_loader(number, 1, type)
    batch = next(iter(loader))[0].detach()
    batch = batch.view(number, 1, 32, 32)
    np.save(path, batch)


def generate_samples(number, path_model, path_output):
    generator = GeneratorNet()
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))
    print("reached")
    samples = generator(noise(number, False)).detach()
    samples = samples.view(number, 1, 32, 32)
    np.save(path_output, samples)


if __name__ == "__main__":
    main()
