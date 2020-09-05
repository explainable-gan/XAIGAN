"""
This file is the main function that parses user argument and runs experiments.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from utils.vector_utils import noise_cifar
import torch
import argparse
import numpy as np
from models.generators import GeneratorNetCifar10
import os
from PIL import Image
import glob
import cv2 as cv
import tensorflow as tf
import sys
sys.path.append("..")


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--file", default="./../results/CIFAR100Normal/generator.pt", help="path of the file")
    parser.add_argument("-n", "--number_of_samples",
                        type=int, default=2048,
                        help="number of samples to generate")

    args = parser.parse_args()
    calculate_metrics_cifar(path=args.file, numberOfSamples=args.number_of_samples)


def calculate_metrics_cifar(path, numberOfSamples=2048):
    from fid import fid
    folder = f'{os.getcwd()}/tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    generate_samples_cifar(number=numberOfSamples, path_model=path, path_output=folder)

    image_list = glob.glob(os.path.join(folder, '*.jpg'))
    images = np.array([cv.imread(str(fn)).astype(np.float32) for fn in image_list])

    stats_path = f"{os.getcwd()}/evaluation/metrics/cifar_fid_files/cifar10stats.npz"
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]

    inception_folder = "./metrics/cifar_fid_files/"
    inception_path = fid.check_or_download_inception(inception_folder)
    fid.create_inception_graph(inception_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print(f'FID: {fid_value}')
    return fid_value


def generate_samples_cifar(number, path_model, path_output):
    generator = GeneratorNetCifar10()
    generator.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))
    for i in range(number):
        sample = generator(noise_cifar(1, False)).detach().squeeze(0).numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = ((sample/2) + 0.5) * 255
        sample = sample.astype(np.uint8)
        image = Image.fromarray(sample)
        image.save(f'{path_output}/{i}.jpg')
    return


if __name__ == "__main__":
    main()
