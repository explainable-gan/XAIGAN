"""
This file contains the experiment method which runs the experiments

Date:
    November 6, 2019

Project:
    XAI-GAN

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott, Vijay Ganesh
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.experiment_enums import ExperimentEnums
from src.get_data import get_loader
from src.utils.vector_utils import noise, ones_target, zeros_target, images_to_vectors, vectors_to_images
from src.logger import Logger
from torch.autograd import Variable
from torch import nn
import torch


class Experiment:

    def __init__(self, experimentType, verbose=False, cuda=False):
        if experimentType not in ExperimentEnums:
            raise Exception(f"Type: {experimentType} not defined.")
        self.name = experimentType.name
        self.type = experimentType.value
        self.explainable = self.type["explainable"]
        self.explanationType = self.type["explanationType"]
        self.generator = self.type["generator"]()
        self.discriminator = self.type["discriminator"]()
        self.g_optim = self.type["g_optim"](self.generator.parameters(), lr=self.type["glr"], betas=(0.5, 0.99))
        self.d_optim = self.type["d_optim"](self.discriminator.parameters(), lr=self.type["dlr"], betas=(0.5, 0.99))
        self.loss = self.type["loss"]

        self.verbose = verbose
        self.cuda = cuda
        # self.trainloader, self.testloader = mnist_data(self.type["batchSize"])
        self.epochs = self.type["epochs"]

    def run(self, logging_frequency=4) -> (list, list):
        logger = Logger(self.name, samples=16)
        loader = get_loader(self.type["batchSize"], self.type["percentage"], self.type["dataset"])

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        test_noise = noise(logger.samples)
        num_batches = len(loader)

        # track losses
        G_losses = []
        D_losses = []

        # Start training
        for epoch in range(1, self.epochs + 1):

            for n_batch, (real_batch, _) in enumerate(loader):

                N = real_batch.size(0)

                # 1. Train Discriminator
                real_data = Variable(images_to_vectors(real_batch))

                if self.cuda:
                    real_data = real_data.cuda()

                # Generate fake data and detach (so gradients are not calculated for generator)
                fake_data = self.generator(noise(N)).detach()

                # Train D
                d_error, d_pred_real, d_pred_fake = self._train_discriminator(real_data=real_data, fake_data=fake_data)

                # 2. Train Generator
                # Generate fake data
                fake_data = self.generator(noise(N))

                # Train G
                g_error = self._train_generator(fake_data=fake_data)

                # Save Losses for plotting later
                G_losses.append(g_error.item())
                D_losses.append(d_error.item())

                logger.log(d_error, g_error, epoch, n_batch, num_batches)

                if n_batch % (num_batches // logging_frequency) == 0:
                    test_images = vectors_to_images(self.generator(test_noise)).data
                    logger.log_images(test_images, epoch, n_batch, num_batches)

                    # Display status Logs
                    logger.display_status(
                        epoch, self.epochs, n_batch, num_batches,
                        d_error, g_error, d_pred_real, d_pred_fake
                    )

        logger.save_models(generator=self.generator, discriminator=self.discriminator)

        return G_losses, D_losses

    def _train_generator(self, fake_data: torch.Tensor) -> torch.Tensor:
        """
        This function performs one iteration of training the generator
        :param discriminator: DiscriminatorNet discriminator neural network
        :param optimizer: torch.optim the optimizer for generator
        :param loss: torch.loss the loss function
        :param fake_data: tensor data created by generator
        :param explainable: bool whether to use explanations system
        :return: error of generator on this training step
        """
        N = fake_data.size(0)

        # Reset gradients
        self.g_optim.zero_grad()

        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data).squeeze()

        # Calculate error and back-propagate
        error = self.loss(prediction, ones_target(N))

        error.backward()

        # clip gradients to avoid exploding gradient problem
        nn.utils.clip_grad_norm_(self.generator.parameters(), 10)

        self.g_optim.step()

        # Return error
        return error

    def _train_discriminator(self, real_data: Variable, fake_data: torch.Tensor):

        N = real_data.size(0)

        # Reset gradients
        self.d_optim.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_data).squeeze()

        # Calculate error and backpropagate
        error_real = self.loss(prediction_real, ones_target(N))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(fake_data).squeeze()

        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, zeros_target(N))
        error_fake.backward()

        # 1.3 Update weights with gradients
        self.d_optim.step()

        # Return error and predictions for real and fake inputs
        return (error_real + error_fake) / 2, prediction_real, prediction_fake
