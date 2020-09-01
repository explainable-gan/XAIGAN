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

from src.get_data import get_loader
from src.utils.vector_utils import noise, ones_target, zeros_target, images_to_vectors, vectors_to_images
from src.logger import Logger
from torch.autograd import Variable
from torch import nn
import torch


class Experiment:
    def __init__(self, experimentType):
        self.name = experimentType.name
        self.type = experimentType.value
        self.explainable = self.type["explainable"]
        self.explanationType = self.type["explanationType"]
        self.generator = self.type["generator"]()
        self.discriminator = self.type["discriminator"]()
        self.g_optim = self.type["g_optim"](self.generator.parameters(), lr=self.type["glr"], betas=(0.5, 0.99))
        self.d_optim = self.type["d_optim"](self.discriminator.parameters(), lr=self.type["dlr"], betas=(0.5, 0.99))
        self.loss = self.type["loss"]
        self.epochs = self.type["epochs"]
        self.cuda = True if torch.cuda.is_available() else False

    def run(self, logging_frequency=4) -> (list, list):

        logger = Logger(self.name, samples=16)
        test_noise = noise(logger.samples, self.cuda)

        loader = get_loader(self.type["batchSize"], self.type["percentage"], self.type["dataset"])
        num_batches = len(loader)

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.loss = self.loss.cuda()

        # track losses
        G_losses = []
        D_losses = []

        # Start training
        for epoch in range(1, self.epochs + 1):

            for n_batch, (real_batch, _) in enumerate(loader):

                N = real_batch.size(0)

                # 1. Train Discriminator
                real_data = Variable(images_to_vectors(real_batch))

                # Generate fake data and detach (so gradients are not calculated for generator)
                fake_data = self.generator(noise(N, self.cuda)).detach()

                if self.cuda:
                    real_data = real_data.cuda()
                    fake_data = fake_data.cuda()

                # Train D
                d_error, d_pred_real, d_pred_fake = self._train_discriminator(real_data=real_data, fake_data=fake_data)

                # 2. Train Generator
                # Generate fake data
                fake_data = self.generator(noise(N, self.cuda))

                if self.cuda:
                    fake_data = fake_data.cuda()

                # Train G
                g_error = self._train_generator(fake_data=fake_data)

                # Save Losses for plotting later
                G_losses.append(g_error.item())
                D_losses.append(d_error.item())

                logger.log(d_error, g_error, epoch, n_batch, num_batches)

                if n_batch % (num_batches // logging_frequency) == 0:
                    test_images = vectors_to_images(self.generator(test_noise)).cpu().data
                    logger.log_images(test_images, epoch, n_batch, num_batches)

                    # Display status Logs
                    logger.display_status(
                        epoch, self.epochs, n_batch, num_batches,
                        d_error, g_error, d_pred_real, d_pred_fake
                    )

        logger.save_models(generator=self.generator, discriminator=self.discriminator)
        logger.save_errors(g_loss=G_losses, d_loss=D_losses)
        return

    def _train_generator(self, fake_data: torch.Tensor) -> torch.Tensor:
        """
        This function performs one iteration of training the generator
        :param fake_data: tensor data created by generator
        :return: error of generator on this training step
        """
        N = fake_data.size(0)

        # Reset gradients
        self.g_optim.zero_grad()

        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data).squeeze()

        # Calculate error and back-propagate
        error = self.loss(prediction, zeros_target(N, self.cuda))

        error.backward()

        # clip gradients to avoid exploding gradient problem
        nn.utils.clip_grad_norm_(self.generator.parameters(), 10)

        # update parameters
        self.g_optim.step()

        # Return error
        return error

    def _train_discriminator(self, real_data: Variable, fake_data: torch.Tensor):
        """
        This function performs one iteration of training the discriminator
        :param real_data:
        :type real_data:
        :param fake_data:
        :type fake_data:
        :return:
        :rtype:
        """
        N = real_data.size(0)

        # Reset gradients
        self.d_optim.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_data).squeeze()

        # Calculate error
        error_real = self.loss(prediction_real, zeros_target(N, self.cuda))

        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(fake_data).squeeze()

        # Calculate error
        error_fake = self.loss(prediction_fake, ones_target(N, self.cuda))

        # Sum up error and backpropagate
        error = error_real + error_fake
        error.backward()

        # 1.3 Update weights with gradients
        self.d_optim.step()

        # Return error and predictions for real and fake inputs
        return (error_real + error_fake)/2, prediction_real, prediction_fake
