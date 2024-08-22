"""
---
title: Cycle GAN
summary: >
  A simple PyTorch implementation/tutorial of Cycle GAN introduced in paper
  Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.
---

# Cycle GAN
I've taken pieces of code from [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN).
It is a very good resource if you want to checkout other GAN variations too.

Cycle GAN does image-to-image translation.
It trains a model to translate an image from given distribution to another, say, images of class A and B.
Images of a certain distribution could be things like images of a certain style, or nature.
The models do not need paired images between A and B.
Just a set of images of each class is enough.
This works very well on changing between image styles, lighting changes, pattern changes, etc.
For example, changing summer to winter, painting style to photos, and horses to zebras.

Cycle GAN trains two generator models and two discriminator models.
One generator translates images from A to B and the other from B to A.
The discriminators test whether the generated images look real.

This file contains the model code as well as the training code.
We also have a Google Colab notebook.

"""
import random
from typing import Tuple
import torch
import torch.nn as nn
from PIL import Image
from labml_helpers.module import Module


class GeneratorResNet(Module):
    def __init__(self, input_channels: int, n_residual_blocks: int):
        super().__init__()
        out_features = 64
        
        # Initial convolution layer
        layers = [
            nn.Conv2d(input_channels, out_features, kernel_size=3, stride=1, padding=1),  # 3x3 kernel, stride 1, padding 1
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Down-sample layers, still preserving the 30x30 size
        for _ in range(2):
            out_features *= 2
            layers += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),  # 3x3 kernel, stride 1, padding 1
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks, maintaining the 30x30 size
        for _ in range(n_residual_blocks):
            layers += [ResidualBlock(out_features)]

        # Up-sample layers, still preserving the 30x30 size
        for _ in range(2):
            out_features //= 2
            layers += [
                nn.Upsample(scale_factor=2),  # This increases dimensions, needs careful control with Conv2d
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),  # 3x3 kernel, stride 1, padding 1
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Final layer, preserving the 30x30 size
        layers += [nn.Conv2d(out_features, input_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()]

        self.layers = nn.Sequential(*layers)
        self.apply(weights_init_normal)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(Module):
    """
    This is the residual block, with two convolution layers.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),  # 3x3 kernel, stride 1, padding 1
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),  # 3x3 kernel, stride 1, padding 1
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x: torch.Tensor):
        return x + self.block(x)


class Discriminator(Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        channels, height, width = input_shape

        self.layers = nn.Sequential(
            DiscriminatorBlock(channels, 64, normalize=False),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            DiscriminatorBlock(512, 1024),
            DiscriminatorBlock(1024, 512),
            DiscriminatorBlock(512, 256),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)  # Final layer, maintaining size
        )

        self.output_shape = (1, height, width)  # Ensure the output shape remains consistent

        self.apply(weights_init_normal)

    def forward(self, img):
        return self.layers(img)


class DiscriminatorBlock(Module):
    def __init__(self, in_filters: int, out_filters: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1),  # Preserve dimensions
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


def weights_init_normal(m):
    """
    Initialize convolution layer weights
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def load_image(path: str):
    """
    Load an image and change to RGB if in grey-scale.
    """
    image = Image.open(path)
    if image.mode != 'RGB':
        image = Image.new("RGB", image.size).paste(image)

    return image


class ReplayBuffer:
    """
    ### Replay Buffer

    Replay buffer is used to train the discriminator.
    Generated images are added to the replay buffer and sampled from it.

    The replay buffer returns the newly added image with a probability of $0.5$.
    Otherwise, it sends an older generated image and replaces the older image
    with the newly generated image.

    This is done to reduce model oscillation.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor):
        """Add/retrieve an image"""
        data = data.detach()
        res = []
        for element in data:
            if len(self.data) < self.max_size:
                self.data.append(element)
                res.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    res.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    res.append(element)
        return torch.stack(res)
