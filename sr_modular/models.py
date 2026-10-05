from __future__ import annotations

import math

import torch
from torch import nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super().__init__()
        if activation is not None:
            activation = activation.lower()
            assert activation in {"prelu", "leakyrelu", "tanh"}

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        ]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "prelu":
            layers.append(nn.PReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "tanh":
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class SubPixelConvolutionalBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * (scaling_factor ** 2),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super().__init__()
        self.conv_block1 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation="prelu",
        )
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation=None,
        )

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x + residual


class SRResNet(nn.Module):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super().__init__()
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}

        self.conv_block1 = ConvolutionalBlock(
            in_channels=3,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="prelu",
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for _ in range(n_blocks)]
        )
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=True,
            activation=None,
        )
        n_subpixel_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[
                SubPixelConvolutionalBlock(
                    kernel_size=small_kernel_size,
                    n_channels=n_channels,
                    scaling_factor=2,
                )
                for _ in range(n_subpixel_blocks)
            ]
        )
        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="tanh",
        )

    def forward(self, lr_imgs):
        x = self.conv_block1(lr_imgs)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv_block2(x)
        x = x + residual
        x = self.subpixel_convolutional_blocks(x)
        return self.conv_block3(x)


class Generator(nn.Module):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super().__init__()
        self.net = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )

    def initialize_with_srresnet_state_dict(self, srresnet_state_dict: dict):
        self.net.load_state_dict(srresnet_state_dict)
        print("Loaded weights from pre-trained SRResNet state_dict.")

    def forward(self, lr_imgs):
        return self.net(lr_imgs)


class Discriminator(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        super().__init__()
        in_channels = 3
        conv_blocks = []

        for i in range(n_blocks):
            if i % 2 == 0:
                out_channels = n_channels if i == 0 else in_channels * 2
                stride = 1
            else:
                out_channels = in_channels
                stride = 2

            conv_blocks.append(
                ConvolutionalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    batch_norm=(i != 0),
                    activation="leakyrelu",
                )
            )
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        x = self.conv_blocks(imgs)
        x = self.adaptive_pool(x)
        x = self.fc1(x.view(batch_size, -1))
        x = self.leaky_relu(x)
        return self.fc2(x)


class TruncatedVGG19(nn.Module):
    def __init__(self, i: int, j: int):
        super().__init__()
        import torchvision

        try:
            weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
            vgg19 = torchvision.models.vgg19(weights=weights)
        except AttributeError:
            vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        assert maxpool_counter == i - 1 and conv_counter == j, f"Invalid VGG19 truncation point i={i}, j={j}"
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.truncated_vgg19(x)
