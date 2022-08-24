# This code is referenced from MMEditing with modifications.
# Reference: https://github.com/open-mmlab/mmediting
# Original licence: Copyright (c) OpenMMLab, under the Apache 2.0 license.

import math
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, channel_growth, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[
            DenseLayer(in_channels + channel_growth * i, channel_growth)
            for i in range(num_layers)
        ])

        self.lff = nn.Conv2d(
            in_channels + channel_growth * num_layers,
            channel_growth,
            kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat):

        m = []
        if scale in [2, 4, 8]:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=3, padding=1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class RDN(nn.Module):
    def __init__(self,
                 n_channels=3,
                 mid_channels=64,
                 num_blocks=16,
                 scale=4,
                 num_layers=8,
                 channel_growth=64):
        """
        Args:
            n_channels (int): Channel number of image. Default: 3.
            mid_channels (int): Channel number of intermediate features. Default: 64.
            num_blocks (int): Block number in the trunk network. Default: 16.
            scale (int): Upsampling factor. Support 2, 3, and 4. Default: 4.
            num_layer (int): Layer number in the Residual Dense Block. Default: 8.
            channel_growth(int): Channels growth in each layer of RDB. Default: 64.
        """

        super().__init__()
        self.mid_channels = mid_channels
        self.channel_growth = channel_growth
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        self.sfe1 = nn.Conv2d(
            n_channels, mid_channels, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1)

        self.rdbs = nn.ModuleList(
            [ResidualDenseBlock(self.mid_channels, self.channel_growth, self.num_layers)])
        for _ in range(self.num_blocks - 1):
            self.rdbs.append(
                ResidualDenseBlock(self.channel_growth, self.channel_growth, self.num_layers))

        self.gff = nn.Sequential(
            nn.Conv2d(
                self.channel_growth * self.num_blocks,
                self.mid_channels,
                kernel_size=1),
            nn.Conv2d(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                padding=1))

        self.upscale = Upsampler(scale, self.mid_channels)

        self.output = nn.Conv2d(
            self.mid_channels, n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1
        x = self.upscale(x)
        x = self.output(x)
        return x


def rdn_x2():
    return RDN(scale=2)


def rdn_x3():
    return RDN(scale=3)


def rdn_x4():
    return RDN(scale=4)
