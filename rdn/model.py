import math
import megengine.module as M
import megengine.functional as F
import megengine.hub as hub


class DenseLayer(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = M.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = M.ReLU()

    def forward(self, x):
        return F.concat([x, self.relu(self.conv(x))], 1)


class ResidualDenseBlock(M.Module):
    def __init__(self, in_channels, channel_growth, num_layers):
        super().__init__()
        self.layers = M.Sequential(*[
            DenseLayer(in_channels + channel_growth * i, channel_growth)
            for i in range(num_layers)
        ])

        self.lfuse = M.Conv2d(
            in_channels + channel_growth * num_layers,
            channel_growth,
            kernel_size=1)

    def forward(self, x):
        return x + self.lfuse(self.layers(x))


class Upsampler(M.Sequential):
    def __init__(self, scale, n_feat):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(M.Conv2d(n_feat, 4 * n_feat, kernel_size=3, padding=1))
                m.append(M.PixelShuffle(2))
        elif scale == 3:
            m.append(M.Conv2d(n_feat, 9 * n_feat, kernel_size=3, padding=1))
            m.append(M.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class RDN(M.Module):
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

        self.sfe1 = M.Conv2d(
            n_channels, mid_channels, kernel_size=3, padding=1)
        self.sfe2 = M.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1)

        self.rdbs = [ResidualDenseBlock(self.mid_channels, self.channel_growth, self.num_layers)]
        for _ in range(self.num_blocks - 1):
            self.rdbs.append(
                ResidualDenseBlock(self.channel_growth, self.channel_growth, self.num_layers))

        self.gfuse = M.Sequential(
            M.Conv2d(
                self.channel_growth * self.num_blocks,
                self.mid_channels,
                kernel_size=1),
            M.Conv2d(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                padding=1))

        self.upscale = Upsampler(scale, self.mid_channels)

        self.output = M.Conv2d(
            self.mid_channels, n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gfuse(F.concat(local_features, 1)) + sfe1
        x = self.upscale(x)
        x = self.output(x)
        return x

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/118/files/83b4849e-1d23-4a0b-9d55-7ea8a974d4fa"
)
def rdn_x2():
    return RDN(scale=2)

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/118/files/4b4de99d-08ac-402e-b2a3-0d06031586c9"
)
def rdn_x3():
    return RDN(scale=3)

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/118/files/513accff-18df-417d-b43f-a1d4396f6c26"
)
def rdn_x4():
    return RDN(scale=4)
