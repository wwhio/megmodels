import megengine
import megengine.module as M
import megengine.functional as F
import megengine.hub as hub


class MeanShift(M.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = megengine.tensor(rgb_std)
        weight = F.eye(3).reshape(3, 3, 1, 1) / std.reshape(3, 1, 1, 1)
        bias = sign * rgb_range * megengine.tensor(rgb_mean) / std
        bias = bias.reshape(1, 3, 1, 1)

        self.weight = megengine.Tensor(weight)
        self.bias = megengine.Tensor(bias)


class ResidualBlock(M.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale

        self.conv1 = M.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=1)
        self.act = act
        self.conv2 = M.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        residual = self.conv2(self.act(self.conv1(x)))
        return x + residual * self.res_scale


class Upsampler(M.Sequential):
    def __init__(self, scale_factor, nf):
        super(Upsampler, self).__init__()
        self.scale = scale_factor

        block = []

        if scale_factor == 3:
            block += [
                M.Conv2d(nf, nf * 9, 3, padding=1, bias=True)
            ]
            self.pixel_shuffle = M.PixelShuffle(3)
        else:
            self.block_num = scale_factor // 2
            self.pixel_shuffle = M.PixelShuffle(2)

            for _ in range(self.block_num):
                block += [
                    M.Conv2d(nf, nf * (2 ** 2), 3, padding=1, bias=True)
                ]
        self.blocks = M.Sequential(*block)

    def forward(self, x):
        res = x

        if self.scale == 3:
            for block in self.blocks:
                res = block(x)
                res = self.pixel_shuffle(x)
        else:
            for block in self.blocks:
                res = block(res)
                res = self.pixel_shuffle(res)

        return res


class EDSR_S(M.Module):
    def __init__(self, n_colors=3, n_resblocks=32, n_feats=64, res_scale=0.1):
        super(EDSR_S, self).__init__()

        self.n_colors = n_colors
        self.n_feats = n_feats
        self.kernel_size = 3
        scale = 4
        act = M.ReLU()
        self.sub_mean = MeanShift(rgb_range=255)
        self.add_mean = MeanShift(rgb_range=255, sign=1)

        self.head = M.Conv2d(n_colors, self.n_feats, self.kernel_size, padding=(self.kernel_size // 2), bias=True)

        m_body = [
            ResidualBlock(
                self.n_feats, self.kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        self.body = M.Sequential(*m_body)
        self.body_conv = M.Conv2d(self.n_feats, self.n_feats, self.kernel_size,
                                  padding=(self.kernel_size // 2), bias=True)

        self.upsampler = Upsampler(scale, self.n_feats)
        self.tail_conv = M.Conv2d(self.n_feats, n_colors, self.kernel_size,
                                  padding=(self.kernel_size // 2), bias=True)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        residual = x
        for block in self.body:
            residual = block(residual)
        residual = self.body_conv(residual)
        residual += x

        x = self.upsampler(residual)
        x = self.tail_conv(x)
        x = self.add_mean(x)

        return x


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/99/files/a55d751f-2114-4806-b39b-43fd68144c58"
)
def csd_edsr_student():
    """EDSR in CSD for 4x SR. Student model."""
    return EDSR_S()
