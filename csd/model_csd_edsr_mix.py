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
        self.n_feats = n_feats
        self.res_scale = res_scale
        self.kernel_size = kernel_size

        self.conv1 = M.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=1)
        self.act = act
        self.conv2 = M.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=1)

    def forward(self, x, width_mult=1):
        width = int(self.n_feats * width_mult)
        weight = self.conv1.weight[:width, :width, :, :]
        bias = self.conv1.bias[:, :width, :, :]
        residual = F.conv2d(x, weight, bias, padding=(self.kernel_size // 2))
        residual = self.act(residual)
        weight = self.conv2.weight[:width, :width, :, :]
        bias = self.conv2.bias[:, :width, :, :]
        residual = F.conv2d(residual, weight, bias, padding=(self.kernel_size // 2))

        return x + residual * self.res_scale


class Upsampler(M.Sequential):
    def __init__(self, scale_factor, nf):
        super(Upsampler, self).__init__()
        block = []
        self.nf = nf
        self.scale = scale_factor

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

    def forward(self, x, width_mult=1):
        res = x
        nf = self.nf
        if self.scale == 3:
            width = int(width_mult * nf)
            width9 = width * 9
            for block in self.blocks:
                weight = block.weight[:width9, :width, :, :]
                bias = block.bias[:, :width9, :, :]
                res = F.conv2d(res, weight, bias, padding=1)
                res = self.pixel_shuffle(res)
        else:
            for block in self.blocks:
                width = int(width_mult * nf)
                width4 = width * 4
                weight = block.weight[:width4, :width, :, :]
                bias = block.bias[:, :width4, :, :]
                res = F.conv2d(res, weight, bias, padding=1)
                res = self.pixel_shuffle(res)

        return res


def SlimModule(input, module, width_mult):
    weight = module.weight
    out_ch, in_ch = weight.shape[:2]
    out_ch = int(out_ch * width_mult)
    in_ch = int(in_ch * width_mult)
    weight = weight[:out_ch, :in_ch, :, :]
    bias = module.bias
    if bias is not None:
        bias = module.bias[:, :out_ch, :, :]
    return F.conv2d(input, weight, bias, stride=module.stride, padding=module.padding)


class EDSR(M.Module):
    def __init__(self, n_colors=3, n_resblocks=32, n_feats=256, res_scale=0.1):
        super(EDSR, self).__init__()

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

    def forward(self, x, width_mult=1):
        feature_width = int(self.n_feats * width_mult)

        x = self.sub_mean(x)
        weight = self.head.weight[:feature_width, :self.n_colors, :, :]
        bias = self.head.bias[:, :feature_width, :, :]
        x = F.conv2d(x, weight, bias, padding=(self.kernel_size // 2))

        residual = x
        for block in self.body:
            residual = block(residual, width_mult)
        weight = self.body_conv.weight[:feature_width, :feature_width, :, :]
        bias = self.body_conv.bias[:, :feature_width, :, :]
        residual = F.conv2d(residual, weight, bias, padding=(self.kernel_size // 2))
        residual += x

        x = self.upsampler(residual, width_mult)
        weight = self.tail_conv.weight[:self.n_colors, :feature_width, :, :]
        bias = self.tail_conv.bias[:, :self.n_colors, :, :]
        x = F.conv2d(x, weight, bias, padding=(self.kernel_size // 2))
        x = self.add_mean(x)

        return x


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/99/files/f8686584-0007-4955-a9fe-9d8cd5456768"
)
def csd_edsr_mix():
    """EDSR in CSD for 4x SR. A mix model includes Teacher and Student."""
    return EDSR()
