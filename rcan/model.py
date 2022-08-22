import math
import megengine
import megengine.module as M
import megengine.functional as F
import megengine.hub as hub


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return M.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(M.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = megengine.tensor(rgb_std)
        self.weight = F.eye(3).reshape(3, 3, 1, 1)
        self.weight = self.weight / std.reshape(3, 1, 1, 1)
        self.bias = sign * rgb_range * megengine.tensor(rgb_mean)
        self.bias = self.bias
        self.bias = self.bias / std
        self.bias = self.bias.reshape(1, 3, 1, 1)


class Upsampler(M.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(M.PixelShuffle(2))
                if bn:
                    m.append(M.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(M.PixelShuffle(3))
            if bn:
                m.append(M.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CALayer(M.Module):  # Channel Attention (CA) Layer
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.conv_du = M.Sequential(
            M.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            M.ReLU(),
            M.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            M.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(M.Module):  # Residual Channel Attention Block (RCAB)
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=M.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(M.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = M.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualGroup(M.Module):  # Residual Group (RG)
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=M.ReLU(), res_scale=1)
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = M.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(M.Module):
    def __init__(self,
                 n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16,
                 scale=4, n_colors=3, rgb_range=255, conv=default_conv):
        super(RCAN, self).__init__()

        kernel_size = 3
        act = M.ReLU()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = M.Sequential(*modules_head)
        self.body = M.Sequential(*modules_body)
        self.tail = M.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/117/files/e012ec90-c4da-4bda-9e70-97819c7b2009"
)
def rcan_x2():
    """RCAN model for 2x SR."""
    return RCAN(scale=2)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/117/files/4f0ccaed-c00a-45e3-aae6-7188c3b16749"
)
def rcan_x3():
    """RCAN model for 3x SR."""
    return RCAN(scale=3)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/117/files/e0467e21-c78b-4222-80ac-4220d6eebefe"
)
def rcan_x4():
    """RCAN model for 4x SR."""
    return RCAN(scale=4)
