import megengine.module as M
import megengine.functional as F
import megengine.hub as hub


class ResidualDenseBlock_5C(M.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = M.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = M.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = M.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = M.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = M.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(F.concat((x, x1), 1)), 0.2)
        x3 = F.leaky_relu(self.conv3(F.concat((x, x1, x2), 1)), 0.2)
        x4 = F.leaky_relu(self.conv4(F.concat((x, x1, x2, x3), 1)), 0.2)
        x5 = self.conv5(F.concat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(M.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(M.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = M.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = M.Sequential(
            *[RRDB(nf, gc) for _ in range(nb)]
        )
        self.trunk_conv = M.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.upconv1 = M.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = M.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = M.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = M.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = M.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.vision.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.vision.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/100/files/238d2663-579a-4586-a5c0-ef66b2ddc462"
)
def rrdb_psnr():
    """PSNR oriented ESRGAN model for 4x SR."""
    return RRDBNet(3, 3, 64, 23, gc=32)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/100/files/cb6897d3-1e00-478e-985a-e7c028ec309e"
)
def rrdb_esrgan():
    """ESRGAN model for 4x SR."""
    return RRDBNet(3, 3, 64, 23, gc=32)
