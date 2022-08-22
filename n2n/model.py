import megengine
import megengine.module as M
import megengine.functional as F
import megengine.hub as hub


class UpsampleCat(M.Module):
    def __init__(self, in_nc, out_nc):
        super(UpsampleCat, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc

        self.deconv = M.ConvTranspose2d(in_nc, out_nc, 2, 2, 0)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        return F.concat([x1, x2], axis=1)


def conv_func(x, conv, blindspot):
    size = conv.kernel_size[0]
    if blindspot:
        assert (size % 2) == 1
    ofs = 0 if (not blindspot) else size // 2

    if ofs > 0:
        # (padding_left, padding_right, padding_top, padding_bottom)
        x = F.pad(x, ((0, 0), (0, 0), (ofs, 0), (0, 0)), mode='constant', constant_value=0)
    x = conv(x)
    if ofs > 0:
        x = x[:, :, :-ofs, :]
    return x


def pool_func(x, pool, blindspot):
    if blindspot:
        x = F.pad(x[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)), mode='constant', constant_value=0)
    x = pool(x)
    return x


def rot90(x, k):
    if k == 1:
        return x[:, :, ::-1, :].tranpose(0, 1, 3, 2)
    elif k == 2:
        return x[:, :, ::-1, ::-1]
    elif k == 3:
        return x[:, :, :, ::-1].transpose(0, 1, 3, 2)


def rotate(x, angle):
    if angle == 0:
        return x
    elif angle == 90:
        return rot90(x, k=1)
    elif angle == 180:
        return rot90(x, k=2)
    elif angle == 270:
        return rot90(x, k=3)


class UNet(M.Module):
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 n_feature=48,
                 blindspot=False):
        super(UNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_feature = n_feature
        self.blindspot = blindspot
        self.act = M.LeakyReLU(negative_slope=0.2)

        # Encoder part
        self.enc_conv0 = M.Conv2d(self.in_nc, self.n_feature, 3, 1, 1)
        self.enc_conv1 = M.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool1 = M.MaxPool2d(2)

        self.enc_conv2 = M.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool2 = M.MaxPool2d(2)

        self.enc_conv3 = M.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool3 = M.MaxPool2d(2)

        self.enc_conv4 = M.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool4 = M.MaxPool2d(2)

        self.enc_conv5 = M.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool5 = M.MaxPool2d(2)

        self.enc_conv6 = M.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)

        # Decoder part
        self.up5 = UpsampleCat(self.n_feature, self.n_feature)
        self.dec_conv5a = M.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1, 1)
        self.dec_conv5b = M.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1, 1)

        self.up4 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv4a = M.Conv2d(self.n_feature * 3, self.n_feature * 2, 3, 1, 1)
        self.dec_conv4b = M.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1, 1)

        self.up3 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv3a = M.Conv2d(self.n_feature * 3, self.n_feature * 2, 3, 1, 1)
        self.dec_conv3b = M.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1, 1)

        self.up2 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv2a = M.Conv2d(self.n_feature * 3, self.n_feature * 2, 3, 1, 1)
        self.dec_conv2b = M.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1, 1)

        self.up1 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)

        # Output stages
        self.dec_conv1a = M.Conv2d(self.n_feature * 2 + self.in_nc, 96, 3, 1, 1)
        self.dec_conv1b = M.Conv2d(96, 96, 3, 1, 1)
        if blindspot:
            self.nin_a = M.Conv2d(96 * 4, 96 * 4, 1, 1, 0)
            self.nin_b = M.Conv2d(96 * 4, 96, 1, 1, 0)
        else:
            self.nin_a = M.Conv2d(96, 96, 1, 1, 0)
            self.nin_b = M.Conv2d(96, 96, 1, 1, 0)
        self.nin_c = M.Conv2d(96, self.out_nc, 1, 1, 0)

    def forward(self, x):
        # Input stage
        blindspot = self.blindspot
        if blindspot:
            x = F.concat([rotate(x, a) for a in [0, 90, 180, 270]], dim=0)
        # Encoder part
        pool0 = x
        x = self.act(conv_func(x, self.enc_conv0, blindspot))
        x = self.act(conv_func(x, self.enc_conv1, blindspot))
        x = pool_func(x, self.pool1, blindspot)
        pool1 = x

        x = self.act(conv_func(x, self.enc_conv2, blindspot))
        x = pool_func(x, self.pool2, blindspot)
        pool2 = x

        x = self.act(conv_func(x, self.enc_conv3, blindspot))
        x = pool_func(x, self.pool3, blindspot)
        pool3 = x

        x = self.act(conv_func(x, self.enc_conv4, blindspot))
        x = pool_func(x, self.pool4, blindspot)
        pool4 = x

        x = self.act(conv_func(x, self.enc_conv5, blindspot))
        x = pool_func(x, self.pool5, blindspot)

        x = self.act(conv_func(x, self.enc_conv6, blindspot))

        # Decoder part
        x = self.up5(x, pool4)
        x = self.act(conv_func(x, self.dec_conv5a, blindspot))
        x = self.act(conv_func(x, self.dec_conv5b, blindspot))

        x = self.up4(x, pool3)
        x = self.act(conv_func(x, self.dec_conv4a, blindspot))
        x = self.act(conv_func(x, self.dec_conv4b, blindspot))

        x = self.up3(x, pool2)
        x = self.act(conv_func(x, self.dec_conv3a, blindspot))
        x = self.act(conv_func(x, self.dec_conv3b, blindspot))

        x = self.up2(x, pool1)
        x = self.act(conv_func(x, self.dec_conv2a, blindspot))
        x = self.act(conv_func(x, self.dec_conv2b, blindspot))

        x = self.up1(x, pool0)

        # Output stage
        if blindspot:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            x = F.pad(x[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)), mode='constant', constant_value=0)
            x = F.split(x, nsplits_or_sections=x.shape[0] // 4, dim=0)
            x = [rotate(y, a) for y, a in zip(x, [0, 270, 180, 90])]
            x = F.concat(x, dim=1)
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        else:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        return x


def neighbor2neighbor():
    return UNet(in_nc=3, out_nc=3, n_feature=48)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/101/files/21e35c5c-6f3d-44fd-9aa0-b6bc233b7504"
)
def n2n_gauss25():
    """Neighbor2neighbor model for gaussian noise (sigma = 25)"""
    return UNet(3, 3, 48, False)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/101/files/4a4799ea-9406-45d9-98dc-deb5a56b9672"
)
def n2n_gauss5to50():
    """Neighbor2neighbor model for gaussian noise (sigma ranges from 5 to 50)"""
    return UNet(3, 3, 48, False)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/101/files/84d89846-b95e-44a2-92fb-5c719d0d109a"
)
def n2n_poisson30():
    """Neighbor2neighbor model for poisson noise (lambda = 30)"""
    return UNet(3, 3, 48, False)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/101/files/13663d09-2f88-4f5f-9d52-4a0108d28623"
)
def n2n_poisson5to50():
    """Neighbor2neighbor model for poisson noise (lambda ranges from 5 to 50)"""
    return UNet(3, 3, 48, False)


if __name__ == "__main__":
    import numpy as np
    x = megengine.tensor(np.zeros((10, 3, 32, 32), dtype=np.float32))
    print(x.shape)
    net = UNet(in_nc=3, out_nc=3, blindspot=False)
    y = net(x)
    print(y.shape)
