import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale):
        super(ResidualBlock, self).__init__()
        self.n_feats = n_feats
        self.res_scale = res_scale
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=1)

    def forward(self, x, width_mult=1):
        width = int(self.n_feats * width_mult)
        weight = self.conv1.weight[:width, :width, :, :]
        bias = self.conv1.bias[:width]
        residual = nn.functional.conv2d(x, weight, bias, padding=(self.kernel_size // 2))
        residual = self.act(residual)
        weight = self.conv2.weight[:width, :width, :, :]
        bias = self.conv2.bias[:width]
        residual = nn.functional.conv2d(residual, weight, bias, padding=(self.kernel_size // 2))

        return x + residual.mul(self.res_scale)


class Upsampler(nn.Sequential):
    def __init__(self, scale_factor, nf):
        super(Upsampler, self).__init__()
        block = []
        self.nf = nf
        self.scale = scale_factor

        if scale_factor == 3:
            block += [
                nn.Conv2d(nf, nf * 9, 3, padding=1, bias=True)
            ]
            self.pixel_shuffle = nn.PixelShuffle(3)
        else:
            self.block_num = scale_factor // 2
            self.pixel_shuffle = nn.PixelShuffle(2)

            for _ in range(self.block_num):
                block += [
                    nn.Conv2d(nf, nf * (2 ** 2), 3, padding=1, bias=True)
                ]
        self.blocks = nn.ModuleList(block)

    def forward(self, x, width_mult=1):
        res = x
        nf = self.nf
        if self.scale == 3:
            width = int(width_mult * nf)
            width9 = width * 9
            for block in self.blocks:
                weight = block.weight[:width9, :width, :, :]
                bias = block.bias[:width9]
                res = nn.functional.conv2d(res, weight, bias, padding=1)
                res = self.pixel_shuffle(res)
        else:
            for block in self.blocks:
                width = int(width_mult * nf)
                width4 = width * 4
                weight = block.weight[:width4, :width, :, :]
                bias = block.bias[:width4]
                res = nn.functional.conv2d(res, weight, bias, padding=1)
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
        bias = module.bias[:out_ch]
    return nn.functional.conv2d(input, weight, bias, stride=module.stride, padding=module.padding)


class EDSR(nn.Module):
    def __init__(self, n_colors=3, n_resblocks=32, n_feats=256, res_scale=0.1):
        super(EDSR, self).__init__()

        self.n_colors = n_colors
        n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.kernel_size = 3
        scale = 4
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        self.head = nn.Conv2d(n_colors, self.n_feats, self.kernel_size, padding=(self.kernel_size // 2), bias=True)

        m_body = [
            ResidualBlock(
                self.n_feats, self.kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.ModuleList(m_body)
        self.body_conv = nn.Conv2d(self.n_feats, self.n_feats, self.kernel_size,
                                   padding=(self.kernel_size // 2), bias=True)

        self.upsampler = Upsampler(scale, self.n_feats)
        self.tail_conv = nn.Conv2d(self.n_feats, n_colors, self.kernel_size,
                                   padding=(self.kernel_size // 2), bias=True)

    def forward(self, x, width_mult=1):
        feature_width = int(self.n_feats * width_mult)

        x = self.sub_mean(x)
        weight = self.head.weight[:feature_width, :self.n_colors, :, :]
        bias = self.head.bias[:feature_width]
        x = nn.functional.conv2d(x, weight, bias, padding=(self.kernel_size // 2))

        residual = x
        for block in self.body:
            residual = block(residual, width_mult)
        weight = self.body_conv.weight[:feature_width, :feature_width, :, :]
        bias = self.body_conv.bias[:feature_width]
        residual = nn.functional.conv2d(residual, weight, bias, padding=(self.kernel_size // 2))
        residual += x

        x = self.upsampler(residual, width_mult)
        weight = self.tail_conv.weight[:self.n_colors, :feature_width, :, :]
        bias = self.tail_conv.bias[:self.n_colors]
        x = nn.functional.conv2d(x, weight, bias, padding=(self.kernel_size // 2))
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


def csd_edsr_mix():
    '''Student and Teacher in one model'''
    return EDSR()
