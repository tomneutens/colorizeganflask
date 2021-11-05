# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Remove normalization from (last) upsampling layers?
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch, affine=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, stride=2),
            nn.ReLU(inplace=True),
            double_conv(out_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class inner(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inner, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_ch, in_ch, 3, dilation=2),  # conv5_1
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_ch, in_ch, 3, dilation=2),  # conv5_2
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_ch, out_ch, 3, dilation=2),  # conv5_3
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch, affine=True),
        )

    def forward(self, x):
        return self.seq(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = None
        else:
            assert False
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.up is None:
            x1 = nn.functional.interpolate(x1, scale_factor=2, mode='nearest')
        else:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2), 'reflect')

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
