# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)

        self.inner_down1 = down(256, 256)
        self.inner_down2 = down(256, 256)

        self.inner_up1 = up(512, 256)
        self.inner_up2 = up(512, 256)

        self.inner_first = inner(512, 256)
        self.inner_inner = inner(256, 256)
        self.inner_last = inner(256, 256)

        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.inner_first(x4)

        x6 = self.inner_down1(x5)
        x7 = self.inner_down2(x6)
        x8 = self.inner_inner(x7)

        x9 = self.inner_up1(x8, x6)
        x10 = self.inner_up2(x9, x5)

        x11 = self.inner_last(x10)

        x = self.up1(x11, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
