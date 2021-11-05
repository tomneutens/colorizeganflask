import torch
import torch.nn as nn
import torch.nn.functional as F

from gan.image_pool import ImagePool
from unet.unet_model import UNet

import utils

class GANModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.GAN_enabled = True

        self.lambda_L1 = 100
        self.lambda_grad = 25
        self.lambda_GAN = 1

        self.pool_size = 50
        self.beta1 = 0.5
        self.lr = 0.0001

        self.netG = UNet(1, 3)
        self.netD_all = nn.ModuleList([PatchDiscriminator(scale) for scale in [0.5, 0.25, 0.125]]) if self.GAN_enabled else []

        self.losses_D = [None for _ in self.netD_all]
        self.losses_G_GAN = [None for _ in self.netD_all]

        self.fake_image_pool = ImagePool(self.pool_size)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(sum([list(netD.parameters()) for netD in self.netD_all], []), lr=self.lr, betas=(self.beta1, 0.999)) if self.GAN_enabled else None

        # Define the loss functions
        self.criterionGAN = GANLoss()
        self.criterionL1 = nn.L1Loss()

    def forward(self):
        """ Generate a fake image for the current line art """
        self.fake_image = self.netG(self.line_art)

    def backward_D(self):
        assert self.GAN_enabled

        fake_image = self.fake_image_pool.query(self.fake_image)

        self.loss_D = 0

        for i, netD in enumerate(self.netD_all):
            # Fake
            # https://pytorch.org/blog/pytorch-0_4_0-migration-guide/
            pred_fake = netD(fake_image.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)

            # Real
            pred_real = netD(self.real_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # Combined loss
            self.losses_D[i] = (loss_D_fake + loss_D_real) * 0.5
            self.loss_D += self.losses_D[i]

        self.loss_D /= len(self.netD_all)
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G_GAN = 0

        if self.GAN_enabled:
            for i, netD in enumerate(self.netD_all):
                pred_fake = netD(self.fake_image)
                self.losses_G_GAN[i] = self.criterionGAN(pred_fake, True)
                self.loss_G_GAN += self.losses_G_GAN[i]

            self.loss_G_GAN /= len(self.netD_all)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_image, self.real_image)

        self.loss_G_grad = utils.image_gradient_loss(self.fake_image, self.fake_image.device, self.line_art)
        #self.loss_G_grad = torch.tensor(0, device=self.fake_image.device, dtype=self.fake_image.dtype)

        self.loss_G = self.lambda_L1 * self.loss_G_L1 \
                    + self.lambda_GAN * self.loss_G_GAN \
                    + self.lambda_grad * self.loss_G_grad

        self.loss_G.backward()

    def optimize_parameters(self, line_art, real_image):
        self.line_art = line_art
        self.real_image = real_image

        self.forward()

        if self.GAN_enabled:
            # update D
            for netD in self.netD_all:
                self.set_requires_grad(netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update G
        for netD in self.netD_all:
            self.set_requires_grad(netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        return {
            'L1-Loss': self.loss_G_L1.item(),
            'GAN-Loss': self.loss_G_GAN.item() if self.GAN_enabled else 0,
            'Grad-Loss': self.loss_G_grad.item(),
            'Total-Loss': self.loss_G.item(),
            'Disc-Loss': self.loss_D.item() if self.GAN_enabled else 0,
            ** {('Disc-Loss-' + str(netD.scale_factor)): loss for netD, loss in zip(self.netD_all, self.losses_D)},
            ** {('GAN-Loss-' + str(netD.scale_factor)): loss for netD, loss in zip(self.netD_all, self.losses_G_GAN)}
        }

    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad


class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def __call__(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return self.loss(input, target_tensor.expand_as(input))


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc):
        super().__init__()

        ndf = 128

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(ndf * 2),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)
            # No Sigmoid since we're using LSGAN
        )

    def forward(self, input):
        downsampled = F.interpolate(input, scale_factor=0.25, mode='bilinear', align_corners=False)
        return self.net(downsampled)


class PatchDiscriminator(nn.Module):

    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(64, affine=True),

            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(128, affine=True),

            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(128, affine=True),

            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(128, affine=True),

            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(128, affine=True),

            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, input):
        downsampled = F.interpolate(input, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.net(downsampled)

# TODO: Play with architecture of discriminator.. Maybe opt for a multi-level one eventually?
# TODO: Optimize ImagePool.. Maybe detach when saving instead of when retrieving the element?
# TODO: How do optimizer betas affect the result?
# TODO: Incorporate GAN training tips (see GitHub)
