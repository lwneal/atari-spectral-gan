# DCGAN-like generator and discriminator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from spectral_normalization import SpectralNorm

channels = 3
leak = 0.1


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        # 80 x 80
        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))
        # 80 x 80
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        # 40 x 40
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        # 40 x 40
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        # 20 x 20
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        # 20 x 20
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        # 10 x 10
        self.conv7 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        # 5 x 5
        self.conv8 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(0,0)))
        # 3 x 3

        self.fc = SpectralNorm(nn.Linear(3 * 3 * 512, latent_size))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m = nn.LeakyReLU(leak)(self.conv8(m))

        return self.fc(m.view(-1, 3 * 3 * 512))


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.ConvTranspose2d(z_dim, 512, 4, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(0,0)) # 10
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)) # 20
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=(1,1)) # 40
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1))  # 80
        self.bn5 = nn.BatchNorm2d(64)

        self.conv_to_rgb = nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1))

    def forward(self, z):
        batch_size = z.shape[0]
        x = z.view(-1, self.z_dim, 1, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv_to_rgb(x)
        x = F.sigmoid(x)
        return x

# What is w_g supposed to be?
w_g = 3
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 80 x 80
        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))
        # 80 x 80
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        # 40 x 40
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        # 40 x 40
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        # 20 x 20
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        # 20 x 20
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        # 10 x 10
        self.conv7 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        # 5 x 5
        self.conv8 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(0,0)))
        # 3 x 3

        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m = nn.LeakyReLU(leak)(self.conv8(m))

        return self.fc(m.view(-1,w_g * w_g * 512))

