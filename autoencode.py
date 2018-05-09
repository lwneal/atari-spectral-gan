import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch import autograd
from torch.autograd import Variable
import model
from dataloader import CustomDataloader

import numpy as np
import os
import imutil


print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--latent_size', type=int, default=10)
parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--env_name', type=str, default='Pong-v0')

parser.add_argument('--start_epoch', type=int, required=True)

args = parser.parse_args()


loader = CustomDataloader('/mnt/data/mnist-fgsm.dataset', batch_size=args.batch_size, image_size=80)


print('Building model...')
Z_dim = args.latent_size
disc_iters = 5

discriminator = model.Discriminator().cuda()
generator = model.Generator(Z_dim).cuda()
encoder = model.Encoder(Z_dim).cuda()

generator.load_state_dict(torch.load('checkpoints/gen_{}'.format(args.start_epoch)))
encoder.load_state_dict(torch.load('checkpoints/enc_{}'.format(args.start_epoch)))
discriminator.load_state_dict(torch.load('checkpoints/disc_{}'.format(args.start_epoch)))


def main():
    x, _ = next(loader)
    imutil.show(x, filename='original.jpg')
    z = encoder(x)
    x_r = generator(z)
    imutil.show(x_r, filename='reconstructed.jpg')


if __name__ == '__main__':
    main()
