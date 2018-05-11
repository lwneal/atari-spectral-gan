import argparse
import numpy as np
import os
import random
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch import autograd
from torch.autograd import Variable

import model
from atari_dataloader import AtariDataloader
from series import TimeSeries
import imutil


print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--latent_size', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)

args = parser.parse_args()
Z_dim = args.latent_size


loader = AtariDataloader(name='Pong-v0', batch_size=args.batch_size)


print('Building model...')

discriminator = model.Discriminator().cuda()
generator = model.Generator(Z_dim).cuda()
encoder = model.Encoder(Z_dim).cuda()

if args.start_epoch:
    generator.load_state_dict(torch.load('checkpoints/gen_{}'.format(args.start_epoch)))
    encoder.load_state_dict(torch.load('checkpoints/enc_{}'.format(args.start_epoch)))
    discriminator.load_state_dict(torch.load('checkpoints/disc_{}'.format(args.start_epoch)))

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
print('Building optimizers')
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_e = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)
print('Finished building model')


def sample_z(batch_size, z_dim):
    # Normal Distribution
    z = torch.randn(batch_size, z_dim)
    return Variable(z.cuda())


def train(epoch, max_batches=100, disc_iters=5):
    datasource = islice(loader, max_batches)
    ts = TimeSeries('Wasserstein GAN', max_batches)

    for batch_idx, (data, target) in enumerate(datasource):
        data = Variable(data.cuda())

        # update discriminator
        for _ in range(disc_iters):
            z = sample_z(args.batch_size, Z_dim)
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            disc_loss.backward()
            optim_disc.step()

        optim_enc.zero_grad()
        optim_gen.zero_grad()

        # reconstruct images
        reconstructed = generator(encoder(data))
        aac_loss = torch.mean((reconstructed - data)**2)
        aac_loss.backward()

        # update generator
        z = sample_z(args.batch_size, Z_dim)
        gen_loss = -discriminator(generator(z)).mean()
        gen_loss.backward()

        optim_enc.step()
        optim_gen.step()

        ts.collect('Disc Loss', disc_loss)
        ts.collect('Gen Loss', gen_loss)
        ts.collect('L2 Loss', aac_loss)

        ts.print_every(n_sec=4)

    scheduler_e.step()
    scheduler_d.step()
    scheduler_g.step()
    print(ts)


fixed_z = sample_z(args.batch_size, Z_dim)
def evaluate(epoch):
    samples = generator(fixed_z).cpu().data.numpy()[:64]

    # TODO: Autoencode a few images, check how good they are

    # TODO: Generate a few images, check out good they are

    # TODO: For adjacent frames, compute their distance in the latent space

    print('TODO: evaluate')
    return {
        'foo_score': 1,
        'sample_avg': samples.mean(),
    }


fixed_z = Variable(torch.randn(1, Z_dim).cuda())
fixed_zprime = Variable(torch.randn(1, Z_dim).cuda())
def make_video(output_video_name):
    v = imutil.VideoMaker(output_video_name)
    for i in range(100):
        theta = abs(i - 50) / 50.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        samples = generator(z).cpu().data.numpy()
        pixels = samples.transpose((0,2,3,1)) * 0.5 + 0.5
        v.write_frame(pixels)
    v.finish()


def main():
    print('creating checkpoint directory')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ts = TimeSeries('Evaluation', args.epochs)
    for epoch in range(args.epochs):
        print('starting epoch {}'.format(epoch))
        metrics = evaluate(epoch)
        for key, value in metrics.items():
            ts.collect(key, value)
        print(ts)
        make_video('epoch_{:03d}'.format(epoch))
        train(epoch)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
        torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, 'enc_{}'.format(epoch)))
    # TODO: Generate graphs from evaluation metrics
    print(ts)


if __name__ == '__main__':
    main()
