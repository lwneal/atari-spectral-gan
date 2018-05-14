# Generate a video from a trained generator
import argparse
import torch
from torch.autograd import Variable
import model

import numpy as np
import os
import imutil


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--image_dir', type=str, default='input/')
parser.add_argument('--video_name', type=str, default='interpolation')
parser.add_argument('--latent_dim', type=int, default=4)
parser.add_argument('--generator_filename', type=str, required=True)

args = parser.parse_args()



def main():
    print('building model...')
    Z_dim = args.latent_dim

    generator = model.Generator(Z_dim).cuda()

    print('Loading model...')
    generator.load_state_dict(torch.load(args.generator_filename))
    print('Loaded model')

    output_video_name = args.video_name
    fixed_z = Variable(torch.randn(1, Z_dim).cuda())
    fixed_zprime = Variable(torch.randn(1, Z_dim).cuda())

    v = imutil.VideoMaker(output_video_name)
    for i in range(400):
        theta = abs(i - 200) / 200.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        samples = generator(z[0]).cpu().data.numpy()
        samples = samples.transpose((0,2,3,1))
        v.write_frame(samples)
    v.finish()


if __name__ == '__main__':
    main()
