# Generate a video from a trained generator
import argparse
import torch
from torch.autograd import Variable
import model

import numpy as np
import os
import imutil


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--image_dir', type=str, default='input/')
parser.add_argument('--video_name', type=str, default='sample')
parser.add_argument('--latent_dim', type=int, default=4)

args = parser.parse_args()

GENERATOR_FILENAME = 'checkpoints/gen_34'

print('building model...')
Z_dim = args.latent_dim

generator = model.Generator(Z_dim).cuda()

print('Loading model...')
generator.load_state_dict(torch.load(GENERATOR_FILENAME))
print('Loaded model')

output_video_name = args.video_name
output_count = 1
fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
fixed_zprime = Variable(torch.randn(args.batch_size, Z_dim).cuda())
def evaluate():
    v = imutil.VideoMaker(output_video_name)
    for i in range(400):
        theta = abs(i - 200) / 200.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        print(z.cpu().data)
        samples = generator(z[output_count]).cpu().data.numpy()
        samples = samples.transpose((0,2,3,1))
        v.write_frame(samples)
    v.finish()


def main():
    print('creating checkpoint directory')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    evaluate()


if __name__ == '__main__':
    main()
