# Generate some frames of Pong
import random
import sys
import numpy as np
import gym
from scipy.misc import imresize
from imutil import show

import torch


class AtariDataloader():
    def __init__(self, name, batch_size):
        self.environments = []
        for i in range(batch_size):
            env = gym.make(name)
            env.reset()
            # Start each environment at a random time
            for _ in range(random.randint(1, 100)):
                env.step(env.action_space.sample())
            self.environments.append(env)

    def __iter__(self):
        return self

    def __next__(self):
        observations = []
        positions = []
        for env in self.environments:
            while True:
                obs, r, done, info = env.step(env.action_space.sample())
                if done:
                    env.reset()
                pixels = get_pixels(obs)
                ly, ry, bx, by = find_positions(pixels)
                if bx or by:
                    break
            pixels = (pixels - 128) / 128

            # Output batch x channels x height x width
            pixels = pixels.transpose((2,0,1))
            observations.append(pixels)
            positions.append((ly, ry, bx, by))
        # Standard API: next() returns two tensors (x, y)
        return torch.Tensor(np.array(observations)), positions


def get_pixels(obs):
    pixels = obs[34:194]
    pixels = pixels[::2,::2]
    return pixels


def find_positions(pixels):
    ball = (pixels[:,:,2] == 236)
    ball_x = ball.argmax(axis=0).max()
    ball_y = ball.argmax(axis=1).max()

    left_paddle_pos = np.argmax(pixels[:,8,0])
    right_paddle_pos = np.argmax(pixels[:,71,1] == 186)

    return left_paddle_pos, right_paddle_pos, ball_x, ball_y
