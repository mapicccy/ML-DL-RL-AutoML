import gym
import sys
from config import *


class Gym_instance:
    def __init__(self, name, seed=1, action_space=None, observation_space=None):
        self.name = name
        self.env = gym.make(name).unwrapped
        if seed is not None:
            self.env.seed(seed)

        self.env.reset()
        if self.name == 'Pendulum-v0':
            if action_space and observation_space:
                self.n_actions = action_space
                self.n_features = observation_space
                self.action_space_high = self.env.action_space.high
            else:
                print('not support "%d" action_space or "%d" observation_space' % (action_space, observation_space))
                sys.exit(2)
        else:
            self.n_actions = self.env.action_space.n
            self.n_features = self.env.state.shape[0]
