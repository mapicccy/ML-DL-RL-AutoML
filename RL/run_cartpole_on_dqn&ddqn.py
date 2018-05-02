import sys
import gym
import numpy as np
import tensorflow as tf

sys.path.append('./')

from ddqn import DDQN
from memory import Memory

np.random.seed(1)
tf.set_random_seed(1)

NUM_EPISODE = 3000
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.005

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)
env.reset()

n_features = env.state.shape[0]
n_actions = env.action_space.n

print(n_features, n_actions)

memory0 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + 2, BATCH_SIZE)
memory1 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + 2, BATCH_SIZE)

sess0 = tf.Session()
sess1 = tf.Session()
dqn = [DDQN(n_actions,
            n_features,
            memory0,
            name='dqn0',
            learning_rate=LEARNING_RATE,
            e_greedy_increment=0.001,
            double_q=False,
            sess=sess0),
       DDQN(n_actions,
            n_features,
            memory1,
            name='dqn1',
            learning_rate=LEARNING_RATE,
            e_greedy_increment=0.001,
            double_q=True,
            sess=sess1)]

sess0.run(tf.global_variables_initializer())
sess1.run(tf.global_variables_initializer())