import gym
import numpy as np
import tensorflow as tf
import sys

sys.path.append('.')
from memory import Memory
from ddqn import DDQN

MEMORY_CAPACITY = 3000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.9
NUM_EPISODE = 2000

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

action_space = 11
n_features = 3
memory0 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + 2, BATCH_SIZE)
memory1 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + 2, BATCH_SIZE)

sess = tf.Session()
with tf.variable_scope('dqn'):
    dqn = DDQN(n_actions=action_space,
               n_features=n_features,
               memory=memory0,
               name='dqn',
               learning_rate=LEARNING_RATE,
               e_greedy_increment=0.001,
               double_q=False,
               sess=sess)

with tf.variable_scope('ddqn'):
    ddqn = DDQN(n_actions=action_space,
                n_features=n_features,
                memory=memory1,
                name='ddqn',
                learning_rate=LEARNING_RATE,
                e_greedy_increment=0.001,
                double_q=True,
                sess=sess)

sess.run(tf.global_variables_initializer())

episode_dqn = []
episode_ddqn = []
total_step = 0
for episode in range(NUM_EPISODE):
    step = 0
    observation = env.reset()
    RL = dqn if episode % 2 == 0 else ddqn
    sum_reward = 0
    while True:
        # env.render()
        action = RL.choose_action(observation)
        f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10
        sum_reward += reward

        RL.store_transition(observation, action, reward, observation_)

        if total_step > 200:
            RL.learn()

        if step > 2000:
            if episode % 2 == 0:
                episode_dqn.append(sum_reward)
            else:
                episode_ddqn.append(sum_reward)
            print(RL.name, episode, sum_reward)
            break

        observation = observation_
        step += 1
        total_step += 1

import matplotlib.pyplot as plt
plt.scatter(np.arange(len(episode_dqn)), episode_dqn, s=7, marker='o')
plt.scatter(np.arange(len(episode_ddqn)), episode_ddqn, s=7, marker='o')
plt.show()