import gym
import numpy as np
import tensorflow as tf
import sys

sys.path.append('.')
from memory import Memory
from ddqn import DDQN

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

action_space = 11
n_features = 3
memory = Memory(n_features, 3000, n_features * 2 + 2, 32)

sess = tf.Session()
with tf.variable_scope('dqn'):
    dqn = DDQN(n_actions=action_space,
               n_features=n_features,
               memory=memory,
               e_greedy_increment=0.001,
               double_q=False,
               sess=sess)

sess.run(tf.global_variables_initializer())


def train(RL):
    step = 0
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)
        f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10

        RL.store_transition(observation, action, reward, observation_)

        if step > 200:
            RL.learn()

        if step > 20000:
            break

        observation = observation_
        step += 1

    return RL.q

q_dqn = train(dqn)
