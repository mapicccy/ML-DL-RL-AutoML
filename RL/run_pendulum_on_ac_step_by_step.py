import tensorflow as tf
import numpy as np
import gym
import sys

sys.path.append('.')
from actor_critic import Actor, Critic

np.random.seed(1)
tf.set_random_seed(1)

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

MAX_EPISODE = 2000
n_features = 3
action_space = 401

sess = tf.Session()
actor = Actor(sess, n_features, action_space, lr=0.001)
critic = Critic(sess, n_features, lr=0.01)
sess.run(tf.global_variables_initializer())

episode_reward = []
for episode in range(MAX_EPISODE):
    step = 0
    sum_reward = 0
    state = env.reset()
    while True:
        action = actor.choose_action(state)
        f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
        state_, reward, done, _ = env.step(np.array([f_action]))

        reward /= 10
        sum_reward += reward
        step += 1

        if step > 2000:
            episode_reward.append(sum_reward)
            print(episode, sum_reward)
            break

import matplotlib.pyplot as plt
plt.scatter(np.arange(len(episode_reward)), episode_reward, s=2, marker='o')
plt.show()