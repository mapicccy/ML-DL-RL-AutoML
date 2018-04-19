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

MAX_EPISODE = 600
n_features = 3
action_space = 401

sess0 = tf.Session()
sess1 = tf.Session()
rl = [[Actor(sess0, n_features, action_space, name='actor0', lr=0.0001),
       Critic(sess0, n_features, name='critic0', lr=0.001)],
      [Actor(sess1, n_features, action_space, name='actor1', lr=0.0001),
       Critic(sess1, n_features, name='critic1', lr=0.001)]]
sess0.run(tf.global_variables_initializer())
sess1.run(tf.global_variables_initializer())

episode_positive = []
episode_negative = []
episode_mix = []
for episode in range(MAX_EPISODE):
    for i in range(len(rl)):
        step = 0
        sum_reward = 0
        state = env.reset()
        while True:
            action = rl[i][0].choose_action(state)
            f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
            state_, reward, done, _ = env.step(np.array([f_action]))

            reward /= 10
            sum_reward += reward
            step += 1

            new_reward = reward if i == 0 else -reward
            td_error = rl[i][1].learn(state, new_reward, state_)
            rl[i][0].learn(state, action, td_error)

            state = state_

            if step > 2000:
                if i == 0:
                    episode_positive.append(sum_reward)
                else:
                    episode_negative.append(sum_reward)
                print(i, episode, sum_reward)
                break

    step = 0
    sum_reward = 0
    state = env.reset()
    while True:
        action_values = rl[0][0].action_probs(state) - rl[1][0].action_probs(state)
        action_values[action_values < 0] = 0
        # action_values[action_values > 1] = 1
        action_values = action_values/action_values.sum()
        action = np.random.choice(np.arange(len(action_values)), p=action_values)
        f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
        state_, reward, done, _ = env.step(np.array([f_action]))

        reward /= 10
        sum_reward += reward
        step += 1
        state = state_

        if step > 2000:
            episode_mix.append(sum_reward)
            print('-------- episode_mix: reward=', sum_reward)
            break

import matplotlib.pyplot as plt
plt.scatter(np.arange(len(episode_positive)), episode_positive, s=2, marker='o')
plt.scatter(np.arange(len(episode_negative)), episode_negative, s=2, marker='o')
plt.scatter(np.arange(len(episode_mix)), episode_mix, s=2, marker='o')
plt.show()