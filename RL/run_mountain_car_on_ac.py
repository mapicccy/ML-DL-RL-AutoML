import tensorflow as tf
import numpy as np
import gym
import sys

sys.path.append('.')
from actor_critic import Actor, Critic

np.random.seed(2)
tf.set_random_seed(2)

MAX_EPISODE = 1000
MAX_EP_STEPS = 2000
LR_A = 0.001
LR_C = 0.01

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n

sess0 = tf.Session()
sess1 = tf.Session()
rl = [[Actor(sess0, n_features, n_actions, name='actor0', lr=LR_A),
       Critic(sess0, n_features, name='critic0', lr=LR_C)],
      [Actor(sess1, n_features, n_actions, name='actor1', lr=LR_A),
       Critic(sess1, n_features, name='critic1', lr=LR_C)]]
sess0.run(tf.global_variables_initializer())
sess1.run(tf.global_variables_initializer())

episode_positive = []
episode_negative = []
episode_mix = []
for episode in range(MAX_EPISODE):
    for i in range(len(rl)):
        step = 0
        state = env.reset()
        while True:
            action = rl[i][0].choose_action(state)
            state_, reward, done, _ = env.step(action)

            if done:
                reward = 30
            else:
                reward = abs(state[0] - state_[0])

            new_reward = reward if i == 0 else -reward
            # if new_reward < 0 and new_reward != -30:
            #     new_reward *= 100

            td_error = rl[i][1].learn(state, new_reward, state_)
            rl[i][0].learn(state, action, td_error)

            state = state_
            step += 1

            if done or step > MAX_EP_STEPS:
                if i == 0:
                    episode_positive.append(step)
                else:
                    episode_negative.append(step)
                print(i, episode, 'step=', step)
                break

    step = 0
    state = env.reset()
    while True:
        action_values = rl[0][0].action_probs(state) - rl[1][0].action_probs(state)
        action_values[action_values < 0] = 0
        action_values = action_values/action_values.sum()
        action = np.random.choice(np.arange(len(action_values)), p=action_values)
        state_, reward, done, _ = env.step(action)

        step += 1
        state = state_

        if done or step > MAX_EP_STEPS:
            episode_mix.append(step)
            print('------------------- episode=', episode, 'step=', step)
            break

import matplotlib.pyplot as plt

plt.scatter(np.arange(len(episode_positive)), episode_positive, s=3, marker='o')
plt.scatter(np.arange(len(episode_negative)), episode_negative, s=3, marker='o')
plt.scatter(np.arange(len(episode_mix)), episode_mix, s=3, marker='o')
plt.show()
