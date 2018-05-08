import gym
import numpy as np
import tensorflow as tf
import sys

sys.path.append('.')
from memory import Memory
from ddqn import DDQN

MEMORY_CAPACITY = 100000
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

sess0 = tf.Session()
sess1 = tf.Session()
dqn = [DDQN(action_space,
            n_features,
            memory0,
            name='dqn0',
            learning_rate=LEARNING_RATE,
            e_greedy_increment=0.001,
            double_q=False,
            sess=sess0),
       DDQN(action_space,
            n_features,
            memory1,
            name='dqn1',
            learning_rate=LEARNING_RATE,
            e_greedy_increment=0.001,
            double_q=False,
            sess=sess1)]

sess0.run(tf.global_variables_initializer())
sess1.run(tf.global_variables_initializer())

episode_position = []
episode_negative = []
episode_mix = []
total_step = [0, 0]
for episode in range(NUM_EPISODE):
    for i in range(len(dqn)):
        step = 0
        state = env.reset()
        sum_reward = 0
        while True:
            # env.render()
            action = dqn[i].choose_action(state)
            f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
            state_, reward, done, info = env.step(np.array([f_action]))

            reward /= 10
            sum_reward += reward

            new_reward = reward if i == 0 else -reward
            dqn[i].store_transition(state, action, new_reward, state_)

            if total_step[i] > 200:
                dqn[i].learn()

            state = state_
            step += 1
            total_step[i] += 1

            if step > 2000:
                if i == 0:
                    episode_position.append(sum_reward)
                    print(dqn[i].name, episode, sum_reward)
                break

    sum_reward = 0
    step = 0
    state = env.reset()
    while True:
        action_values = dqn[1].action_value(state)
        action = np.argmin(action_values)
        f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
        state_, reward, done, info = env.step(np.array([f_action]))
        reward /= 10
        sum_reward += reward
        state = state_
        step += 1
        if step > 2000:
            episode_negative.append(sum_reward)
            break

    sum_reward = 0
    step = 0
    state = env.reset()
    while True:
        action_values = dqn[0].action_value(state) - dqn[1].action_value(state)
        action = np.argmax(action_values)
        f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
        state_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10
        sum_reward += reward
        state = state_
        step += 1

        if step > 2000:
            episode_mix.append(sum_reward)
            print('---------episode', episode, 'sum_reward', sum_reward)
            break

import matplotlib.pyplot as plt
plt.scatter(np.arange(len(episode_position)), episode_position, s=7, marker='o')
plt.scatter(np.arange(len(episode_negative)), episode_negative, s=7, marker='o')
plt.scatter(np.arange(len(episode_mix)), episode_mix, s=7, marker='o')
plt.show()