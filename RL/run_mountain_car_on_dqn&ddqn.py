import sys

import gym
import numpy as np
import tensorflow as tf

sys.path.append('./')
from ddqn import DDQN
from memory import Memory

MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.9
NUM_EPISODE = 2000

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

n_actions = env.action_space.n
n_features = env.state.shape[0]
print('actions=', n_actions, 'n_features=', n_features)

memory0 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + 2, BATCH_SIZE)
memory1 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + 2, BATCH_SIZE)

sess0 = tf.Session()
sess1 = tf.Session()

dqn = [DDQN(n_actions,
            n_features,
            memory0,
            name='dqn0',
            learning_rate=LEARNING_RATE,
            reward_delay=GAMMA,
            replace_target_iter=200,
            double_q=False,
            sess=sess0),
       DDQN(n_actions,
            n_features,
            memory1,
            name='dqn1',
            learning_rate=LEARNING_RATE,
            reward_delay=GAMMA,
            replace_target_iter=200,
            double_q=False,
            sess=sess1)]

sess0.run(tf.global_variables_initializer())
sess1.run(tf.global_variables_initializer())

episode_reward = []
episode_position = []
episode_negative = []
total_step = [0, 0]
for episode in range(NUM_EPISODE):
    for i in range(len(dqn)):
        sum_reward = 0
        step = 0
        state = env.reset()
        while True:
            step += 1
            total_step[i] += 1

            # regular way for position and negative network
            action = dqn[0].choose_action(state)

            # another way
            # action_value = dqn[i].action_value(state)
            # action = np.argmax(action_value) if i == 0 else np.argmin(action_value)
            # if np.random.uniform() > dqn[i].epsilon:
            #     action = np.random.randint(0, n_actions)

            state_, reward, done, _ = env.step(action)

            # this condition considering the velocity
            if done:
                reward = 30
            else:
                reward = abs(state[0] - state_[0])

            # if done:
            #     reward = 30

            dqn[0].store_transtion(state, action, reward, state_)
            if reward != 30:
                dqn[1].store_transtion(state, action, -reward*100, state_)
            else:
                dqn[1].store_transition(state, action, -reward, state_)

            # this condition is to get a good result relatively
            # if step > 200 and step % 5 == 0:
            #     dqn[i].learn()

            # this condition is to get the best result
            if (total_step[i] > 200) and step % 5 == 0:
                dqn[i].learn()

            sum_reward += reward
            state = state_

            if done or step > 1999:
                print(dqn[0].name, episode, step)
                episode_position.append(step)
                dqn[0].set_replace_target_iter(5 * step)
                dqn[1].set_replace_target_iter(5 * step)
                break

    step = 0
    state = env.reset()
    while True:
        # env.render()
        step += 1

        action_values = dqn[0].action_value(state) - dqn[1].action_value(state)

        action = np.argmax(action_values)

        state_, reward, done, _ = env.step(action)

        state = state_

        if done or step > 1999:
            episode_reward.append(step)
            print('episode', episode, 'step', step)
            break

import matplotlib.pyplot as plt
plt.scatter(np.arange(len(episode_position)), episode_position, s=7, marker='o')
plt.scatter(np.arange(len(episode_reward)), episode_reward, s=7, marker='o')
plt.show()
