import gym
import sys
import tensorflow as tf
import numpy as np

sys.path.append('./')
from ddqn import DDQN
from memory import Memory

MEMORY_CAPACITY = 3000
BATCH_SIZE = 32
LEARNING_RATE = 0.01
GAMMA = 0.9
NUM_EPISODE = 2000

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

n_actions = env.action_space.n
n_features = env.state.shape[0]
print('actions=', n_actions, 'n_features=', n_features)

memory0 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + 2, BATCH_SIZE)

sess = tf.Session()

dqn = DDQN(n_actions,
           n_features,
           memory0,
           name='ddqn0',
           learning_rate=LEARNING_RATE,
           reward_delay=GAMMA,
           replace_target_iter=200,
           double_q=False)

episode_reward = []
for i in range(NUM_EPISODE):
    sum_reward = 0
    step = 0
    state = env.reset()
    while True:
        step += 1
        action = dqn.choose_action(state)
        state_, reward, done, _ = env.step(action)

        if done:
            reward = 30

        dqn.store_transition(state, action, reward, state_)

        if step > 200 and step % 5 == 0:
            dqn.learn()

        sum_reward += reward
        state = state_

        if done or step > 2000:
            print(dqn.name, i, step)
            episode_reward.append(step)
            # dqn.set_replace_iter(step * 5)
            break

import matplotlib.pyplot as plt
plt.scatter(np.arange(len(episode_reward)), episode_reward, s=7, marker='o')
plt.show()