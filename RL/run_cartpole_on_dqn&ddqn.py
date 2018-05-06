import sys
import gym
import numpy as np
import tensorflow as tf

sys.path.append('./')

from ddqn import DDQN
from memory import Memory

np.random.seed(1)
tf.set_random_seed(1)

NUM_EPISODE = 1000
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

episode_mix = []
episode_positive = []
episode_negative = []
for episode in range(NUM_EPISODE):
    for i in range(len(dqn)):
        step = 0
        sum_reward = 0
        state = env.reset()
        while True:
            action = dqn[i].choose_action(state)
            state_, reward, done, _ = env.step(action)
            if done:
                reward = -20

            new_reward = reward if i == 0 else -reward
            dqn[i].store_transition(state, action, new_reward, state_)
            sum_reward += reward
            state = state_
            if step % 5 == 0:
                dqn[i].learn()

            if done or step == 200:
                if i == 0:
                    episode_positive.append(step)
                    print(dqn[i].name, episode, 'step =', step)
                break
            step += 1

    step = 0
    state = env.reset()
    while True:
        action = np.argmin(dqn[1].action_value(state))
        state_, reward, done, _ = env.step(action)
        state = state_
        if done or step == 200:
            episode_negative.append(step)
            print(dqn[1].name, episode, 'step =', step)
            break
        step += 1

    step = 0
    state = env.reset()
    while True:
        action = np.argmax(dqn[0].action_value(state) - dqn[1].action_value(state))
        state_, reward, done, _ = env.step(action)
        state = state_
        if done or step == 200:
            episode_mix.append(step)
            print('---------------- mix: step =', step)
            break
        step += 1


import matplotlib.pyplot as plt
plt.scatter(np.arange(len(episode_positive)), episode_positive, s=3, marker='o')
plt.scatter(np.arange(len(episode_negative)), episode_negative, s=3, marker='o')
plt.scatter(np.arange(len(episode_mix)), episode_mix, s=3, marker='o')
plt.show()
