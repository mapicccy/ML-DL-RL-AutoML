import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ddqn_new import DoubleDQN
from config import *
from memory_new import Memory
import matplotlib.pyplot as plt

def volatility(r):
    l = len(r) - 1
    r = np.array(r)
    r0 = r[:l]
    r1 = r[1:]
    ratio = r1 / r0 - 1
    return np.std(ratio)

class Game(object):
    def __init__(self, env, rls, combine=None):
        self.env = env
        self.rls = rls
        self.reverse = False
        self.c = combine

    def mixed_choose_action(self, agent0, agent1, state, rate=1.0):
        value1 = agent0.action_value(state)
        value2 = agent1.action_value(state)
        # print(‘value1 an value2 is: ‘, value1, value2)
        action_values = self.c(value1, value2, rate)
        action = np.argmax(action_values)
        return action

    def mixed_choose_action_negative(self, agent0, agent1, state, rate=1.0):
        value1 = agent0.action_value(state)
        value2 = agent1.action_value(state)
        # action_values = rate * value1 * (1 – rate) * (-value2)
        action_values = self.c(value1, value2, rate)
        action = np.argmin(action_values)
        return action

    def run_pendulum_SeparateSample(self, use_q_mod=False, q_mod_rate=None, mixed_sample=False, test_rate=0.5):
        episode_reward = []
        episode_positive = []
        episode_negative = []
        total_step = [0, 0]
        action_space = 11
        MAX_EPISODE = 600
        UPPER_LIMIT = 2000
        rl0 = self.rls[0]
        rl1 = self.rls[1]
        for episode in range(MAX_EPISODE):
            for i in range(len(self.rls)):
                step = 0
                sum_reward = 0
                state = self.env.reset()
                while True:
                    step += 1
                    total_step[i] += 1
                    if mixed_sample is False:
                        action = self.rls[i].choose_action(state)
                    else:
                        if np.random.uniform() > self.rls[i].epsilon: # choosing action
                            action = np.random.randint(0, self.rls[i].n_actions)
                        else:
                            if i == 0:
                                action = self.mixed_choose_action(rl0, rl1, state, q_mod_rate[0])
                            else:
                                action = self.mixed_choose_action(rl1, rl0, state, q_mod_rate[1])

                    # RL take action and get next observation and reward
                    f_action = (action - (action_space - 1)/ 2)/ ((action_space - 1) / 4)
                    state_, reward, done, _ = self.env.step(np.array([f_action]))
                    reward /= 10
                    sum_reward += reward
                    new_reward = self.rls[i].reward_transform(reward)
                    transition = np.hstack((state, [action], [new_reward], state_))
                    self.rls[i].memory.save(transition)

                    if total_step[i] > 200: # Train
                        if use_q_mod is True:
                            if i == 0:
                                sample_index = self.rls[0].memory.sampling()
                                q_target, q_eval = self.rls[1].action_value_batch(sample_index)
                                q_mod_eval = -q_eval
                                q_mod_target = -q_target
                                self.rls[0].learn(samples=sample_index,
                                                  q_mod_target=q_mod_target,
                                                  q_mod_eval=q_mod_eval,
                                                  q_mod_rate=q_mod_rate[0])
                            else:
                                sample_index = self.rls[1].memory.sampling()
                                q_target, q_eval = self.rls[0].action_value_batch(sample_index)
                                q_mod_eval = -q_eval
                                q_mod_target = -q_target
                                self.rls[1].learn(samples=sample_index,
                                                  q_mod_target=q_mod_target,
                                                  q_mod_eval=q_mod_eval,
                                                  q_mod_rate=q_mod_rate[1])
                        else:
                            self.rls[i].learn()

                    state = state_
                    if done or step == UPPER_LIMIT:
                        if i == 0:
                            episode_positive.append(sum_reward)
                            print(self.rls[i].name, episode, 'sum_reward=', sum_reward)
                            # self.rls[i].set_replace_target_iter(UPPER_LIMIT)
                            break

            sum_reward = 0
            step = 0
            state = self.env.reset()
            while True:
                step += 1
                action = np.argmin(self.rls[1].action_value(state))
                f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
                state_, reward, done, _ = self.env.step(np.array([f_action]))
                reward /= 10
                sum_reward += reward
                state = state_
                if done or step == UPPER_LIMIT:
                    episode_negative.append(sum_reward)
                    break

            sum_reward = 0
            step = 0
            state = self.env.reset()
            while True:
                step += 1
                action = self.mixed_choose_action(rl0, rl1, state, test_rate)
                f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
                state_, reward, done, _ = self.env.step(np.array([f_action]))
                reward /= 10
                sum_reward += reward
                state = state_
                # break while loop when end of this episode
                if done or step == UPPER_LIMIT:
                    episode_reward.append(sum_reward)
                    print('-------------- mix: sum_reward=', sum_reward)
                    break
        # end of game self.env.close()

        import matplotlib.pyplot as plt
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.ylim(-500, 0)
        plt.scatter(np.arange(len(episode_reward)), episode_reward, s=3, marker='o', c='r')
        plt.show()
        plt.ylim(-500, 0)
        plt.scatter(np.arange(len(episode_positive)), episode_positive, s=3, marker='o', c='r')
        plt.show()
        plt.ylim(-500, 0)
        plt.scatter(np.arange(len(episode_negative)), episode_negative, s=3, marker='o', c='r')
        plt.show()

def original(r):
    return r

def negative(r):
    return -r

def c_func(q1, q2, rate):
    return rate * q1 + (1 - rate) * (-q2)
    # print(q1.shape)
    #  return -(q1 * q2)
    r1 = np.argsort(np.argsort(q1))
    r2 = np.argsort(np.argsort(-q2))
    r = r1 + r2
    # d = np.random.random(np.array(r.shape)) / 100
    d = q1 * (-q2) / 1000000
    return r + d

def PendulumDQN(algo_type="classic"):
    np.random.seed(1)
    tf.set_random_seed(1) # reproducible
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    env.reset()
    # n_actions = env.action_space.n
    # n_features = env.state.shape[0]
    action_space = 11
    n_features = 3
    memory0 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + ACTION_DIM + 1, BATCH_SIZE)
    memory1 = Memory(n_features, MEMORY_CAPACITY, n_features * 2 + ACTION_DIM + 1, BATCH_SIZE)
    rl0 = DoubleDQN(n_actions=action_space,
                    n_features=n_features,
                    name='ddqn0',
                    memory=memory0,
                    learning_rate=0.001,
                    reward_decay=0.9,
                    e_greedy=0.95,
                    e_greedy_increment=0.00001,
                    double_q=True,
                    reward_transform=original,
                    reverse=False, reward_offset=0,
                    algo_type=algo_type,
                    combine=c_func)
    rl1 = DoubleDQN(n_actions=action_space,
                    n_features=n_features,
                    name='ddqn1',
                    memory=memory1,
                    learning_rate=0.001,
                    reward_decay=0.9,
                    e_greedy=0.95,
                    e_greedy_increment=0.00001,
                    double_q=True,
                    reward_transform=negative,
                    reverse=True,
                    reward_offset=0,
                    algo_type=algo_type,
                    combine=c_func)
    game = Game(env, [rl0, rl1], combine=c_func)
    if algo_type is "classic":
        print(algo_type, " Pendulum DQN/ ", "mix/ ", "lr=0.001/0.001/ ", "r= ", "gamma=0.9/0.9")
        game.run_pendulum_mix(action_space)
    else:
        print(algo_type, " Pendulum DQN/ ", "lr=0.001/0.001/ ", "r= ", "gamma=0.9/0.9 ", "lambda=0.5/0.5")
        game.run_pendulum_SeparateSample(use_q_mod=True, mixed_sample=True, q_mod_rate=[0.5, 0.5], test_rate=0.5)


if __name__ == "__main__":
    np.random.seed(1)
    tf.set_random_seed(1) # reproducible
    #  CartPoleDQN(algo_type="new_algo_2")
    #  CartPoleDDQN(algo_type="classic")
    #  MountainCarDQN(algo_type="new_algo_2")
    # MountainCarDDQN(algo_type="new_algo_2")
    PendulumDQN(algo_type="new_algo_2")