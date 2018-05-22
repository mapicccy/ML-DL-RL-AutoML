import numpy as np
import tensorflow as tf
import sys
from ddqn import DoubleDQN
from actor_critic import *
from memory import Memory
from config import *
from gym_instance import Gym_instance


def original(r):
    return r


def negative(r):
    return -r


def r_expand(r, m, s=None, s_=None):
    if REWARD_SCALAR and r != STOP_REWARD:
        r = abs(s[0] - s_[0])
    if abs(r) != STOP_REWARD:
        r = r * m
    return r


def c_func_plus(q1, q2, rate):
    return rate * q1 - (1 - rate) * q2


class Environment:
    def __init__(self, env, rl_name, algo_type, combine=None):
        self.env = env
        self.c = combine
        self.rl_name = rl_name
        if self.rl_name == 'dqn':
            memory0 = Memory(self.env.n_features, MEMORY_CAPACITY, self.env.n_features * 2 + ACTION_DIM + 1, BATCH_SIZE)
            memory1 = Memory(self.env.n_features, MEMORY_CAPACITY, self.env.n_features * 2 + ACTION_DIM + 1, BATCH_SIZE)
            self.rls = self.dqn_list(memory0, memory1, algo_type)
        elif self.rl_name == 'ddqn':
            memory0 = Memory(self.env.n_features, MEMORY_CAPACITY, self.env.n_features * 2 + ACTION_DIM + 1, BATCH_SIZE)
            memory1 = Memory(self.env.n_features, MEMORY_CAPACITY, self.env.n_features * 2 + ACTION_DIM + 1, BATCH_SIZE)
            self.rls = self.ddqn_list(memory0, memory1, algo_type)
        elif self.rl_name == 'ac':
            self.rls = self.ac_list()
        elif self.rl_name == 'ac_continuous':
            self.rls = self.ac_continuous_list()
        else:
            print('more algorithm will be supported soon!...')
            sys.exit(1)

    def ac_list(self):
        sess0 = tf.Session()
        actor0 = Actor(sess0, 'actor0', n_features=self.env.n_features, n_actions=self.env.n_actions, lr=LR_A, fc_node=FC_NODE)
        critic0 = Critic(sess0, 'critic0', n_features=self.env.n_features, gamma=GAMMA_C0, lr=LR_C0, fc_node=FC_NODE)
        critic1 = Critic(sess0, 'critic1', n_features=self.env.n_features, gamma=GAMMA_C1, lr=LR_C1, fc_node=FC_NODE)
        rl0 = RL_a2c(actor0, critic0, critic1, False, original)
        sess0.run(tf.global_variables_initializer())
        return [rl0]

    def ac_continuous_list(self):
        action_space_high = self.env.action_space_high
        # print(action_space_high)
        sess0 = tf.Session()
        actor0 = Actor_Continuous(sess0, n_features=self.env.n_features,
                                  action_bound=[-action_space_high, action_space_high],
                                  lr=LR_A,
                                  fc_node=FC_NODE)
        critic0 = Critic(sess0, 'critic0', n_features=self.env.n_features, gamma=GAMMA_C0, lr=LR_C0, fc_node=FC_NODE)
        critic1 = Critic(sess0, 'critic1', n_features=self.env.n_features, gamma=GAMMA_C1, lr=LR_C1, fc_node=FC_NODE)
        rl0 = RL_a2c(actor0, critic0, critic1, reverse=False, reward_transform=original)
        sess0.run(tf.global_variables_initializer())
        return [rl0]

    def dqn_list(self, memory0, memory1, algo_type):
        return [DoubleDQN(n_actions=self.env.n_actions,
                          n_features=self.env.n_features,
                          name='dqn0',
                          memory=memory0,
                          learning_rate=LR_POSITIVE,
                          reward_decay=REWARD_DECAY_POSITIVE,
                          e_greedy=E_GREEDY_POSITIVE,
                          replace_target_iter=REPLACE_TARGET_ITER,
                          batch_size=BATCH_SIZE,
                          e_greedy_increment=E_GREEDY_INCREMENT,
                          double_q=False,
                          reward_transform=original,
                          reverse=False,
                          reward_offset=0,
                          algo_type=algo_type,
                          combine=self.c,
                          fc_node=FC_NODE
                          ),
                DoubleDQN(n_actions=self.env.n_actions,
                          n_features=self.env.n_features,
                          name='dqn1',
                          memory=memory1,
                          learning_rate=LR_NEGATIVE,
                          reward_decay=REWARD_DECAY_NEGATIVE,
                          e_greedy=E_GREEDY_NEGATIVE,
                          replace_target_iter=REPLACE_TARGET_ITER,
                          batch_size=BATCH_SIZE,
                          e_greedy_increment=E_GREEDY_INCREMENT,
                          double_q=False,
                          reward_transform=negative,
                          reverse=True,
                          reward_offset=0,
                          algo_type=algo_type,
                          combine=self.c,
                          fc_node=FC_NODE
                          )]

    def ddqn_list(self, memory0, memory1, algo_type):
        return [DoubleDQN(n_actions=self.env.n_actions,
                          n_features=self.env.n_features,
                          name='ddqn0',
                          memory=memory0,
                          learning_rate=LR_POSITIVE,
                          reward_decay=REWARD_DECAY_POSITIVE,
                          e_greedy=E_GREEDY_POSITIVE,
                          replace_target_iter=REPLACE_TARGET_ITER,
                          batch_size=BATCH_SIZE,
                          e_greedy_increment=E_GREEDY_INCREMENT,
                          double_q=True,
                          reward_transform=original,
                          reverse=False,
                          reward_offset=0,
                          algo_type=algo_type,
                          combine=self.c,
                          fc_node=FC_NODE
                          ),
                DoubleDQN(n_actions=self.env.n_actions,
                          n_features=self.env.n_features,
                          name='ddqn1',
                          memory=memory1,
                          learning_rate=LR_NEGATIVE,
                          reward_decay=REWARD_DECAY_NEGATIVE,
                          e_greedy=E_GREEDY_NEGATIVE,
                          replace_target_iter=REPLACE_TARGET_ITER,
                          batch_size=BATCH_SIZE,
                          e_greedy_increment=E_GREEDY_INCREMENT,
                          double_q=True,
                          reward_transform=negative,
                          reverse=True,
                          reward_offset=0,
                          algo_type=algo_type,
                          combine=self.c,
                          fc_node=FC_NODE
                          )]

    def discrete_action(self, action):
        f_action = (action - (self.env.n_actions - 1) / 2) / ((self.env.n_actions - 1) / 4)
        return np.array([f_action])

    def mixed_choose_action(self, agent0, agent1, state, rate=1.0, reverse=False):
        value1 = agent0.action_value(state)
        value2 = agent1.action_value(state)
        action_values = self.c(value1, value2, rate)
        if reverse:
            action = np.argmin(action_values)
        else:
            action = np.argmax(action_values)
        return action

    def set_target_iter(self, i, step):
        if self.env.name == 'CartPole-v0':
            self.rls[i].set_replace_target_iter(step * 5)
        else:
            self.rls[i].set_replace_target_iter(FINAL_ITER)

    def train_network(self, total_step, mixed_sample, use_q_mod, q_mod_rate=None, test_rate=0.5, compute_reward=True):
        rl0 = self.rls[0]
        rl1 = self.rls[1]
        for i in range(len(self.rls)):
            sum_reward = 0
            step = 0
            state = self.env.env.reset()
            while True:
                step += 1
                total_step[i] += 1
                if mixed_sample is False:
                    if BOTH_GOOD_ACTION:
                        if np.random.uniform() > self.rls[i].epsilon:  # choosing action
                            action = np.random.randint(0, self.rls[i].n_actions)
                        else:
                            if i == 0:
                                action = np.argmax(self.rls[i].action_value(state))
                            else:
                                action = np.argmin(self.rls[i].action_value(state))
                    else:
                        action = self.rls[i].choose_action(state)
                else:
                    if np.random.uniform() > self.rls[i].epsilon:  # choosing action
                        action = np.random.randint(0, self.rls[i].n_actions)
                    else:
                        if i == 0:
                            action = self.mixed_choose_action(rl0, rl1, state, q_mod_rate[0])
                        else:
                            action = self.mixed_choose_action(rl1, rl0, state, q_mod_rate[1])

                if DISCRETE_ACTION:
                    f_action = self.discrete_action(action)
                    state_, reward, done, _ = self.env.env.step(f_action)
                else:
                    state_, reward, done, _ = self.env.env.step(action)

                if done and step < UPPER_LIMIT:
                    reward = STOP_REWARD

                new_reward = r_expand(reward, R_EXPAND, state, state_)
                sum_reward += reward
                # print('      --------', reward, new_reward)
                new_reward = self.rls[i].reward_transform(new_reward)
                # print(state, state_, action, new_reward)
                transition = np.hstack((state, [action], [new_reward], state_))
                self.rls[i].memory.save(transition)

                if total_step[i] > START_LEARN_STEP and step % 5 == 0:
                    if use_q_mod is True:
                        if i == 0:
                            sample_index = rl0.memory.sampling()
                            q_mod_target, q_mod_eval = rl1.action_value_batch(sample_index)
                            rl0.learn(samples=sample_index, q_mod_target=q_mod_target, q_mod_eval=q_mod_eval,
                                      q_mod_rate=q_mod_rate[0])
                        else:
                            sample_index = rl1.memory.sampling()
                            q_mod_target, q_mod_eval = rl0.action_value_batch(sample_index)
                            rl1.learn(samples=sample_index, q_mod_target=q_mod_target, q_mod_eval=q_mod_eval,
                                      q_mod_rate=q_mod_rate[1])
                    else:
                        self.rls[i].learn()

                state = state_
                if done or step == UPPER_LIMIT:
                    self.set_target_iter(i, step)
                    break

    def test_network(self, episode_reward, i, compute_reward):
        sum_reward = 0
        step = 0
        state = self.env.env.reset()
        while True:
            step += 1
            action_values = self.rls[i].action_value(state)
            action = np.argmax(action_values) if i == 0 else np.argmin(action_values)

            if DISCRETE_ACTION:
                f_action = self.discrete_action(action)
                state_, reward, done, _ = self.env.env.step(f_action)
            else:
                state_, reward, done, _ = self.env.env.step(action)

            new_reward = r_expand(reward, R_EXPAND, state, state_)
            state = state_
            sum_reward += new_reward
            if done or step == UPPER_LIMIT:
                if compute_reward:
                    episode_reward.append(sum_reward)
                else:
                    episode_reward.append(step)
                break

    def test_mixed_network(self, episode_reward, test_rate, compute_reward):
        rl0 = self.rls[0]
        rl1 = self.rls[1]
        sum_reward = 0
        step = 0
        state = self.env.env.reset()
        while True:
            step += 1
            action = self.mixed_choose_action(rl0, rl1, state, rate=test_rate)

            if DISCRETE_ACTION:
                f_action = self.discrete_action(action)
                state_, reward, done, _ = self.env.env.step(f_action)
            else:
                state_, reward, done, _ = self.env.env.step(action)

            if done and (step < UPPER_LIMIT):
                reward = STOP_REWARD

            new_reward = r_expand(reward, R_EXPAND, state, state_)
            sum_reward += new_reward
            state = state_
            if done or step == UPPER_LIMIT:
                if compute_reward:
                    episode_reward.append(sum_reward)
                else:
                    episode_reward.append(step)
                break

    def run_environment_on_dqn_or_ddqn(self, mixed_sample=False, use_q_mod=False, q_mod_rate=None, test_rate=0.5,
                                       compute_reward=True):
        print('run %s on %s' % (self.env.name, self.rl_name))
        print('params:n_actions=%d, n_features=%d, lr=%f/%f, gamma=%f/%f, e_greedy=%f/%f, iter=%d, batch_size=%d'
              % (self.env.n_actions, self.env.n_features, LR_POSITIVE, LR_NEGATIVE,
                 REWARD_DECAY_POSITIVE, REWARD_DECAY_NEGATIVE, E_GREEDY_POSITIVE, E_GREEDY_NEGATIVE,
                 REPLACE_TARGET_ITER,
                 BATCH_SIZE))

        total_step = [0, 0]
        episode_mix = []
        episode_positive = []
        episode_negative = []
        for episode in range(NUM_EPISODE):
            if TRAIN_NETWORK:
                self.train_network(total_step, mixed_sample, use_q_mod, q_mod_rate, test_rate, compute_reward)

            if TEST_POSITIVE_NETWORK:
                self.test_network(episode_positive, 0, compute_reward)

            if TEST_NEGATIVE_NETWORK:
                self.test_network(episode_negative, 1, compute_reward)

            if TEST_MIXED_NETWORK:
                self.test_mixed_network(episode_mix, test_rate=test_rate, compute_reward=compute_reward)

            print(episode, episode_positive[episode], episode_negative[episode], episode_mix[episode])

        import matplotlib.pyplot as plt
        plt.scatter(np.arange(len(episode_mix)), episode_mix, s=10, marker='o', c='r')
        plt.show()
        plt.scatter(np.arange(len(episode_positive)), episode_positive, s=10, marker='o', c='g')
        plt.show()
        plt.scatter(np.arange(len(episode_negative)), episode_negative, s=10, marker='o', c='b')
        plt.show()

        if SAVE_DATA:
            np.save(OUTPUT_FILE_POSITIVE, episode_positive)
            np.save(OUTPUT_FILE_NEGATIVE, episode_negative)
            np.save(OUTPUT_FIEL_MIXED, episode_mix)

    def run_environment_on_ac(self, c=None, c_rate=0.5, compute_reward=True):
        print('run %s on %s' % (self.env.name, self.rl_name))
        print('params:n_actions=%d, n_features=%d, lr=%f/%f/%f, gamma=%f/%f, iter=%d, batch_size=%d'
              % (self.env.n_actions, self.env.n_features, LR_A, LR_C0, LR_C1,
                 GAMMA_C0, GAMMA_C1,
                 FINAL_ITER,
                 BATCH_SIZE))

        episode_reward = []
        sum_reward = 0
        rl = self.rls[0]
        for i_episode in range(NUM_EPISODE):
            state = self.env.env.reset()
            step = 0
            while True:
                action = rl.actor.choose_action(state)

                if DISCRETE_ACTION:
                    f_action = self.discrete_action(action)
                    state_, reward, done, _ = self.env.env.step(f_action)
                else:
                    state_, reward, done, _ = self.env.env.step(action)

                step += 1
                if done and step < UPPER_LIMIT:
                    reward = STOP_REWARD

                new_reward = r_expand(reward, R_EXPAND, state, state_)
                sum_reward += new_reward
                new_reward = rl.reward_transform(new_reward)

                c1_td_error = rl.critic1.learn(state, new_reward, state_)
                c2_td_error = rl.critic2.learn(state, -new_reward, state_)
                td_error = c(c1_td_error, c2_td_error, c_rate)
                rl.actor.learn(state, action, td_error)

                state = state_

                if done or step == UPPER_LIMIT:
                    if compute_reward:
                        episode_reward.append(sum_reward)
                    else:
                        episode_reward.append(step)
                    print("a2c", i_episode, step)
                    break

        import matplotlib.pyplot as plt
        plt.scatter(np.arange(len(episode_reward)), episode_reward, s=3, marker='o')
        plt.show()

        if SAVE_DATA:
            np.save(OUTPUT_FILE_MIXED, y)

    def run_environment_on_ac_continuous(self, c=None, use_q_mod=False, c_rate=0.5, compute_reward=True):
        print('run %s on %s' % (self.env.name, self.rl_name))
        print('params:n_actions=%d, n_features=%d, lr=%f/%f/%f, gamma=%f/%f, iter=%d, batch_size=%d'
              % (self.env.n_actions, self.env.n_features, LR_A, LR_C0, LR_C1,
                 GAMMA_C0, GAMMA_C1,
                 FINAL_ITER,
                 BATCH_SIZE))
        episode_reward = []
        rl = self.rls[0]
        for i_episode in range(NUM_EPISODE):
            state = self.env.env.reset()
            step = 0
            sum_reward = 0
            while True:
                # self.env.render()
                action = rl.actor.choose_action(state)

                if DISCRETE_ACTION:
                    f_action = self.discrete_action(action)
                    state_, reward, done, _ = self.env.env.step(f_action)
                else:
                    state_, reward, done, _ = self.env.env.step(action)

                reward = r_expand(reward, R_EXPAND, state, state_)

                step += 1
                sum_reward += reward

                if use_q_mod is False:
                    c1_td_error = rl.critic1.learn(state, reward, state_)
                    c2_td_error = rl.critic2.learn(state, -reward, state_)
                    td_error = c(c1_td_error, c2_td_error, c_rate)
                    rl.actor.learn(state, action, td_error)
                else:
                    v_next_c1 = rl.critic1.critic_value(state_)
                    v_next_c2 = rl.critic2.critic_value(state_)
                    c1_td_error = rl.critic1.learn(state, reward, state_,
                                                   use_q_mod=True, q_mod_rate=c_rate, q_mod_value=v_next_c2, c=c)
                    c2_td_error = rl.critic2.learn(state, -reward, state_,
                                                   use_q_mod=True, q_mod_rate=c_rate, q_mod_value=v_next_c1, c=c)
                    td_error = c(c1_td_error, c2_td_error, c_rate)
                    rl.actor.learn(state, action, td_error)

                state = state_

                if done or step == UPPER_LIMIT:
                    if compute_reward:
                        episode_reward.append(sum_reward)
                    else:
                        episode_reward.append(step)
                    print("a2c", i_episode, sum_reward)
                    break

        import matplotlib.pyplot as plt
        y = np.array(episode_reward)
        x = range(y.shape[0])
        plt.scatter(x, y, s=10, marker='o', c='r')
        plt.show()

        if SAVE_DATA:
            np.save(OUTPUT_FILE_MIXED, y)


if __name__ == '__main__':
    np.random.seed(1)
    tf.set_random_seed(1)
    # gym_env = Gym_instance('CartPole-v0')
    # gym_env = Gym_instance('MountainCar-v0')
    gym_env = Gym_instance('Pendulum-v0', action_space=ACTION_SPACE, observation_space=OBSERVATION_SPACE)
    env = Environment(gym_env, ALGORITHM_NAME, algo_type=ALGORITHM_TYPE, combine=c_func_plus)
    # env.run_environment_on_dqn_or_ddqn(mixed_sample=MIXED_SAMPLE, use_q_mod=USE_Q_MOD, q_mod_rate=Q_MOD_RATE, test_rate=TEST_RATE, compute_reward=COMPUTE_REWARD)
    # env.run_environment_on_ac(c=c_func_plus, c_rate=TEST_RATE, compute_reward=COMPUTE_REWARD)
    env.run_environment_on_ac_continuous(c=c_func_plus, c_rate=TEST_RATE, use_q_mod=USE_Q_MOD, compute_reward=COMPUTE_REWARD)
