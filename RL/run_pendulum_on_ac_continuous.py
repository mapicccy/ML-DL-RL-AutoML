import tensorflow as tf
import numpy as np
import gym

np.random.seed(1)
tf.set_random_seed(1)


class Actor(object):
    def __init__(self, sess, n_features, action_space, name='actor', lr=0.0001):
        self.sess = sess
        self.name = name
        self.s = tf.placeholder(tf.float32, [1, n_features], 'state')
        self.a = tf.placeholder(tf.float32, None, 'action')
        self.td_error = tf.placeholder(tf.float32, None, 'td_error')

        l1 = tf.layers.dense(inputs=self.s,
                             units=20,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.random_normal_initializer(0., .1),
                             bias_initializer=tf.constant_initializer(.1),
                             name=self.name + 'l1')

        mu = tf.layers.dense(inputs=l1,
                             units=1,
                             activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(0., .1),
                             bias_initializer=tf.constant_initializer(.1),
                             name=self.name + 'mu')

        sigma = tf.layers.dense(inputs=l1,
                                units=1,
                                activation=tf.nn.softplus,
                                kernel_initializer=tf.random_normal_initializer(0., .1),
                                bias_initializer=tf.constant_initializer(1.),
                                name=self.name + 'sigma')

        global_step = tf.Variable(0, trainable=False)
        self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + .1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_space[0], action_space[1])

        with tf.name_scope(self.name + 'exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.exp_v = log_prob * self.td_error
            self.exp_v += .01 * self.normal_dist.entropy()

        with tf.name_scope(self.name + 'train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action_probs = self.sess.run(self.action, {self.s: s})
        # print(action_probs)
        return action_probs

    def action_probs(self, s):
        s = s[np.newaxis, :]
        action_probs = self.sess.run(self.action, {self.s: s})
        return action_probs


class Critic(object):
    def __init__(self, sess, n_features, name='critic', lr=0.01):
        self.sess = sess
        self.name = name
        with tf.name_scope(self.name + 'inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], 'state')
            self.v_ = tf.placeholder(tf.float32, [1, 1], name='v_next')
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope(self.name + 'critic'):
            l1 = tf.layers.dense(inputs=self.s,
                                 units=20,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(.1),
                                 name=self.name + 'l1')

            self.v = tf.layers.dense(inputs=l1,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(.1),
                                     name=self.name + 'V')

        with tf.variable_scope(self.name + 'squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error


MAX_EPISODE = 1000
MAX_EP_STEP = 2000
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

n_features = env.observation_space.shape[0]
action_space_high = env.action_space.high

# sess = tf.Session()
# actor = Actor(sess, n_features, action_space=[-action_space_high, action_space_high], name='actor', lr=LR_A)
# critic = Critic(sess, n_features, name='critic', lr=LR_C)
# sess.run(tf.global_variables_initializer())
sess0 = tf.Session()
sess1 = tf.Session()
rl = [[Actor(sess0, n_features, action_space=[-action_space_high, action_space_high], name='actor0', lr=LR_A),
       Critic(sess0, n_features, name='critic0', lr=LR_C)],
      [Actor(sess1, n_features, action_space=[-action_space_high, action_space_high], name='actor1', lr=LR_A),
       Critic(sess1, n_features, name='critic1', lr=LR_C)]]
sess0.run(tf.global_variables_initializer())
sess1.run(tf.global_variables_initializer())

episode_positive = []
episode_negative = []
for episode in range(MAX_EPISODE):
    for i in range(len(rl)):
        step = 0
        sum_reward = 0
        state = env.reset()
        while True:
            action = rl[i][0].choose_action(state)
            state_, reward, done, _ = env.step(action)

            reward /= 10

            new_reward = reward if i == 0 else -reward

            td_error = rl[i][1].learn(state, new_reward, state_)
            rl[i][0].learn(state, action, td_error)

            state = state_
            step += 1
            sum_reward += reward

            if step > MAX_EP_STEP:
                if i == 0:
                    episode_positive.append(sum_reward)
                else:
                    episode_negative.append(sum_reward)
                print(i, episode, sum_reward)
                break

    step = 0
    sum_reward = 0
    state = env.reset()
    # while True:
    #     action = rl[0][0].choose_action(state)

import matplotlib.pyplot as plt

plt.scatter(np.arange(len(episode_positive)), episode_positive, s=3, marker='o')
plt.scatter(np.arange(len(episode_negative)), episode_negative, s=3, marker='o')
plt.show()
