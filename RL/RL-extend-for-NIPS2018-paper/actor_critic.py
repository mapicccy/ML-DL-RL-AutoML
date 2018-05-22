import numpy as np
import tensorflow as tf
from config import *


class Actor(object):
    def __init__(self, sess, name, n_features, n_actions, lr=0.001, fc_node=128):
        self.sess = sess
        self.name = name

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=fc_node,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name=self.name + 'l1'
            )

            logits = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name=self.name + 'acts_prob'
            )

            self.acts_prob_logits = tf.check_numerics(logits, "NaN in logits")
            # self.acts_prob_logits = logits

            self.acts_prob = tf.nn.softmax(self.acts_prob_logits) # get action probabilities

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(tf.clip_by_value(self.acts_prob[0, self.a], 1e-20, 1.0))
            # log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        logits, probs = self.sess.run([self.acts_prob_logits, self.acts_prob], {self.s: s})
        # print(logits, probs)
        # probs = self.sess.run([self.acts_prob], {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

    def action_values(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return probs


class Actor_Continuous(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001, fc_node=128):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=fc_node,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        mu = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)    # min(v) = max(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions


class Critic(object):
    def __init__(self, sess, name, n_features, lr=0.01, gamma=0.9, fc_node=128):
        self.sess = sess
        self.name = name
        self.gamma = gamma

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = fc_node,  # number of hidden units
                activation = tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer = tf.random_normal_initializer(0., .1),  # weights
                bias_initializer = tf.constant_initializer(0.1),  # biases
                name = self.name + 'l1'
            )

            self.v = tf.layers.dense(
                inputs = l1,
                units = 1,  # output units
                activation = None,
                kernel_initializer = tf.random_normal_initializer(0., .1),  # weights
                bias_initializer = tf.constant_initializer(0.1),  # biases
                name = self.name + 'V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, use_q_mod=False, q_mod_rate=None, q_mod_value=None, c=None):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        if use_q_mod is True and q_mod_rate is not None:
            v_ = c(v_, q_mod_value, q_mod_rate)
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})

        return td_error

    def critic_value(self,s_):
        s_ = s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        return v_


class RL(object):
    def __init__(self, actor, critic, reverse, reward_transform):
        self.actor = actor
        self.critic = critic
        self.reverse = reverse
        self.reward_transform = reward_transform


class RL_a2c(object):
    def __init__(self, actor, critic1, critic2, reverse=False, reward_transform=None):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.reverse = reverse
        self.reward_transform = reward_transform