import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)

GAMMA = 0.9


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], 'state')
        self.a = tf.placeholder(tf.int32, None, 'act')
        self.td_error = tf.placeholder(tf.float32, None, 'td_error')

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(inputs=self.s,
                                 units=20,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name='l1')

            self.acts_prob = tf.layers.dense(inputs=l1,
                                             units=n_actions,
                                             activation=tf.nn.softmax,
                                             kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                             bias_initializer=tf.constant_initializer(0.1),
                                             name='acts_prob')

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def action_probs(self, s):
        s = s[np.newaxis, :]
        action_probs = self.sess.run(self.acts_prob, {self.s: s})
        return action_probs[0]


class Critic(object):
    def __init__(self, sess, n_features, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], name='state')
        self.v_ = tf.placeholder(tf.float32, [1, 1], name='v_next')
        self.r = tf.placeholder(tf.float32, None, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(inputs=self.s,
                                 units=20,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(.1),
                                 name='l1')

            self.v = tf.layers.dense(inputs=l1,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(.1),
                                     name='v')

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})

        return td_error
