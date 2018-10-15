import tensorflow as tf
from memory_new import Memory
from config import MEMORY_CAPACITY, BATCH_SIZE


LR_A = 0.001
LR_C = 0.002
TAU = 0.01


class DDPG:
    def __init__(self, sess, state_dim, action_dim, action_bound):
        self.memory = Memory(state_dim * 2 + action_dim + 1, MEMORY_CAPACITY, BATCH_SIZE)
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
        self.state_ = tf.placeholder(tf.float32, [None, self.state_dim], 'state_')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

        self.action = self.build_actor(self.state)
        q = self.build_critic(self.state, self.action)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            ema.average(getattr(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        a_ = self.build_actor(self.state_, reuse=True, custom_getter=ema_getter)
        q_ = self.build_critic(self.state_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = -tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)




    def build_actor(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(inputs=s,
                                  units=30,
                                  activation=tf.nn.relu,
                                  name='l1',
                                  trainable=trainable)
            a = tf.layers.dense(inputs=net,
                                units=self.action_dim,
                                activation=tf.nn.tanh,
                                name='a',
                                trainable=trainable)
            return tf.multiply(a, self.action_bound, name='scale_a')

    def build_critic(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.action_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)