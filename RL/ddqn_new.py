import numpy as np
import tensorflow as tf
from memory_new import Memory
from config import *

class DoubleDQN:
    def __init__(
        self,
        n_actions,
        n_features,
        name='ddqn',
        memory=None,
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False,
        double_q=True,
        sess=None,
        reverse=False,
        reward_transform=None,
        reward_offset=0,
        algo_type='classic',
        combine=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.epsilon_max = 0.9999
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy

        self.double_q = double_q # decide to use double q or not

        self.name = name
        self.reverse = reverse
        self.reward_transform = reward_transform
        self.reward_offset = reward_offset

        self.learn_step_counter = 0
        self.last_replace_target = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.algo_type = algo_type
        self.c = combine

        if memory is None:
            print('Please allocate Memory object first.')
            exit(0)
        else:
            self.memory = memory

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)
        self.cost_his = []

    def set_replace_target_iter(self, n):
        self.replace_target_iter = n

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(self.name+'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(self.name+'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable(self.name+'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(self.name+'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out

        # —————— build evaluate_net ——————
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 128, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # —————— build target_net ——————
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'): # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon: # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def action_value(self, observation, target_value=False):
        observation = observation[np.newaxis, :]
        if target_value is True:
            actions_value = self.sess.run(self.q_next, feed_dict={self.s_: observation})
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return actions_value[0]

    def action_value_batch(self, sample_index):
        batch_memory = self.memory.batch_load(sample_index, self.reward_offset)
        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: batch_memory[:, -self.n_features:], # next observation for target net
                                                       self.s: batch_memory[:, -self.n_features:]}) # next observation for main net
        return q_next, q_eval4next

    def learn(self, samples=None, q_mod_eval=None, q_mod_target=None, q_mod_rate=None):
        if self.learn_step_counter - self.last_replace_target >= self.replace_target_iter:
            self.sess.run(self.replace_target_op)
            # print(‘target_params_replaced:{}’.format(self.replace_target_iter))
            self.last_replace_target = self.learn_step_counter

        if samples is not None:
            sample_index = samples
        else:
            sample_index = self.memory.sampling()
        if sample_index is None:
            print('Error: cannot get valid sample index')
            return
        batch_memory = self.memory.batch_load(sample_index, self.reward_offset)
        # print(batch_memory)

        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: batch_memory[:, -self.n_features:], # next observation for target net
                                                       self.s: batch_memory[:, -self.n_features:]}) # next observation for main net
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # print(self.algo_type)
        if self.algo_type is 'classic':
            if self.double_q:
                max_act4next = np.argmax(q_eval4next, axis=1) # the action that brings the highest value is evaluated by q_eval
                selected_q_next = q_next[batch_index, max_act4next] # Double DQN, select q_next depending on above actions
            else:
                selected_q_next = np.max(q_next, axis=1) # the natural DQN

        elif self.algo_type is 'new_algo_1':
            if self.double_q:
                if q_mod_eval is not None and q_mod_target is not None and q_mod_rate is not None:
                    q_eval4next = self.c(q_eval4next, q_mod_eval, q_mod_rate)
                    q_next = self.c(q_next, q_mod_target, q_mod_rate)
                max_act4next = np.argmax(q_eval4next, axis=1) # the action that brings the highest value is evaluated by q_eval
                selected_q_next = q_next[batch_index, max_act4next] # Double DQN, select q_next depending on above actions
            else: # Natural DQN
                if q_mod_target is not None and q_mod_rate is not None:
                    q_next = self.c(q_next, q_mod_target, q_mod_rate)
                selected_q_next = np.max(q_next, axis=1) # the natural DQN

        elif self.algo_type is 'new_algo_2':
            if self.double_q:
                if q_mod_eval is not None and q_mod_target is not None and q_mod_rate is not None:
                    q_eval4next = self.c(q_eval4next, q_mod_eval, q_mod_rate)
                max_act4next = np.argmax(q_eval4next, axis=1) # the action that brings the highest value is evaluated by q_eval
                selected_q_next = q_next[batch_index, max_act4next] # Double DQN, select q_next depending on above actions
            else: # Natural DQN
                if q_mod_target is not None and q_mod_rate is not None:
                    q_next_mod = self.c(q_next, q_mod_target, q_mod_rate)
                else:
                    q_next_mod = q_next
                max_next = np.argmax(q_next_mod, axis=1)
                selected_q_next = q_next[batch_index, max_next]

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                     self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1 # print(self.learn_step_counter, self.last_replace_target,self.replace_target_iter)
        return q_next, q_eval4next