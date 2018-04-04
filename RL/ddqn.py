import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)


class DDQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 memory,
                 name='ddqn',
                 learning_rate=0.001,
                 reward_delay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 e_greedy_increment=None,
                 output_graph=False,
                 double_q=True,
                 sess=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.memory = memory
        self.name = name
        self.lr = learning_rate
        self.gamma = reward_delay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q
        self.learn_step_counter = 0

        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(self.name + 'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(self.name + 'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable(self.name + 'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(self.name + 'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2

            return out

        # ---------- build evaluate net ------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params',
                                                           tf.GraphKeys.GLOBAL_VARIABLES], 128, \
                                                          tf.random_normal_initializer(0., .3), \
                                                          tf.constant_initializer(.1)
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------- build target net ------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        with tf.variable_scope('target_get'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.save(transition)

    def set_replace_target_iter(self, n):
        self.replace_target_iter = n

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(action_value)

        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(action_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)
        return action

    def action_value(self, observation):
        observation = observation[np.newaxis, :]
        action = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return action[0]

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget params replaced\n')

        if self.memory.memory_counter > self.memory.memory_size:
            sample_index = np.random.choice(self.memory.memory_size, self.memory.batch_size)
        else:
            sample_index = np.random.choice(self.memory.memory_counter, self.memory.batch_size)
        batch_memory = self.memory.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],
                       self.s: batch_memory[:, -self.n_features:]})
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.memory.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next,
                                     axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
