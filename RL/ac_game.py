import tensorflow as tf
import numpy as np
import gym
import sys

sys.path.append('./')
from actor_critic import Actor, Critic

# pylint: disable=protected-access

np.random.seed(2)
tf.set_random_seed(2)

MAX_EPISODE = 3000
MAX_EP_STEPS = 1000
DISPLAY_REWARD_THRESHOLD = 1000
LR_A = 0.001
LR_C = 0.01


class Game(object):
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.reward_his = []

    # build single positive network
    def run(self):
        RENDER = False
        for i_episode in range(MAX_EPISODE):
            s = self.env.reset()
            t = 0
            track_r = []
            while True:
                if RENDER:
                    self.env.render()

                a = self.actor.choose_action(s)
                s_, r, done, info = self.env.step(a)

                if done:
                    r = -20

                track_r.append(r)

                self.td_error = self.critic.learn(s, r, s_)
                self.actor.learn(s, a, self.td_error)

                s = s_
                t += 1

                if done or t > MAX_EP_STEPS:
                    ep_rs_sum = sum(track_r)
                    self.reward_his.append(ep_rs_sum)

                    if not hasattr(self, 'reward'):
                        self.reward = ep_rs_sum
                    else:
                        self.reward = self.reward * 0.95 + ep_rs_sum * 0.05

                    if r > DISPLAY_REWARD_THRESHOLD:
                        RENDER = True
                    print('episode: ', i_episode, 'reward: ', int(self.reward))
                    break

        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.reward_his)), self.reward_his)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.show()

    # build positive and negative network
    def run_double(self):
        RENDER = False
        reward_his = [[], []]
        for i_episode in range(MAX_EP_STEPS):
            t = 0
            track_r = []
            s = [self.env[0].reset(), self.env[1].reset()]
            done = [False, False]
            while done[0] is False or done[1] is False:
                for i in range(len(self.env)):
                    if RENDER:
                        self.env[i].render()

                    if done[i]:
                        continue

                    a = self.actor.choose_action(s[i])
                    s_, r, done[i], info = self.env[i].step(a)

                    if done[i]:
                        r = -20

                    # ToDo build network
                    td_error = self.critic.learn()


def CartPoleAC():
    env1 = gym.make('CartPole-v0')
    env2 = gym.make('CartPole-v0')
    env1.seed(2)
    env2.seed(2)
    env1 = env1.unwrapped
    env1.reset()
    env2 = env2.unwrapped
    env2.reset()

    n_features = env1.observation_space.shape[0]
    n_actions = env1.action_space.n

    sess = tf.Session()

    actor = Actor(sess, n_features, n_actions, lr=LR_A)
    critic = Critic(sess, n_features, lr=LR_C)
    sess.run(tf.global_variables_initializer())

    game = Game([env1, env2], actor, critic)
    game.run()


if __name__ == '__main__':
    CartPoleAC()
