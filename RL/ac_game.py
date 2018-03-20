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

def CartPoleAC():
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    env.reset()

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    sess = tf.Session()

    actor = Actor(sess, n_features, n_actions, lr=LR_A)
    critic = Critic(sess, n_features, lr=LR_C)
    sess.run(tf.global_variables_initializer())

    game = Game(env, actor, critic)
    game.run()


if __name__ == '__main__':
    CartPoleAC()
