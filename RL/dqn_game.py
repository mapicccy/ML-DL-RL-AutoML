import sys
import gym

sys.path.append('./')

from dqn import DeepQNetwork

NUM_EPISODE = 1000
MEMORY_CAPACITY = 100000


class Game(object):
    def __init__(self, env, rls):
        self.env = env
        self.rls = rls

    def run(self):
        step = 0
        for episode in range(NUM_EPISODE):
            observation = self.env.reset()
            ep_r = 0
            print('Episode %d, step = %d' % (episode, step))
            while True:
                self.env.render()

                action = self.rls.choose_action(observation)

                observation_, reward, done, info = self.env.step(action)
                x, x_dot, theta, theta_dot = observation_

                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                reward = r1 + r2

                self.rls.store_transition(observation, action, reward, observation_)

                if step > 1000:
                    self.rls.learn()

                ep_r += reward
                if done:
                    print('episode: ', episode,
                          'ep_r: ', round(ep_r, 2),
                          'epsilon: ', round(self.rls.epsilon, 2))
                    break

                observation = observation_
                step += 1


def CartPoleDQN():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.reset()

    rl0 = DeepQNetwork(env.action_space.n,
                       env.state.shape[0],
                       learning_rate=0.01,
                       reward_decay=0.9,
                       e_greedy=0.9,
                       replace_target_iter=200,
                       memory_size=100000,
                       batch_size=32,
                       e_greedy_increment=None,
                       output_graph=None)

    game = Game(env, rl0)
    game.run()


if __name__ == '__main__':
    CartPoleDQN()
