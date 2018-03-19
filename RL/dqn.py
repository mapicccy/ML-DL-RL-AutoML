import gym

from RL_brain import DeepQNetwork

def CartPoleDQN():
    env = gym.make('CartPole-v0')
    env.reset()

    rl0 = DeepQNetwork(env.env.action_space.n)
