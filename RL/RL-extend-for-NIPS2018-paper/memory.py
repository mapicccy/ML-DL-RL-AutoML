import numpy as np
from config import *


class Memory(object):
    def __init__(self, n_features, memory_size, record_size, batch_size):
        self.policies = []
        self.n_features = n_features
        self.memory_size = memory_size
        self.record_size = record_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, self.record_size))
        self.memory_counter = 0

    def add_policy(self, policy):
        self.policies.append(policy)

    def save(self, transition):
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def batch_load(self, index, reward_offset):
        elements = np.arange(self.record_size)
        remove_list = []
        for i in range(NUM_POLICIES):
            if i != reward_offset:
                remove_list.append(self.n_features + ACTION_DIM + i)
        elements = np.delete(elements, remove_list)
        batch_memory = self.memory[index, :]

        return batch_memory[:, elements]

    def sampling(self):
        size = min(self.memory_counter, self.memory_size)
        return np.random.choice(size, self.batch_size)

    def save_memory(self, name):
        np.save(name, self.memory)

    def set_memory(self, m):
        self.memory = m
        self.memory_counter = 100000
