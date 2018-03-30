import numpy as np


class Memory:
    def __init__(self, n_features, memory_size, record_size, batch_size):
        self.policies = []
        self.n_features = n_features
        self.memory_size = memory_size
        self.record_size = record_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, self.record_size))
        self.memory_counter = 0

    def save(self, transition):
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def load_batch(self, index, reward_offset):
        pass