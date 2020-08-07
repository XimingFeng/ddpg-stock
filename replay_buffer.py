import numpy as np


class Buffer:

    def __init__(self, state_dim, action_dim, buffer_capacity=1000, batch_size=32):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset_buffer()

    def reset_buffer(self):
        state_buffer_dim = (self.buffer_capacity,) + self.state_dim
        action_buffer_dim = (self.buffer_capacity,) + self.action_dim
        self.state_buffer = np.zeros(shape=state_buffer_dim)
        self.action_buffer = np.zeros(shape=action_buffer_dim)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros(state_buffer_dim)
        self.is_term_buffer = np.zeros((self.buffer_capacity, 1))

    def insert(self, state, action, reward, nxt_state, is_term):
        """
        Insert observation into replay buffer
        :param state:
        :param action:
        :param reward:
        :param nxt_state:
        :param is_term:
        :return:
        """
        idx = self.buffer_counter % self.buffer_capacity

        self.state_buffer[idx] = np.squeeze(state)
        self.action_buffer[idx] = np.squeeze(action)
        self.reward_buffer[idx] = np.squeeze(reward)
        self.next_state_buffer[idx] = np.squeeze(nxt_state)
        self.is_term_buffer[idx] = np.squeeze(is_term)

        self.buffer_counter += 1

    def sample_batch(self):
        """
        sample a batch from replay buffer
        :return:
        """
        num_replays = self.buffer_counter if self.buffer_counter < self.buffer_capacity else self.buffer_capacity
        if self.buffer_counter >= self.batch_size:
            batch_idxs = np.random.choice(num_replays, self.batch_size, replace=False)
        else:
            batch_idxs = np.random.choice(num_replays, self.buffer_counter, replace=False)

        return self.state_buffer[batch_idxs], self.action_buffer[batch_idxs], \
               self.reward_buffer[batch_idxs], self.next_state_buffer[batch_idxs], \
               self.is_term_buffer[batch_idxs]
