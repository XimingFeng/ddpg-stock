import numpy as np

NUM_ASSETS = None
STATE_DIM = NUM_ASSETS * 4  # Each asset:
ACTION_DIM = None


class Buffer:

    def __init__(self, buffer_capacity=1000, batch_size=32):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, STATE_DIM))
        self.action_buffer = np.zeros((self.buffer_capacity, ACTION_DIM))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, STATE_DIM))
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

        self.state_buffer[idx] = state
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward
        self.next_state_buffer[idx] = nxt_state
        self.is_term_buffer[idx] = is_term

        self.buffer_counter += 1

    def sample_batch(self):
        """
        sample a batch from replay buffer
        :return:
        """
        if self.buffer_counter >= self.batch_size:
            batch_idxs = np.random.choice(self.buffer_counter, self.batch_size, replace=False)
        else:
            batch_idxs = np.random.choice(self.buffer_counter, self.buffer_counter, replace=False)

        return self.state_buffer[batch_idxs], self.action_buffer[batch_idxs], \
               self.reward_buffer[batch_idxs], self.next_state_buffer[batch_idxs], \
               self.is_term_buffer[batch_idxs]
