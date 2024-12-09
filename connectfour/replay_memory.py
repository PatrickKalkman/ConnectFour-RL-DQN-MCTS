import random
from collections import deque

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity, device):
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)

        self.memory.append(
            (
                state,
                torch.LongTensor([action]).to(self.device),
                torch.FloatTensor([reward]).to(self.device),
                next_state,
                torch.BoolTensor([done]).to(self.device),
            )
        )

    def sample(self, batch_size):
        if len(self) < batch_size:
            batch_size = len(self)

        experiences = random.sample(self.memory, batch_size)

        batch = list(zip(*experiences))

        states = torch.stack(batch[0])
        actions = torch.cat(batch[1])
        rewards = torch.cat(batch[2])
        next_states = torch.stack(batch[3])
        dones = torch.cat(batch[4])

        return (states, actions, rewards, next_states, dones)

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
