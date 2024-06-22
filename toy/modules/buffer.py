
from typing import Tuple
import numpy as np
import torch


class ReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        state_size: int,
        action_size: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.state = np.zeros((buffer_size, state_size))
        self.action = np.zeros((buffer_size, action_size))
        self.reward = np.zeros((buffer_size, 1))
        self.state_next = np.zeros((buffer_size, state_size))
        self.done = np.zeros((buffer_size, 1))
        self.idx = 0
        self.size = 0

    def sample(self, batch_size, device) \
    -> Tuple[np.array, np.array, np.array, np.array]:
        i = np.random.randint(self.size, size=batch_size)
        s = torch.tensor(self.state[i, ...], dtype=torch.float32, device=device)
        a = torch.tensor(self.action[i, ...], dtype=torch.float32, device=device)
        r = torch.tensor(self.reward[i, ...], dtype=torch.float32, device=device)
        s_ = torch.tensor(self.state_next[i, ...], dtype=torch.float32, device=device)
        d = torch.tensor(self.done[i, ...], dtype=torch.float32, device=device)
        return {
            'states': s,
            'actions': a,
            'rewards': r,
            'next_states': s_,
            'dones': d,
        }

    def store(self, s, a, r, s_, d) -> None:
        self.state[self.idx] = s
        self.action[self.idx] = a
        self.reward[self.idx] = r
        self.state_next[self.idx] = s_
        self.done[self.idx] = d
        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def __len__(self) -> int:
        return self.size

