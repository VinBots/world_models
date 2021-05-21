"""
This module implements a centralized replay buffer.
"""

# Import libraries
import random
from collections import deque
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Buffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, name, max_size, seed=0):
        """
        Initializes a ReplayBuffer object
        """

        self.name = name
        self.max_size = max_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=self.max_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, exp):
        """
        Add a new experience to memory
        """
        self.memory.append(exp)

    def shuffle(self):
        random.shuffle(self.memory)

    def stack(self, samples):
        """
        Stacks the states, p, v in tensors so that it can be ingested for neural network training
        """

        all_states = torch.stack(
            [
                torch.tensor([e[0]], dtype=torch.float, device=device)
                for e in samples
                if e is not None
            ]
        )
        return all_states

    def buffer_len(self):
        """Return the current size of the buffer."""

        return len(self.memory)
