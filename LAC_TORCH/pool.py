from collections import deque

import numpy as np
import torch


class Pool(object):
    """Memory buffer class."""

    def __init__(
        self,
        s_dim,
        a_dim,
        memory_capacity,
        store_last_n_paths,
        min_memory_size,
        device="cpu",
    ):
        # TODO: Update docstring
        """Initialize memory buffer object.

        Args:
            variant (dict): Dictionary containing all the required memory buffer
                parameters.
            device (str): Whether you want to return the sample on the CPU or GPU.
        """
        self.memory_capacity = memory_capacity
        self.paths = deque(maxlen=store_last_n_paths)
        self.reset()
        self.device = device
        self.memory = {
            "s": torch.zeros([1, s_dim], dtype=torch.float32).to(self.device),
            "a": torch.zeros([1, a_dim], dtype=torch.float32).to(self.device),
            "r": torch.zeros([1, 1], dtype=torch.float32).to(self.device),
            "terminal": torch.zeros([1, 1], dtype=torch.float32).to(self.device),
            "s_": torch.zeros([1, s_dim], dtype=torch.float32).to(self.device),
        }
        self.memory_pointer = 0
        self.min_memory_size = min_memory_size

    def reset(self):
        """Reset memory buffer.
        """
        self.current_path = {
            "s": torch.tensor([], dtype=torch.float32),
            "a": torch.tensor([], dtype=torch.float32),
            "r": torch.tensor([], dtype=torch.float32),
            "terminal": torch.tensor([], dtype=torch.float32),
            "s_": torch.tensor([], dtype=torch.float32),
        }

    def store(self, s, a, r, terminal, s_):
        """Store experience tuple.

        Args:
            s (numpy.ndarray): State.
            a (numpy.ndarray): Action.
            r (numpy.ndarray): Reward.
            terminal (numpy.ndarray): Whether the terminal state was reached.
            s_ (numpy.ndarray): Next state.

        Returns:
            int: The current memory buffer size.
        """

        # Store experience in memory buffer
        transition = {
            "s": torch.as_tensor(s, dtype=torch.float32).to(self.device),
            "a": torch.as_tensor(a, dtype=torch.float32).to(self.device),
            "r": torch.as_tensor([r], dtype=torch.float32).to(self.device),
            "terminal": torch.as_tensor([terminal], dtype=torch.float32).to(
                self.device
            ),
            "s_": torch.as_tensor(s_, dtype=torch.float32).to(self.device),
        }
        if len(self.current_path["s"]) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key].unsqueeze(dim=0)
        else:
            for key in transition.keys():
                self.current_path[key] = torch.cat(
                    [self.current_path[key], transition[key].unsqueeze(dim=0)], axis=0
                )  # FIXME: This needs to be fixed in old version (Always torch.cat)
        if terminal == 1.0:
            # FIXME: DIFFERENCE WITH SPINNINGUP
            # NOTE: WHY the hell only update when Paths are terminal? Done because
            # evaluation is on path basis?
            for key in self.current_path.keys():
                self.memory[key] = torch.cat(
                    [self.memory[key], self.current_path[key]], axis=0
                )  # DEBUG
            self.paths.appendleft(self.current_path)
            self.reset()
            self.memory_pointer = len(self.memory["s"])

        # Return current memory buffer size
        return self.memory_pointer

    def sample(self, batch_size):
        """Sample from memory buffer.

        Args:
            batch_size (int): The memory buffer sample size.

        Returns:
            numpy.ndarray: The batch of experiences.
        """
        if self.memory_pointer < self.min_memory_size:
            return None
        else:

            # Sample a random batch of experiences
            indices = np.random.choice(
                min(self.memory_pointer, self.memory_capacity) - 1,
                size=batch_size,
                replace=False,
            ) + max(1, 1 + self.memory_pointer - self.memory_capacity) * np.ones(
                [batch_size], np.int
            )
            batch = {}
            for key in self.memory.keys():
                if "s" in key:
                    sample = self.memory[key][indices]  # DEBUG
                    batch.update({key: sample})
                else:
                    batch.update({key: self.memory[key][indices]})  # DEBUG:
            return batch