"""Contains the replay buffer class.
"""

from collections import deque

import numpy as np


class Pool(object):
    """Memory buffer class.

    Attributes:
        self.memory_capacity (int): The current memory capacity.

        paths (collections.deque): A storage bucket for storing paths.

        memory (dict): The replay memory storage bucket.

        min_memory_size (np.float32): The minimum memory size before we start to sample
            from the memory buffer.

        memory_pointer (): The number of experiences that are currently stored in the
            replay buffer.
    """

    def __init__(
        self, s_dim, a_dim, memory_capacity, store_last_n_paths, min_memory_size,
    ):
        """Initializes memory buffer object.

        Args:
            s_dim (int): The observation space dimension.

            a_dim (int): The action space dimension.

            memory_capacity (int): The size of the memory buffer.

            store_last_n_paths (int): How many paths you want to store in the replay
                buffer.

            min_memory_size (int): The minimum memory size before we start to sample
                from the memory buffer.
        """
        self.memory_capacity = memory_capacity
        self.paths = deque(maxlen=store_last_n_paths)  # TODO: Why is this used?
        self.reset()
        self.memory = {
            "s": np.zeros([1, s_dim], dtype=np.float32),
            "a": np.zeros([1, a_dim], dtype=np.float32),
            "r": np.zeros([1, 1], dtype=np.float32),
            "terminal": np.zeros([1, 1], dtype=np.float32),
            "s_": np.zeros([1, s_dim], dtype=np.float32),
        }
        self.memory_pointer = 0
        self.min_memory_size = min_memory_size

    def reset(self):
        """Reset memory buffer.
        """
        self.current_path = {
            "s": [],
            "a": [],
            "r": [],
            "terminal": [],
            "s_": [],
        }

    def store(self, s, a, r, terminal, s_):
        """Stores experience tuple.

        Args:
            s (numpy.ndarray): State.

            a (numpy.ndarray): Action.

            r (numpy.ndarray): Reward.

            terminal (numpy.ndarray): Whether the terminal state was reached.

            s_ (numpy.ndarray): Next state.

        Returns:
            int: The current memory buffer size.
        """
        transition = {
            "s": np.array(s, dtype=np.float32),
            "a": np.array(a, dtype=np.float32),
            "r": np.array([r], dtype=np.float32),
            "terminal": np.array([terminal], dtype=np.float32),
            "s_": np.array(s_, dtype=np.float32),
        }
        if len(self.current_path["s"]) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key][np.newaxis, :]
        else:
            for key in transition.keys():
                self.current_path[key] = np.concatenate(
                    (self.current_path[key], transition[key][np.newaxis, :])
                )
        if terminal == 1.0:
            # Question (rickstaa): Why only update when Paths are terminal?
            # evaluation is on path basis?
            for key in self.current_path.keys():
                self.memory[key] = np.concatenate(
                    (self.memory[key], self.current_path[key]), axis=0
                )
            self.paths.appendleft(self.current_path)
            self.reset()
            self.memory_pointer = len(self.memory["s"])

        # Return current memory buffer size
        return self.memory_pointer

    def sample(self, batch_size):
        """Samples from memory buffer.

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
                    sample = self.memory[key][indices]
                    batch.update({key: sample})
                else:
                    batch.update({key: self.memory[key][indices]})
            return batch
