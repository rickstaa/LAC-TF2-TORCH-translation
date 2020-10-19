"""Contains the Q-critic.
"""

import torch
from torch import nn
from utils import mlp


# TODO: Update docstring
class QCritic(nn.Module):
    """Soft Q critic network.

    Attributes:
        q (torch.nn.modules.container.Sequential): The layers of the network.
    """

    # TODO: ADD SEEDING:
    def __init__(
        self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, use_fixed_seed=False
    ):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (torch.nn.modules.activation): The activation function.
        """
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the Q values of the input observations
                and actions.
        """
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
