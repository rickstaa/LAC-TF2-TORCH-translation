"""Contains the Lyapunov Critic Class.
"""

import torch
import torch.nn as nn

from utils import mlp

# FIXME: Check weight initialization based on main random seed


class MLPLyapunovCritic(nn.Module):
    """Soft Lyapunov critic Network.

    Attributes:
        l (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            activation (torch.nn.modules.activation): The activation function.
        """
        # TODO: UPDATE DOCSTRING
        super().__init__()
        self.l = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation)

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the lyapunov values of the input
                observations and actions.
        """
        # IMPROVEMENT: Make squaring layer from class so it shows up named in the graph!
        # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        l_out = self.l(torch.cat([obs, act], dim=-1))
        l_out_squared = torch.square(l_out)
        l_out_summed = torch.sum(l_out_squared, dim=1)
        return l_out_summed.unsqueeze(dim=1)
