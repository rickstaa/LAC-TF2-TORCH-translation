"""Contains the pytorch lyapunov critic. I first tried to create this as a
sequential model using the
`torch.nn.Sequential class <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_
but the network unfortunately is to difficult (Uses Square in the output).
"""

import torch
import torch.nn as nn

from utils import mlp


class MLPLyapunovCritic(nn.Module):
    """Soft Lyapunov critic Network.

    Attributes:
        q (torch.nn.modules.container.Sequential): The layers of the network.
    """

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
        # TODO: Make squaring layer from class so it shows up named in the graph!
        # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        l_out = self.l(torch.cat([obs, act], dim=-1))
        l_out_squared = torch.square(l_out)
        l_out_summed = torch.sum(l_out_squared, dim=1)
        return l_out_summed.unsqueeze(dim=1)
