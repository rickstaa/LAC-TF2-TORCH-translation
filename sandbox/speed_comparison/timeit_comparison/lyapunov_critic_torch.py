"""Contains an in Pytorch implemented LYapunov Critic.
"""

import torch
import torch.nn as nn

from utils_torch import mlp


def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a multi-layered perceptron using pytorch.

    Args:
        sizes (list): The size of each of the layers.

        activation (torch.nn.modules.activation): The activation function used for the
            hidden layers.

        output_activation (torch.nn.modules.activation, optional): The activation
            function used for the output layers. Defaults to torch.nn.Identity.

    Returns:
        torch.nn.modules.container.Sequential: The multi-layered perceptron.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPLyapunovCritic(nn.Module):
    """Soft Lyapunov critic Network.

    Attributes:
        q (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
        """
        super().__init__()
        self.l = mlp([obs_dim + act_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the lyapunov values of the input
                observations and actions.
        """
        l_out = self.l(torch.cat([obs, act], dim=-1))
        l_out_squared = torch.square(l_out)
        l_out_summed = torch.sum(l_out_squared, dim=1)
        return l_out_summed.unsqueeze(dim=1)
