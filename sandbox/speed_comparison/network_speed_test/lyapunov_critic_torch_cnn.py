"""Contains an in Pytorch implemented LYapunov Critic.
"""

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, obs_dim, act_dim, hidden_sizes, cnn_output_size=18):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
        """
        super().__init__()
        self._conv_output = cnn_output_size
        self._cnn_output = [self._conv_output] + [int(item / 2) for item in obs_dim[1:]]

        # CNN networks
        self.conv1 = torch.nn.Conv2d(
            obs_dim[0], self._conv_output, kernel_size=3, stride=1, padding=1
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # MLP
        self.l = mlp(
            [reduce(lambda x, y: x * y, self._cnn_output) + act_dim]
            + list(hidden_sizes),
            nn.ReLU,
            nn.ReLU,
        )

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the lyapunov values of the input
                observations and actions.
        """

        # CNN pass
        x = F.relu(self.conv1(obs))
        x = self.pool(x)
        x = x.view(-1, reduce(lambda x, y: x * y, self._cnn_output))

        # Further pass
        l_out = self.l(torch.cat([x, act], dim=-1))
        l_out_squared = torch.square(l_out)
        l_out_summed = torch.sum(l_out_squared, dim=1)
        return l_out_summed.unsqueeze(dim=1)
