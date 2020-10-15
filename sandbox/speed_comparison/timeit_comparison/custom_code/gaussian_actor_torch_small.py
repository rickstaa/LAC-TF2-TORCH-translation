"""Contains an in Pytorch implemented Squashed Gaussian Actor.
"""

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.normal import Normal


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


class SquashedGaussianMLPActor(nn.Module):
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.modules.container.Sequential): The input/hidden layers of the
            network.

        mu (torch.nn.modules.linear.Linear): The output layer which returns the mean of
            the actions.

        log_sigma (torch.nn.modules.linear.Linear): The output layer which returns
            the log standard deviation of the actions.
    """

    def __init__(
        self, obs_dim, act_dim, hidden_sizes, log_std_min=-20, log_std_max=2.0,
    ):
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            log_std_min (int, optional): The minimum log standard deviation. Defaults
                to -20.

            log_std_max (float, optional): The maximum log standard deviation. Defaults
                to 2.0.
        """
        super().__init__()
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Create networks
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_sigma = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

        Returns:
            torch.Tensor,  torch.Tensor: The actions given by the policy, the log
            probabilities of each of these actions.
        """
        net_out = self.net(obs)
        _ = self.mu(net_out)
        log_std = self.log_sigma(net_out)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        return _, _, _
