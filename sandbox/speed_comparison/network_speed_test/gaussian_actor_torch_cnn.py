"""Contains an in Pytorch implemented Squashed Gaussian Actor.
"""

from functools import reduce

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
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        cnn_output_size=18,
        activation=nn.ReLU,
        log_std_min=-20,
        log_std_max=2.0,
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
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self._obs_dim = obs_dim
        self._conv_output = cnn_output_size
        self._cnn_output = [self._conv_output] + [int(item / 2) for item in obs_dim[1:]]

        # CNN networks
        self.conv1 = torch.nn.Conv2d(
            obs_dim[0], self._conv_output, kernel_size=3, stride=1, padding=1
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # MLP networks
        self.net = mlp(
            [reduce(lambda x, y: x * y, self._cnn_output)] + list(hidden_sizes),
            activation,
            activation,
        )
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

        # CNN pass
        x = F.relu(self.conv1(obs))
        x = self.pool(x)
        x = x.view(-1, reduce(lambda x, y: x * y, self._cnn_output))

        # Further pass
        net_out = self.net(x)
        mu = self.mu(net_out)
        log_std = self.log_sigma(net_out)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        std = torch.exp(log_std)

        # Check summing axis
        sum_axis = 0 if obs.shape.__len__() == 1 else 1

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        pi_action = (
            pi_distribution.rsample()
        )  # Sample while using the parameterization trick

        # Compute logprob from Gaussian, and then apply correction for Tanh
        # squashing. NOTE: The correction formula is a little bit magic. To get an
        # understanding of where it comes from, check out the original SAC paper
        # (arXiv 1801.01290) and look in appendix C. This is a more
        # numerically-stable equivalent to Eq 21. Try deriving it yourself as a
        # (very difficult) exercise. :)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
        #     axis=sum_axis
        # ) # DEBUG

        # Calculate scaled action and return the action and its log probability
        clipped_mu = torch.tanh(mu)
        pi_action = torch.tanh(pi_action)  # Squash gaussian to be between -1 and 1

        # Return action and log likelihood
        return pi_action, clipped_mu, logp_pi  # Here I return two times the
