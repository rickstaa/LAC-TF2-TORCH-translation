"""Contains the tensorflow critic.
"""

import tensorflow as tf
from tensorflow import nn

from utils import mlp


class QCritic(tf.keras.Model):
    """Soft Q critic network.

    Attributes:
        q (tf.keras.Sequential): The layers of the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.relu,
        output_activation=None,
        name="q_critic",
        **kwargs,
    ):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            activation (function): The hidden layer activation function.

            output_activation (function, optional): The activation function used for
                the output layers. Defaults to tf.keras.activations.linear.

            name (str, optional): The Q-Critic name. Defaults to "q_critic".
        """
        super().__init__(name=name, **kwargs)
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            activation,
            output_activation,
            name=name,
        )

    @tf.function
    def call(self, obs, act):
        """Performs forward pass through the network.

        Args:
            obs (tf.Tensor): The tensor of observations.

            act (tf.Tensor): The tensor of actions.

        Returns:
            tf.Tensor: The tensor containing the Q values of the input observations
                and actions.
        """
        return self.q(tf.concat([obs, act], axis=-1))  # Q(s,a)
