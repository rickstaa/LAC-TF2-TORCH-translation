"""Contains the Lapunov critic.
"""

import tensorflow as tf
from tensorflow import nn

from utils import mlp


class LyapunovCritic(tf.keras.Model):
    """Soft Lyapunov critic Network.

    Attributes:
        lya (tf.keras.Sequential): The layers of the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.relu,
        output_activation=nn.relu,  # DEPLOY: Put back to identity when deploy
        name="lyapunov_critic",
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

            name (str, optional): The Lyapunov critic name. Defaults to
                "lyapunov_critic".
        """
        super().__init__(name=name, **kwargs)
        self.lya = mlp(
            [obs_dim + act_dim] + list(hidden_sizes),
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
            tf.Tensor: The tensor containing the Lyapunov values of the input
                observations and actions.
        """
        # DEPLOY: Make squaring layer from class so it shows up named in the graph!
        net_out = self.lya(tf.concat([obs, act], axis=-1))
        return tf.expand_dims(
            tf.reduce_sum(tf.math.square(net_out), axis=1), axis=1
        )  # L(s,a)
