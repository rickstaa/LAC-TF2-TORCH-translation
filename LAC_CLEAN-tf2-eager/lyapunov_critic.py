"""Contains the tensorflow critic.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector


class LyapunovCritic(tf.keras.Model):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        name,
        log_std_min=-20,
        log_std_max=2.0,
        trainable=True,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Get class parameters
        self.s_dim = obs_dim
        self.a_dim = act_dim

        # Create input layer weights and biases
        # TODO: Remove or fix trainable
        self.base = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32,
                    input_shape=(obs_dim + act_dim),
                    name=name + "/input",
                )
            ]
        )
        for i, hidden_size_i in enumerate(hidden_sizes):
            self.base.add(
                tf.keras.layers.Dense(
                    hidden_size_i, activation="relu", name=name + "/l{}".format(i),
                )
            )

    def call(self, input_tensor):
        """Perform forward pass."""

        # Perform forward pass through fully connected layers
        net_out = self.base(input_tensor)

        # Return result
        return tf.expand_dims(
            tf.reduce_sum(tf.square(net_out), axis=1), axis=1
        )  # L(s,a)
