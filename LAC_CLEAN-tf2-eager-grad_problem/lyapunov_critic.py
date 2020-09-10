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
        trainable=True,
        log_std_min=-20,
        log_std_max=2.0,
        **kwargs
    ):
        super().__init__(name=name, trainable=trainable, **kwargs)

        # Get class parameters
        self.s_dim = obs_dim
        self.a_dim = act_dim

        # Create input layer weights and biases
        # FIXME: Check if same initializer is used in pytorch
        self.w1_s = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=(self.s_dim, hidden_sizes[0])),
            name="w1_s",
            trainable=trainable,
        )
        self.w1_a = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=(self.s_dim, hidden_sizes[0])),
            name="w1_a",
            trainable=trainable,
        )
        self.b1 = tf.Variable(
            tf.zeros_initializer()(shape=(1, hidden_sizes[0])),
            name="b1",
            trainable=trainable,
        )

        # Create fully connected layers
        layers = []
        for i in range(1, len(hidden_sizes)):
            n = hidden_sizes[i]
            layers.append(
                tf.keras.layers.Dense(
                    n,
                    activation="relu",
                    name="l" + str(i + 1),
                    trainable=trainable,
                )
            )
            layers[i - 1].build(input_shape=(hidden_sizes[i - 1]))  # Init weights
        self.net = tf.keras.Sequential(layers)

    def call(self, inputs, training=False):
        """Perform forward pass."""

        # Retrieve inputs
        obs = inputs[0]
        acts = inputs[1]

        # Perform forward pass through input layers
        # TODO: make sure training is correct
        net_out = tf.nn.relu(
            tf.matmul(tf.convert_to_tensor(obs), self.w1_s)
            + tf.matmul(tf.convert_to_tensor(acts), self.w1_a)
            + self.b1
        )

        # Pass through fully connected layers
        net_out = self.net(net_out, training=training)

        # Return result
        return tf.expand_dims(
            tf.reduce_sum(tf.square(net_out), axis=1), axis=1
        )  # L(s,a)
