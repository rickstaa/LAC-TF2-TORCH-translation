"""Contains the tensorflow critic.
"""

import tensorflow as tf


class LyapunovCritic(tf.keras.Model):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        name,
        log_std_min=-20,
        log_std_max=2.0,
        **kwargs
    ):
        # TODO: Check if name results in problem
        super().__init__(name=name, **kwargs)

        # Get class parameters
        self.s_dim = obs_dim
        self.a_dim = act_dim

        # Create input layer weights and biases
        # FIXME: Cleanup build!
        # FIXME: Check if same initializer is used in pytorch
        self.w1_s = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=(self.s_dim, hidden_sizes[0])),
            name="w1_s",
        )
        self.w1_a = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=(self.s_dim, hidden_sizes[0])),
            name="w1_a",
        )
        self.b1 = tf.Variable(
            tf.zeros_initializer()(shape=(1, hidden_sizes[0])), name="b1",
        )

        # Create fully connected layers
        # TODO: Check if this is right!
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=(hidden_sizes[0]), name=name + "/input",
                )
            ]
        )
        for i in range(1, len(hidden_sizes)):
            n = hidden_sizes[i]
            self.net.add(
                tf.keras.layers.Dense(n, activation="relu", name="l" + str(i + 1),)
            )

    @tf.function
    def call(self, inputs):
        """Perform forward pass."""

        # Retrieve inputs
        obs = inputs[0]
        acts = inputs[1]

        # Perform forward pass through input layers
        net_out = tf.nn.relu(
            tf.matmul(obs, self.w1_s) + tf.matmul(acts, self.w1_a) + self.b1
        )

        # Pass through fully connected layers
        net_out = self.net(net_out)

        # Return result
        return tf.expand_dims(
            tf.reduce_sum(tf.math.square(net_out), axis=1), axis=1
        )  # L(s,a)
