"""Contains an in Tensorflow implemented LYapunov Critic.
"""


import tensorflow as tf


class LyapunovCritic(tf.keras.Model):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        name="test",
        log_std_min=-20,
        log_std_max=2.0,
        trainable=True,
    ):
        """Lyapunov Critic network.

        Args:
            obs_dim (int): The dimension of the observation space.

            act_dim (int): The dimension of the action space.

            hidden_sizes (list): Array containing the sizes of the hidden layers.

        """
        super().__init__()

        # Get class parameters
        self.s_dim = obs_dim
        self.a_dim = act_dim

        # Create network layers
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=(obs_dim + act_dim), name="input",
                )
            ]
        )
        for i, hidden_size_i in enumerate(hidden_sizes):
            self.net.add(
                tf.keras.layers.Dense(
                    hidden_size_i,
                    activation="relu",
                    name="LyapunovCritic" + "/{}".format(i),
                    trainable=True,
                )
            )

    @tf.function
    def call(self, inputs):
        """Perform forward pass."""
        net_out = self.net(tf.concat(inputs, axis=-1))
        return tf.expand_dims(
            tf.reduce_sum(tf.math.square(net_out), axis=1), axis=1
        )  # L(s,a)
