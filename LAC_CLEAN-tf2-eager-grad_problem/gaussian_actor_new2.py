"""Contains the tensorflow actor.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector


class SquashedGaussianActor(tf.keras.Model):
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
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self.s_dim = obs_dim
        self.a_dim = act_dim

        # Create fully connected layers
        self.base = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=(self.s_dim), name=name + "/input"
                )
            ]
        )
        for i, hidden_size_i in enumerate(hidden_sizes):
            self.base.add(
                tf.keras.layers.Dense(
                    hidden_size_i,
                    activation="relu",
                    name=name + "/l{}".format(i),
                    trainable=trainable,
                )
            )

        # Create Mu and log sigma output layers
        self.mu = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1]
                ),
                tf.keras.layers.Dense(
                    act_dim, activation=None, name=name + "mu", trainable=trainable
                ),
            ]
        )
        self.log_std = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1]
                ),
                tf.keras.layers.Dense(
                    act_dim, activation=None, name=name + "log_std", trainable=trainable
                ),
            ]
        )

    @tf.function
    def call(self, input_tensor, training=False):
        """Perform forward pass."""

        # Retrieve inputs
        obs = inputs

        # Perform forward pass through fully connected layers
        # TODO: Validate trianing
        net_out = self.base(input_tensor, training=training)

        # Calculate mu and log_std
        mu = self.mu(net_out)
        log_std = tf.clip_by_value(
            self.log_std(net_out, training=training),
            self._log_std_min,
            self._log_std_max,
        )

        # Calculate mu and log_sigma
        mu = self.mu(net_out, training=training)
        log_sigma = self.mu(net_out, training=training)

        # Perform re-parameterization trick
        std = tf.exp(log_sigma)

        # Create bijectors (Used in the re-parameterization trick)
        squash_bijector = SquashBijector()
        affine_bijector = tfp.bijectors.Affine(shift=mu, scale_diag=std)

        # Sample from the normal distribution and calculate the action
        batch_size = tf.shape(input=obs).numpy()[0]
        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
        )
        epsilon = base_distribution.sample(batch_size)
        raw_action = affine_bijector.forward(epsilon)
        clipped_a = squash_bijector.forward(raw_action)

        # Transform distribution back to the original policy distribution
        reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
        distribution = tfp.distributions.TransformedDistribution(
            distribution=base_distribution, bijector=reparm_trick_bijector
        )
        clipped_mu = squash_bijector.forward(mu)

        return clipped_a, clipped_mu, distribution
