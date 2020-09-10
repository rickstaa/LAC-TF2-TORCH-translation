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
        self.net_0 = tf.keras.layers.Dense(
            hidden_sizes[0], activation="relu", name="l1", trainable=trainable
        )
        self.net_0.build(input_shape=(self.s_dim))
        self.net_0.trainable = trainable
        self.net_1 = tf.keras.layers.Dense(
            hidden_sizes[1], activation="relu", name="l4", trainable=trainable
        )
        self.net_1.build(input_shape=(hidden_sizes[0]))
        self.net_1.trainable = trainable

        # Create Mu and log sigma output layers
        self.mu = tf.keras.layers.Dense(act_dim, name="a", trainable=trainable)
        self.mu.build(input_shape=(hidden_sizes[1]))
        self.mu.trainable = trainable
        self.log_sigma = tf.keras.layers.Dense(act_dim, trainable=trainable)
        self.log_sigma.build(input_shape=(hidden_sizes[1]))
        self.log_sigma.trainable = trainable

    def call(self, inputs, training=False):
        """Perform forward pass."""

        # Retrieve inputs
        obs = inputs

        # Perform forward pass through fully connected layers
        # TODO: VAlidate training variable
        net_out = self.net_0(obs, training=training)
        net_out = self.net_1(net_out, training=training)

        # Calculate mu and log_sigma
        mu = self.mu(net_out, training=training)
        log_sigma = self.mu(net_out, training=training)
        # TEST: Validate clipping
        log_sigma = tf.clip_by_value(log_sigma, self._log_std_min, self._log_std_max)

        # Perform reparameterization trick
        sigma = tf.exp(log_sigma)

        # Create bijectors (Used in the reparameterization trick)
        squash_bijector = SquashBijector()
        affine_bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)

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
