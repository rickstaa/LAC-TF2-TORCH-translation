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
        log_std_min=-20,
        log_std_max=2.0,
        trainable=True,
        seeds=None,
        **kwargs
    ):
        """Squashed Gaussian actor network.

        Args:
            obs_dim (int): The dimension of the observation space.

            act_dim (int): The dimension of the action space.

            hidden_sizes (list): Array containing the sizes of the hidden layers.

            name (str): The keras module name.

            trainable (bool, optional): Whether the weights of the network layers should
                be trainable. Defaults to True.

            seeds (list, optional): The random seeds used for the weight initialization
                and the sampling ([weights_seed, sampling_seed]). Defaults to
                [None, None]
        """
        super().__init__(name=name, **kwargs)

        # Get class parameters
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self.s_dim = obs_dim
        self.a_dim = act_dim
        self._seed = seeds[0]
        self._initializer = tf.keras.initializers.GlorotUniform(
            seed=self._seed
        )  # Seed weights initializer
        self._tfp_seed = seeds[1]

        # Create fully connected layers
        self.net_0 = tf.keras.layers.Dense(
            hidden_sizes[0], activation="relu", name="l1"
        )
        self.net_0.build(input_shape=(self.s_dim))
        self.net_1 = tf.keras.layers.Dense(
            hidden_sizes[1], activation="relu", name="l2"
        )
        self.net_1.build(input_shape=(hidden_sizes[0]))

        # Create Mu and log sigma output layers
        self.mu = tf.keras.layers.Dense(act_dim, name="mu", activation=None)
        self.mu.build(input_shape=(hidden_sizes[1]))
        self.log_sigma = tf.keras.layers.Dense(
            act_dim, name="log_sigma", activation=None
        )
        self.log_sigma.build(input_shape=(hidden_sizes[1]))

    @tf.function
    def call(self, inputs):
        """Perform forward pass."""

        # Retrieve inputs
        obs = inputs

        # Perform forward pass through fully connected layers
        net_out = self.net_0(obs)
        net_out = self.net_1(net_out)

        # Calculate mu and log_sigma
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = tf.clip_by_value(log_sigma, self._log_std_min, self._log_std_max)

        # Perform re-parameterization trick
        sigma = tf.exp(log_sigma)

        # Create bijectors (Used in the re-parameterization trick)
        squash_bijector = SquashBijector()
        affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(sigma))

        # Sample from the normal distribution and calculate the action
        batch_size = tf.shape(input=obs)[0]
        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
        )
        epsilon = base_distribution.sample(batch_size, seed=self._tfp_seed)
        raw_action = affine_bijector.forward(epsilon)
        clipped_a = squash_bijector.forward(raw_action)

        # Transform distribution back to the original policy distribution
        reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
        distribution = tfp.distributions.TransformedDistribution(
            distribution=base_distribution, bijector=reparm_trick_bijector
        )
        clipped_mu = squash_bijector.forward(mu)
        return clipped_a, clipped_mu, distribution.log_prob(clipped_a)
