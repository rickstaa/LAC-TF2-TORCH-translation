"""Contains the tensorflow actor.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector

# @tf.function
def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(
        input_tensor=2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1
    )

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


class SquashedGaussianActor(tf.keras.Model):
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
        super().__init__(name=name, **kwargs)

        # Get class parameters
        # TODO: Remove or fix trainable
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self.s_dim = obs_dim
        self.a_dim = act_dim

        # Create fully connected layers
        self.base = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=(obs_dim), name=name + "/input"
                )
            ]
        )
        for i, hidden_size_i in enumerate(hidden_sizes):
            self.base.add(
                tf.keras.layers.Dense(
                    hidden_size_i,
                    activation="relu",
                    name=name + "/l{}".format(i),
                )
            )

        # Create Mu and log sigma output layers
        self.mu = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1],
                ),
                tf.keras.layers.Dense(
                    act_dim, activation=None, name=name + "mu",
                ),
            ],
        )
        self.log_std = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1]
                ),
                tf.keras.layers.Dense(
                    act_dim, activation=None, name=name + "log_std",
                ),
            ]
        )

    # @tf.function
    def call(self, input_tensor):

        # Get current policy mean and std
        # TODO: Check training
        h = self.base(input_tensor)
        mu = self.mu(h)
        log_std = tf.clip_by_value(
            self.log_std(h), self._log_std_min, self._log_std_max
        )
        std = tf.exp(log_std)

        # Create
        policy_dist = tfp.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=std
        )
        # policy_dist = tfp.distributions.MultivariateNormalDiag(
        #     loc=mu, scale_diag=std ** 2
        # ) # TODO: Why std ** 2
        pi = policy_dist.sample()
        logp_pi = policy_dist.log_prob(pi)
        # pi = mu + tf.random.normal(tf.shape(input=mu)) * std
        # logp_pi = gaussian_likelihood(pi, mu, log_std)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        return pi, mu, logp_pi
