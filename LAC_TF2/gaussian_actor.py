"""Contains the Gaussian actor.
"""

import tensorflow as tf
from tensorflow import nn
import tensorflow_probability as tfp

from squash_bijector import SquashBijector

from utils import mlp, clamp

# Script parameters
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedGaussianActor(tf.keras.Model):
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.modules.container.Sequential): The fully connected hidden layers
            of the network.

        mu (tf.keras.Sequential): The output layer which returns the mean of the
            actions.

        log_sigma (tf.keras.Sequential): The output layer which returns the log standard
            deviation of the actions.

        act_limits (dict, optional): The "high" and "low" action bounds of the
            environment. Used for rescaling the actions that comes out of network from
            (-1, 1) to (low, high).
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        act_limits=None,
        activation=nn.relu,
        output_activation=nn.relu,
        name="gaussian_actor",
        **kwargs
    ):
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            activation (function): The hidden layer activation function.

            output_activation (function, optional): The activation function used for
                the output layers. Defaults to tf.keras.activations.linear.

            act_limits (dict or , optional): The "high" and "low" action bounds of the
                environment. Used for rescaling the actions that comes out of network
                from (-1, 1) to (low, high). Defaults to (-1, 1).

            name (str, optional): The Lyapunov critic name. Defaults to
                "guassian_actor".
        """
        super().__init__(name=name, **kwargs)

        # Set class attributes
        self.act_limits = act_limits

        # Create squash bijector, and normal distribution (Used in the
        # re-parameterization trick)
        self._squash_bijector = SquashBijector()
        self._normal_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(act_dim), scale_diag=tf.ones(act_dim)
        )

        # Create networks
        self.net = mlp(
            [obs_dim] + list(hidden_sizes), activation, output_activation, name=name,
        )
        self.mu = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    act_dim,
                    input_shape=(hidden_sizes[-1],),
                    activation=None,
                    name=name + "/mu",
                )
            ]
        )
        self.log_std = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    act_dim,
                    input_shape=(hidden_sizes[-1],),
                    activation=None,
                    name=name + "/log_std",
                )
            ]
        )

    @tf.function
    def call(self, obs, deterministic=False, with_logprob=True):
        """Performs forward pass through the network.

        Args:
            obs (numpy.ndarray): The tensor of observations.

            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to False.

            with_logprob (bool, optional): Whether we want to return the log probability
                of an action. Defaults to True.

        Returns:
            tf.Tensor,  tf.Tensor: The actions given by the policy, the log
                probabilities of each of these actions.
        """

        # Calculate mean action and standard deviation
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)  # Transform to standard deviation

        # Create affine bijector (Used in the re-parameterization trick)
        affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))

        # Use the re-parameterization trick to sample a action from the pre-squashed
        # distribution
        if deterministic:
            pi_action = mu  # Determinstic action used at test time.
        else:
            # Sample from the normal distribution and calculate the action
            batch_size = tf.shape(input=obs)[0]
            epsilon = self._normal_distribution.sample(batch_size)
            pi_action = affine_bijector.forward(
                epsilon
            )  # Transform action as it was sampled from the policy distribution

        # Squash the action between (-1 and 1)
        pi_action = self._squash_bijector.forward(pi_action)

        # Compute log probability of the sampled action in  the squashed gaussian
        if with_logprob:

            # Transform base_distribution to the policy distribution
            reparm_trick_bijector = tfp.bijectors.Chain(
                (self._squash_bijector, affine_bijector)
            )
            pi_distribution = tfp.distributions.TransformedDistribution(
                distribution=self._normal_distribution, bijector=reparm_trick_bijector
            )
            logp_pi = pi_distribution.log_prob(pi_action)
        else:
            logp_pi = None

        #  Clamp the actions such that they are in range of the environment
        if self.act_limits:
            pi_action = clamp(
                pi_action,
                min_bound=self.act_limits["low"],
                max_bound=self.act_limits["high"],
            )

        # Return action and log likelihood
        return pi_action, logp_pi
