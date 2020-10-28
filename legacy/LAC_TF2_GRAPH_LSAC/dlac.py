"""Minimal working version of the LAC algorithm script.
"""

import time
from collections import deque
import sys
import os
import random
from packaging import version

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector
from utils import evaluate_training_rollouts, get_env_from_name, training_evaluation
import logger
from pool import Pool

# check which Tensorflow version is used and load weight initializer
if version.parse(tf.__version__) > version.parse("1.15.4"):
    TF_115 = False
    from tensorflow.keras.initializers import GlorotUniform
else:
    TF_115 = True
    from tensorflow.python.keras.initializers import GlorotUniform

###############################################
# Script settings #############################
###############################################
from variant import (
    USE_GPU,
    ENV_NAME,
    RANDOM_SEED,
    ENV_SEED,
    TRAIN_PARAMS,
    ALG_PARAMS,
    ENV_PARAMS,
    LOG_SIGMA_MIN_MAX,
    SCALE_lambda_MIN_MAX,
)

# Set random seed to get comparable results for each run
if RANDOM_SEED is not None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if TF_115:
        tf.random.set_random_seed(RANDOM_SEED)
    else:
        tf.compat.v1.random.set_random_seed(RANDOM_SEED)
    TFP_SEED_STREAM = tfp.util.SeedStream(RANDOM_SEED, salt="tfp_1")

# Display training info
print("==Train settings==")

# Disable GPU if requested
# NOTE (rickstaa): Done to support both tf1.x and tf2.x.
if not USE_GPU:
    if version.parse(tf.__version__) > version.parse("1.15.4"):
        tf.config.set_visible_devices([], "GPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Tensorflow is using CPU.")
else:
    print("Tensorflow is using GPU.")

# Print LAC/SAC message
if ALG_PARAMS["use_lyapunov"]:
    print("You are training the LAC algorithm.")
else:
    print("You are training the SAC algorithm.")

# Disable eager
tf.compat.v1.disable_eager_execution()


###############################################
# LAC algorithm class #########################
###############################################
class LAC(object):
    """The lyapunov actor critic.

    """

    def __init__(self, a_dim, s_dim):
        """Initiate object state.

        Args:
            a_dim (int): Action space dimension.
            s_dim (int): Observation space dimension.
        """

        # Save action and observation space as members
        self.a_dim = a_dim
        self.s_dim = s_dim

        # Set algorithm parameters as class objects
        self.network_structure = ALG_PARAMS["network_structure"]
        self.polyak = 1 - ALG_PARAMS["tau"]
        self.use_lyapunov = ALG_PARAMS["use_lyapunov"]

        # Create network seeds
        if RANDOM_SEED is not None:
            self.ga_seeds = [
                RANDOM_SEED,
                TFP_SEED_STREAM(),
            ]  # [weight init seed, sample seed]
            self.ga_target_seeds = [
                RANDOM_SEED + 1,
                TFP_SEED_STREAM(),
            ]  # [weight init seed, sample seed]
            # self.lya_ga_target_seeds = [
            #     RANDOM_SEED,
            #     TFP_SEED_STREAM(),
            # ]  # [weight init seed, sample seed]
            self.lc_seed1 = RANDOM_SEED + 2  # Weight init seed
            self.lc_seed2 = RANDOM_SEED + 3  # Weight init seed
            self.lc_target_seed = RANDOM_SEED + 4  # Weight init seed
            self.qc_seed1 = RANDOM_SEED + 5  # Weight init seed
            self.qc_seed2 = RANDOM_SEED + 6  # Weight init seed
            self.qc_target_seed1 = RANDOM_SEED + 7  # Weight init seed
            self.qc_target_seed2 = RANDOM_SEED + 8  # Weight init seed
        else:
            self.ga_seeds = None
            self.ga_target_seeds = None
            self.lc_seed1 = None
            self.lc_seed2 = None
            self.lc_target_seed = None
            self.qc_seed1 = None
            self.qc_seed2 = None
            self.qc_target_seed1 = None
            self.qc_target_seed2 = None

        # Determine target entropy
        if ALG_PARAMS["target_entropy"] is None:
            self.target_entropy = -self.a_dim  # lower bound of the policy entropy
        else:
            self.target_entropy = ALG_PARAMS["target_entropy"]

        # Create tensorflow session
        self.sess = tf.compat.v1.Session()

        # Create networks, optimizers and variables inside the Actor scope
        with tf.compat.v1.variable_scope("Actor"):

            # Create observations placeholders
            self.S = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], "s")
            self.S_ = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], "s_")
            self.a_input = tf.compat.v1.placeholder(
                tf.float32, [None, self.a_dim], "a_input"
            )
            self.a_input_ = tf.compat.v1.placeholder(
                tf.float32, [None, self.a_dim], "a_input_"
            )
            self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], "r")
            self.terminal = tf.compat.v1.placeholder(tf.float32, [None, 1], "terminal")

            # Create Learning rate placeholders
            self.LR_A = tf.compat.v1.placeholder(tf.float32, None, "LR_A")
            self.LR_lag = tf.compat.v1.placeholder(tf.float32, None, "LR_lag")
            self.LR_L = tf.compat.v1.placeholder(tf.float32, None, "LR_L")
            self.LR_C = tf.compat.v1.placeholder(tf.float32, None, "LR_C")

            # Create lagrance multiplier placeholders
            self.log_labda = tf.compat.v1.get_variable(
                "log_lambda",
                None,
                tf.float32,
                initializer=tf.math.log(ALG_PARAMS["labda"]),
            )
            self.log_alpha = tf.compat.v1.get_variable(
                "log_alpha",
                None,
                tf.float32,
                initializer=tf.math.log(ALG_PARAMS["alpha"]),
            )
            self.labda = tf.clip_by_value(tf.exp(self.log_labda), *SCALE_lambda_MIN_MAX)
            self.alpha = tf.exp(self.log_alpha)

            ###########################################
            # Create Networks #########################
            ###########################################

            # Create Gaussian Actor (GA), Lyapunov critic (LC) and Q Critic Networks
            self.a, self.deterministic_a, self.a_dist = self._build_a(
                self.S, seeds=self.ga_seeds
            )
            self.l1 = self._build_l(
                self.S, self.a_input, name="lyapunov_critic_1", seed=self.lc_seed1
            )
            self.l2 = self._build_l(
                self.S, self.a_input, name="lyapunov_critic_2", seed=self.lc_seed2
            )
            self.log_pis = log_pis = self.a_dist.log_prob(
                self.a
            )  # Gaussian actor action log_probability
            self.q1 = self._build_c(
                self.S, self.a_input, "q_critic_1", seed=self.qc_seed1
            )
            self.q2 = self._build_c(
                self.S, self.a_input, "q_critic_2", seed=self.qc_seed2
            )

            # Retrieve GA and LC network parameters
            a_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/gaussian_actor"
            )
            l1_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope="Actor/lyapunov_critic_1",
            )
            l2_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope="Actor/lyapunov_critic_2",
            )
            c1_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/q_critic_1"
            )
            c2_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/q_critic_2"
            )

            # Create EMA target network update policy (Soft replacement)
            ema = tf.train.ExponentialMovingAverage(decay=self.polyak)

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [
                ema.apply(a_params),
                ema.apply(l1_params),
                ema.apply(l2_params),
                ema.apply(c1_params),
                ema.apply(c2_params),
            ]

            # Create GA and LC target networks
            # Don't get optimized but get updated according to the EMA of the main
            # networks
            a_, _, a_dist_ = self._build_a(
                self.S_,
                reuse=True,
                custom_getter=ema_getter,
                seeds=self.ga_target_seeds,
            )
            l1_ = self._build_l(
                self.S_,
                a_,
                name="lyapunov_critic_1",
                reuse=True,
                custom_getter=ema_getter,
                seed=self.lc_target_seed,
            )
            l2_ = self._build_l(
                self.S_,
                a_,
                reuse=True,
                name="lyapunov_critic_2",
                custom_getter=ema_getter,
                seed=self.lc_target_seed,
            )
            q1_ = self._build_c(
                self.S_,
                a_,
                "q_critic_1",
                reuse=True,
                custom_getter=ema_getter,
                seed=self.qc_target_seed1,
            )
            q2_ = self._build_c(
                self.S_,
                a_,
                "q_critic_2",
                reuse=True,
                custom_getter=ema_getter,
                seed=self.qc_target_seed2,
            )

            # Graph for value of current best action out of the Actor
            lya_a_, _, _ = self._build_a(self.S_, reuse=True, seeds=self.ga_seeds)
            self.l1_ = self._build_l(
                self.S_,
                lya_a_,
                name="lyapunov_critic_1",
                reuse=True,
                seed=self.lc_seed1,
            )
            self.l2_ = self._build_l(
                self.S_,
                lya_a_,
                name="lyapunov_critic_2",
                reuse=True,
                seed=self.lc_seed2,
            )
            q1_a = self._build_c(
                self.S, self.a, "q_critic_1", reuse=True, seed=self.qc_seed1
            )
            q2_a = self._build_c(
                self.S, self.a, "q_critic_2", reuse=True, seed=self.qc_seed2
            )

            ###########################################
            # Create Loss functions and optimizers ####
            ###########################################

            # Lyapunov constraint function
            # Question: How do we want to use the double Q trick here?
            # TEST: Test if this goes how it supposed to!
            self.l1_delta = tf.reduce_mean(
                input_tensor=(self.l1_ - self.l1 + (ALG_PARAMS["alpha3"]) * self.R)
            )
            self.l2_delta = tf.reduce_mean(
                input_tensor=(self.l2_ - self.l2 + (ALG_PARAMS["alpha3"]) * self.R)
            )
            self.l_delta_max = tf.reduce_min([self.l1_delta, self.l2_delta], axis=0)

            # Lagrance multiplier loss functions and optimizers graphs
            labda_loss = -tf.reduce_mean(
                input_tensor=(self.log_labda * self.l_delta_max)
            )
            alpha_loss = -tf.reduce_mean(
                input_tensor=(
                    self.log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
                )
            )
            self.alpha_train = tf.compat.v1.train.AdamOptimizer(self.LR_A).minimize(
                alpha_loss, var_list=self.log_alpha
            )
            self.lambda_train = tf.compat.v1.train.AdamOptimizer(self.LR_lag).minimize(
                labda_loss, var_list=self.log_labda
            )

            # Retrieve min Q target (Current state but now actor input)
            min_Q_target = tf.reduce_max((q1_a, q2_a), axis=0)

            # Actor loss and optimizer graph
            if self.use_lyapunov:
                # Test input tensor
                a_loss = self.labda * self.l_delta_max + tf.reduce_mean(
                    self.alpha * log_pis + min_Q_target
                )
            else:
                a_loss = tf.reduce_mean(self.alpha * log_pis + min_Q_target)
            self.a_loss = a_loss  # FIXME: IS this needed?
            self.a_train = tf.compat.v1.train.AdamOptimizer(self.LR_A).minimize(
                a_loss, var_list=a_params
            )

            # Create Lyapunov Critic loss function and optimizer
            # NOTE (rickstaa): We use a control dependency to make sure the target
            # networks are updated first.
            next_log_pis = a_dist_.log_prob(a_)
            with tf.control_dependencies(target_update):

                # Create Lyapunov critic graph
                max_next_l = tf.reduce_max([l1_, l2_], axis=0)
                l1_target = self.R + ALG_PARAMS["gamma"] * (
                    1 - self.terminal
                ) * tf.stop_gradient(max_next_l)
                l2_target = self.R + ALG_PARAMS["gamma"] * (
                    1 - self.terminal
                ) * tf.stop_gradient(max_next_l)

                self.l1_error = tf.compat.v1.losses.mean_squared_error(
                    labels=l1_target, predictions=self.l1
                )
                self.l2_error = tf.compat.v1.losses.mean_squared_error(
                    labels=l2_target, predictions=self.l2
                )
                self.l_train_1 = tf.compat.v1.train.AdamOptimizer(self.LR_L).minimize(
                    self.l1_error, var_list=l1_params
                )
                self.l_train_2 = tf.compat.v1.train.AdamOptimizer(self.LR_L).minimize(
                    self.l2_error, var_list=l2_params
                )

                # Calculate target q values
                max_next_q = tf.reduce_max([q1_, q2_], axis=0)
                q1_target = self.R + ALG_PARAMS["gamma"] * (
                    1 - self.terminal
                ) * tf.compat.v1.stop_gradient(
                    max_next_q - self.alpha * next_log_pis
                )  # ddpg
                q2_target = self.R + ALG_PARAMS["gamma"] * (
                    1 - self.terminal
                ) * tf.compat.v1.stop_gradient(
                    max_next_q - self.alpha * next_log_pis
                )  # ddpg

                # Create Q critic graph
                self.td_error_1 = tf.compat.v1.losses.mean_squared_error(
                    labels=q1_target, predictions=self.q1
                )
                self.td_error_2 = tf.compat.v1.losses.mean_squared_error(
                    labels=q2_target, predictions=self.q2
                )
                self.qc_train_1 = tf.compat.v1.train.AdamOptimizer(self.LR_C).minimize(
                    self.td_error_1, var_list=c1_params
                )
                self.qc_train_2 = tf.compat.v1.train.AdamOptimizer(self.LR_C).minimize(
                    self.td_error_2, var_list=c2_params
                )

            # Initialize variables, create saver and diagnostics graph
            self.entropy = tf.reduce_mean(input_tensor=-self.log_pis)
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()
            # TODO: Add both critics
            if self.use_lyapunov:
                self.diagnostics = [
                    self.entropy,
                    self.labda,
                    self.alpha,
                    self.a_loss,
                    alpha_loss,
                    labda_loss,
                    self.l1_error,
                    self.l2_error,
                    self.td_error_1,
                    self.td_error_2,
                ]
            else:
                self.diagnostics = [
                    self.entropy,
                    self.alpha,
                    self.a_loss,
                    alpha_loss,
                    self.td_error_1,
                    self.td_error_2,
                ]

            # Create optimizer array
            if self.use_lyapunov:
                self.opt = [
                    self.l_train_1,
                    self.l_train_2,
                    self.lambda_train,
                    self.qc_train_1,
                    self.qc_train_2,
                ]
            else:
                self.opt = [self.qc_train_1, self.qc_train_2]
            self.opt.append(self.a_train)
            if ALG_PARAMS["adaptive_alpha"]:
                self.opt.append(self.alpha_train)

    def choose_action(self, s, evaluation=False):
        """Returns the current action of the policy.

        Args:
            s (np.numpy): The current state.
            evaluation (bool, optional): Whether to return a deterministic action.
            Defaults to False.

        Returns:
            np.numpy: The current action.
        """
        if evaluation is True:
            try:
                return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[
                    0
                ]
            except ValueError:
                return
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, LR_A, LR_L, LR_lag, LR_C, batch):
        """Runs the SGD to update all the optimizable parameters.

        Args:
            LR_A (float): Current actor learning rate.
            LR_L (float): Lyapunov critic learning rate.
            LR_lag (float): Lyapunov constraint langrance multiplier learning rate.
            LR_C (float): Q critic learning rate.
            batch (numpy.ndarray): The batch of experiences.

        Returns:
            [type]: [description]
        """

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Fill optimizer variable feed dict
        if self.use_lyapunov:
            feed_dict = {
                self.a_input: ba,
                self.S: bs,
                self.S_: bs_,
                self.R: br,
                self.terminal: bterminal,
                self.LR_A: LR_A,
                self.LR_L: LR_L,
                self.LR_lag: LR_lag,
                self.LR_C: LR_C,
            }
        else:
            feed_dict = {
                self.a_input: ba,
                self.S: bs,
                self.S_: bs_,
                self.R: br,
                self.terminal: bterminal,
                self.LR_A: LR_A,
                self.LR_C: LR_C,
            }

        # Run optimization
        self.sess.run(self.opt, feed_dict)

        # Retrieve diagnostic variables from the optimization
        if self.use_lyapunov:
            (
                entropy,
                labda,
                alpha,
                a_loss,
                alpha_loss,
                labda_loss,
                l1_error,
                l2_error,
                q1_error,
                q2_error,
            ) = self.sess.run(self.diagnostics, feed_dict)

            # Return optimization results
            return labda, alpha, q1_error, q2_error, l1_error, l2_error, entropy, a_loss
        else:
            entropy, alpha, a_loss, alpha_loss, q1_error, q2_error = self.sess.run(
                self.diagnostics, feed_dict
            )

            # Return optimization results
            return alpha, q1_error, q2_error, entropy, a_loss

    def _build_a(
        self,
        s,
        name="gaussian_actor",
        reuse=None,
        custom_getter=None,
        seeds=[None, None],
    ):
        """Setup SquashedGaussianActor Graph.

        Args:
            s (tf.Tensor): [description]

            name (str, optional): Network name. Defaults to "actor".

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

            seeds (list, optional): The random seeds used for the weight initialization
                and the sampling ([weights_seed, sampling_seed]). Defaults to
                [None, None]
        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Check if random seed is set
        if RANDOM_SEED is not None:
            # Create weight initializer
            initializer = GlorotUniform(seed=seeds[0])

        # Create graph
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer sizes
            n1 = self.network_structure["actor"][0]
            n2 = self.network_structure["actor"][1]

            # Create actor hidden/ output layers
            if RANDOM_SEED is not None:
                net_0 = tf.compat.v1.layers.dense(
                    s,
                    n1,
                    activation=tf.nn.relu,
                    name="l1",
                    trainable=trainable,
                    kernel_initializer=initializer,
                )  # 原始是30
                net_1 = tf.compat.v1.layers.dense(
                    net_0,
                    n2,
                    activation=tf.nn.relu,
                    name="l2",
                    trainable=trainable,
                    kernel_initializer=initializer,
                )  # 原始是30
                mu = tf.compat.v1.layers.dense(
                    net_1,
                    self.a_dim,
                    activation=None,
                    name="mu",
                    trainable=trainable,
                    kernel_initializer=initializer,
                )
                log_sigma = tf.compat.v1.layers.dense(
                    net_1,
                    self.a_dim,
                    activation=None,
                    name="log_sigma",
                    trainable=trainable,
                    kernel_initializer=initializer,
                )
            else:
                net_0 = tf.compat.v1.layers.dense(
                    s, n1, activation=tf.nn.relu, name="l1", trainable=trainable,
                )  # 原始是30
                net_1 = tf.compat.v1.layers.dense(
                    net_0, n2, activation=tf.nn.relu, name="l2", trainable=trainable,
                )  # 原始是30
                mu = tf.compat.v1.layers.dense(
                    net_1, self.a_dim, activation=None, name="mu", trainable=trainable,
                )
                log_sigma = tf.compat.v1.layers.dense(
                    net_1,
                    self.a_dim,
                    activation=None,
                    name="log_sigma",
                    trainable=trainable,
                )
            log_sigma = tf.clip_by_value(log_sigma, *LOG_SIGMA_MIN_MAX)

            # Calculate log probability standard deviation
            sigma = tf.exp(log_sigma)

            # Create bijectors (Used in the parameterization trick)
            # NOTE (rickstaa): Done to support both tf1.x and tf2.x.
            squash_bijector = SquashBijector()
            if TF_115:
                affine_bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
            else:
                affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(sigma))

            # Sample from the normal distribution and calculate the action
            batch_size = tf.shape(input=s)[0]
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
            )
            if RANDOM_SEED is not None:
                epsilon = base_distribution.sample(batch_size, seed=seeds[1])
            else:
                epsilon = base_distribution.sample(batch_size)
            raw_action = affine_bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            # Transform distribution back to the original policy distribution
            reparm_trick_bijector = tfp.bijectors.Chain(
                (squash_bijector, affine_bijector)
            )
            distribution = tfp.distributions.TransformedDistribution(
                distribution=base_distribution, bijector=reparm_trick_bijector
            )

            clipped_mu = squash_bijector.forward(mu)

        return clipped_a, clipped_mu, distribution

    def _build_c(
        self, s, a, name="q_critic", reuse=None, custom_getter=None, seed=None
    ):
        """Setup q critic graph.

        Args:
            s (tf.Tensor): Tensor of observations.

            a (tf.Tensor): Tensor with actions.

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

            seed (int, optional): The seed used for the weight initialization. Defaults
                to None.
        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Create weight initializer
        if RANDOM_SEED is not None:
            initializer = GlorotUniform(seed=seed)

        # Create graph
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer size
            n1 = self.network_structure["q_critic"][0]

            # Create actor hidden/ output layers
            if RANDOM_SEED is not None:
                layers = []
                w1_s = tf.compat.v1.get_variable(
                    "w1_s",
                    [self.s_dim, n1],
                    trainable=trainable,
                    initializer=initializer,
                )
                w1_a = tf.compat.v1.get_variable(
                    "w1_a",
                    [self.a_dim, n1],
                    trainable=trainable,
                    initializer=initializer,
                )
                b1 = tf.compat.v1.get_variable(
                    "b1", [1, n1], trainable=trainable, initializer=tf.zeros_initializer
                )
                net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                layers.append(net_0)
                for i in range(1, len(self.network_structure["q_critic"])):
                    n = self.network_structure["q_critic"][i]
                    layers.append(
                        tf.compat.v1.layers.dense(
                            layers[i - 1],
                            n,
                            activation=tf.nn.relu,
                            name="l" + str(i + 1),
                            trainable=trainable,
                            kernel_initializer=initializer,
                        )
                    )
            else:
                layers = []
                # FIXME: UPDATE TO NEW VERSION
                w1_s = tf.compat.v1.get_variable(
                    "w1_s", [self.s_dim, n1], trainable=trainable,
                )
                w1_a = tf.compat.v1.get_variable(
                    "w1_a", [self.a_dim, n1], trainable=trainable,
                )
                b1 = tf.compat.v1.get_variable("b1", [1, n1], trainable=trainable,)
                net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                layers.append(net_0)
                for i in range(1, len(self.network_structure["q_critic"])):
                    n = self.network_structure["q_critic"][i]
                    layers.append(
                        tf.compat.v1.layers.dense(
                            layers[i - 1],
                            n,
                            activation=tf.nn.relu,
                            name="l" + str(i + 1),
                            trainable=trainable,
                        )
                    )

            # Return l-critic graph
            if RANDOM_SEED is not None:
                return tf.compat.v1.layers.dense(
                    layers[-1], 1, trainable=trainable, kernel_initializer=initializer
                )  # Q(s,a)
            else:
                return tf.compat.v1.layers.dense(
                    layers[-1], 1, trainable=trainable
                )  # Q(s,a)

    def _build_l(
        self, s, a, name="lyapunov_critic", reuse=None, custom_getter=None, seed=None
    ):
        """Setup lyapunov critic graph.

        Args:
            s (tf.Tensor): Tensor of observations.

            a (tf.Tensor): Tensor with actions.

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

            seed (int, optional): The seed used for the weight initialization. Defaults
                to None.
        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Create graph
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer size
            n1 = self.network_structure["critic"][0]

            # Create weight initializer
            initializer = GlorotUniform(seed=seed)

            # Create actor hidden/ output layers
            layers = []
            layers.append(
                tf.compat.v1.layers.dense(
                    tf.concat([s, a], axis=1),
                    n1,
                    activation=tf.nn.relu,
                    name="l1",
                    trainable=trainable,
                    kernel_initializer=initializer,
                )
            )
            for i in range(1, len(self.network_structure["critic"])):
                n = self.network_structure["critic"][i]
                layers.append(
                    tf.compat.v1.layers.dense(
                        layers[i - 1],
                        n,
                        activation=tf.nn.relu,
                        name="l" + str(i + 1),
                        trainable=trainable,
                        kernel_initializer=initializer,
                    )
                )

            # Return lyapunov critic object
            return tf.expand_dims(
                tf.reduce_sum(input_tensor=tf.square(layers[-1]), axis=1), axis=1
            )  # L(s,a)

    def save_result(self, path):
        """Save current policy.

        Args:
            path (str): The path where you want to save the policy.
        """

        # Retrieve relative path
        # NOTE (rickstaa): We nee to do this since tf saves the used path in the
        # checkpoint file.
        save_path_rel = os.path.relpath(path + "/policy/model.ckpt", start=os.curdir)

        # Save model
        save_path = self.saver.save(self.sess, save_path_rel)
        print("Save to path: ", save_path)

    def restore(self, path):
        """Restore policy.

        Args:
            path (str): The path where you want to save the policy.

        Returns:
            bool: Boolean specifying whether the policy was loaded successfully.
        """
        model_file = tf.train.latest_checkpoint(path + "/")
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True
        return success_load


def train(log_dir):
    """Performs the agent traning.

    Args:
        log_dir (str): The directory in which the final model (policy) and the
        log data is saved.
    """

    # Create environment
    print(f"Your training in the {ENV_NAME} environment.\n")
    env = get_env_from_name(ENV_NAME, ENV_SEED)
    test_env = get_env_from_name(ENV_NAME, ENV_SEED)

    # Set initial learning rates
    lr_a, lr_l, lr_c = (
        ALG_PARAMS["lr_a"],
        ALG_PARAMS["lr_l"],
        ALG_PARAMS["lr_c"],
    )
    lr_a_now = ALG_PARAMS["lr_a"]  # learning rate for actor, lambda and alpha
    lr_l_now = ALG_PARAMS["lr_l"]  # learning rate for lyapunov critic
    lr_c_now = ALG_PARAMS["lr_c"]  # learning rate for q critic

    # Get observation and action space dimension and limits from the environment
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    # Create the Lyapunov Actor Critic agent
    policy = LAC(a_dim, s_dim)

    # Load model if retraining is selected
    if TRAIN_PARAMS["continue_training"]:

        # Create retrain path
        retrain_model_folder = TRAIN_PARAMS["continue_model_folder"]
        retrain_model_path = os.path.abspath(
            os.path.join(log_dir, "../../" + TRAIN_PARAMS["continue_model_folder"])
        )

        # Check if retrain model exists if not throw error
        if not os.path.exists(retrain_model_path):
            print(
                "Shutting down training since the model you specified in the "
                f"`continue_model_folder` `{retrain_model_folder}` "
                f"argument was not found for the `{ENV_NAME}` environment."
            )
            sys.exit(0)

        # Load retrain model
        print(f"Restoring model `{retrain_model_path}`")
        result = policy.restore(os.path.abspath(retrain_model_path + "/policy"))
        if not result:
            print(
                "Shuting down training as something went wrong while loading "
                f"model `{retrain_model_folder}`."
            )
            sys.exit(0)

        # Create new storage folder
        log_dir_split = log_dir.split("/")
        log_dir_split[-2] = (
            "_".join(TRAIN_PARAMS["continue_model_folder"].split("/"))
            + "_finetune"
            # + "_retrained_"
            # + log_dir_split[-2]
        )
        log_dir = "/".join(log_dir_split)

        # Reset lagrance multipliers if requested
        if ALG_PARAMS["reset_lagrance_multipliers"]:
            policy.sess.run(policy.log_alpha.assign(tf.math.log(ALG_PARAMS["alpha"])))
            policy.sess.run(policy.log_labda.assign(tf.math.log(ALG_PARAMS["labda"])))
    else:
        print(f"Train new model `{log_dir}`")

    # Print logging folder
    print(f"Logging results to `{log_dir}`.")

    # Create replay memory buffer
    pool = Pool(
        s_dim=s_dim,
        a_dim=a_dim,
        store_last_n_paths=TRAIN_PARAMS["num_of_training_paths"],
        memory_capacity=ALG_PARAMS["memory_capacity"],
        min_memory_size=ALG_PARAMS["min_memory_size"],
    )

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=TRAIN_PARAMS["num_of_training_paths"])
    training_started = False

    # Setup logger and log hyperparameters
    logger.configure(dir=log_dir, format_strs=["csv"])
    logger.logkv("tau", ALG_PARAMS["tau"])
    logger.logkv("alpha3", ALG_PARAMS["alpha3"])
    logger.logkv("batch_size", ALG_PARAMS["batch_size"])
    logger.logkv("target_entropy", policy.target_entropy)

    # Training loop
    for i in range(ENV_PARAMS["max_episodes"]):

        # Create variable to store information about the current path
        if policy.use_lyapunov:
            current_path = {
                "rewards": [],
                "lyapunov_error": [],
                "critic_error": [],
                "alpha": [],
                "lambda": [],
                "entropy": [],
                "a_loss": [],
            }
        else:
            current_path = {
                "rewards": [],
                "critic_error": [],
                "alpha": [],
                "entropy": [],
                "a_loss": [],
            }

        # # Stop training if max number of steps has been reached
        # # FIXME: OLD_VERSION This makes no sense since the global steps will never be
        # # the set global steps in this case.
        # if global_step > ENV_PARAMS["max_global_steps"]:
        #     print(f"Training stopped after {global_step} steps.")
        #     break

        # Reset environment
        s = env.reset()

        # Training Episode loop
        for j in range(ENV_PARAMS["max_ep_steps"]):

            # Save intermediate checkpoints if requested
            if TRAIN_PARAMS["save_checkpoints"]:
                if (
                    global_step % TRAIN_PARAMS["checkpoint_save_freq"] == 0
                    and global_step != 0
                ):

                    # Create intermediate result checkpoint folder
                    checkpoint_save_path = os.path.abspath(
                        os.path.join(log_dir, "checkpoints", "step_" + str(j))
                    )
                    os.makedirs(checkpoint_save_path, exist_ok=True)

                    # Save intermediate checkpoint
                    policy.save_result(checkpoint_save_path)

            # Break out of loop if global steps have been reached
            # FIXME: NEW Here makes sense
            if global_step > ENV_PARAMS["max_global_steps"]:

                # Print step count, save model and stop the program
                print(f"Training stopped after {global_step} steps.")
                print("Running time: ", time.time() - t1)
                print("Saving Model")
                policy.save_result(log_dir)
                print("Running time: ", time.time() - t1)
                return

            # Render environment if requested
            if ENV_PARAMS["eval_render"]:
                env.render()

            # Retrieve (scaled) action based on the current policy
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2

            # Perform action in env
            s_, r, done, _ = env.step(action)

            # Increment global step count
            if training_started:
                global_step += 1

            # Stop episode if max_steps has been reached
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            terminal = 1.0 if done else 0.0

            # Store experience in replay buffer
            pool.store(s, a, r, terminal, s_)

            # Optimize weights and parameters using STG
            if (
                pool.memory_pointer > ALG_PARAMS["min_memory_size"]
                and global_step % ALG_PARAMS["steps_per_cycle"] == 0
            ):
                training_started = True

                # Perform STG a set number of times (train per cycle)
                for _ in range(ALG_PARAMS["train_per_cycle"]):
                    batch = pool.sample(ALG_PARAMS["batch_size"])
                    if policy.use_lyapunov:
                        (
                            labda,
                            alpha,
                            q1_error,
                            q2_error,
                            l1_loss,
                            l2_loss,
                            entropy,
                            a_loss,
                        ) = policy.learn(lr_a_now, lr_l_now, lr_a, lr_c_now, batch)
                    else:
                        (alpha, q1_error, q2_error, entropy, a_loss,) = policy.learn(
                            lr_a_now, lr_l_now, lr_a, lr_c_now, batch
                        )

            # Save path results
            if training_started:
                if policy.use_lyapunov:
                    current_path["rewards"].append(r)
                    current_path["lyapunov_error"].append(max(l1_loss, l2_loss))
                    current_path["critic_error"].append(max(q1_error, q2_error))
                    current_path["alpha"].append(alpha)
                    current_path["lambda"].append(labda)
                    current_path["entropy"].append(entropy)
                    current_path["a_loss"].append(a_loss)
                else:
                    current_path["rewards"].append(r)
                    current_path["critic_error"].append(max(q1_error, q2_error))
                    current_path["alpha"].append(alpha)
                    current_path["entropy"].append(entropy)
                    current_path["a_loss"].append(a_loss)

            # Evalute the current performance and log results
            if (
                training_started
                and global_step % TRAIN_PARAMS["evaluation_frequency"] == 0
                and global_step > 0
            ):
                logger.logkv("total_timesteps", global_step)
                training_diagnostics = evaluate_training_rollouts(last_training_paths)
                if training_diagnostics is not None:
                    if TRAIN_PARAMS["num_of_evaluation_paths"] > 0:
                        eval_diagnostics = training_evaluation(test_env, policy)
                        [
                            logger.logkv(key, eval_diagnostics[key])
                            for key in eval_diagnostics.keys()
                        ]
                        training_diagnostics.pop("return")
                    [
                        logger.logkv(key, training_diagnostics[key])
                        for key in training_diagnostics.keys()
                    ]
                    logger.logkv("lr_a", lr_a_now)
                    if policy.use_lyapunov:
                        logger.logkv("lr_l", lr_l_now)
                    else:
                        logger.logkv("lr_c", lr_c_now)
                    string_to_print = ["time_step:", str(global_step), "|"]
                    if TRAIN_PARAMS["num_of_evaluation_paths"] > 0:
                        [
                            string_to_print.extend(
                                [key, ":", str(eval_diagnostics[key]), "|"]
                            )
                            for key in eval_diagnostics.keys()
                        ]
                    [
                        string_to_print.extend(
                            [key, ":", str(round(training_diagnostics[key], 2)), "|"]
                        )
                        for key in training_diagnostics.keys()
                    ]
                    print("".join(string_to_print))
                logger.dumpkvs()

            # Update state
            s = s_

            # Decay learning rate
            if done:

                # Store paths
                if training_started:
                    last_training_paths.appendleft(current_path)

                # Decay learning rates
                frac = 1.0 - (global_step - 1.0) / ENV_PARAMS["max_global_steps"]
                lr_a_now = lr_a * frac  # learning rate for actor, lambda, alpha
                lr_l_now = lr_l * frac  # learning rate for lyapunov critic
                lr_c_now = lr_c * frac  # learning rate for q critic
                break  # FIXME: Redundant
