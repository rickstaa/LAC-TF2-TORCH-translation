"""Small example script used to investigate the difference in performance when enabling
eager execution. This is the non-eager script."""

import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# Disable eager
tf.compat.v1.disable_eager_execution()

# Disable GPU if requested
tf.config.set_visible_devices([], "GPU")

####################################################
# Script parameters ################################
####################################################
S_DIM = 2  # Observation space dimension
A_DIM = 2  # Observation space dimension
BATCH_SIZE = 256  # Replay buffer batch size
LOG_SIGMA_MIN_MAX = (-20, 2)  # Range of log std coming out of the GA network
SCALE_lambda_MIN_MAX = (0, 1)  # Range of lambda lagrance multiplier
ALPHA_3 = 0.2  # The value of the stability condition multiplier
GAMMA = 0.9  # Discount factor
ALPHA = 0.99  # The initial value for the entropy lagrance multiplier
LAMBDA = 0.99  # Initial value for the lyapunov constraint lagrance multiplier
NETWORK_STRUCTURE = {
    "critic": [128, 128],
    "actor": [64, 64],
}  # The network structure of the agent.
POLYAK = 0.995  # Decay rate used in the polyak averaging
LR_A = 1e-4  # The actor learning rate
LR_L = 3e-4  # The lyapunov critic
LR_LAG = 1e-4  # The lagrance multiplier learning rate

####################################################
# Seed random number generators ####################
####################################################
RANDOM_SEED = 0  # The random seed

# Set random seed to get comparable results for each run
if RANDOM_SEED is not None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
    np.random.seed(RANDOM_SEED)
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


####################################################
# Used helper functions ############################
####################################################
class SquashBijector(tfp.bijectors.Bijector):
    """A squash bijector used to keeps track of the distribution properties when the
    distribution is transformed using the tanh squash function."""

    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name
        )

    def _forward(self, x):
        return tf.nn.tanh(x)
        # return x

    def _inverse(self, y):
        return tf.atanh(y)

    def _forward_log_det_jacobian(self, x):
        return 2.0 * (tf.math.log(2.0) - x - tf.nn.softplus(-2.0 * x))


####################################################
# Agent class ######################################
####################################################
class LAC(object):
    """The lyapunov actor critic agent.
    """

    def __init__(self):

        # Save action and observation space as members
        self.a_dim = A_DIM
        self.s_dim = S_DIM

        # Set algorithm parameters as class objects
        self.network_structure = NETWORK_STRUCTURE

        # Determine target entropy
        self.target_entropy = -A_DIM  # lower bound of the policy entropy

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

            # Create lagrance multiplier placeholders
            log_labda = tf.compat.v1.get_variable(
                "lambda", None, tf.float32, initializer=tf.math.log(LAMBDA)
            )
            log_alpha = tf.compat.v1.get_variable(
                "alpha", None, tf.float32, initializer=tf.math.log(ALPHA)
            )
            self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
            self.alpha = tf.exp(log_alpha)

            ###########################################
            # Create Networks #########################
            ###########################################

            # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
            self.a, self.deterministic_a, self.a_dist = self._build_a(self.S)
            self.l = self._build_l(self.S, self.a_input)
            self.log_pis = log_pis = self.a_dist.log_prob(
                self.a
            )  # Gaussian actor action log_probability

            # Retrieve GA and LC network parameters
            a_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/gaussian_actor"
            )
            l_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope="Actor/lyapunov_critic",
            )

            # Create EMA target network update policy (Soft replacement)
            ema = tf.train.ExponentialMovingAverage(decay=(POLYAK))

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [
                ema.apply(a_params),
                ema.apply(l_params),
            ]

            # Create GA and LC target networks
            # Don't get optimized but get updated according to the EMA of the main
            # networks
            a_, _, a_dist_ = self._build_a(
                self.S_, reuse=True, custom_getter=ema_getter
            )
            l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)

            # Create Networks for the (fixed) lyapunov temperature boundary
            # DEBUG: This graph has the same parameters as the original gaussian actor
            # but now it receives the next state. This was needed as the target network
            # uses exponential moving average.
            lya_a_, _, _ = self._build_a(self.S_, reuse=True)
            self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

            ###########################################
            # Create Loss functions and optimizers ####
            ###########################################

            # Lyapunov constraint function
            self.l_delta = tf.reduce_mean(
                input_tensor=(self.l_ - self.l + ALPHA_3 * self.R)
            )

            # Lagrance multiplier loss functions and optimizers graphs
            labda_loss = -tf.reduce_mean(input_tensor=(log_labda * self.l_delta))
            alpha_loss = -tf.reduce_mean(
                input_tensor=(
                    log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
                )
            )
            self.alpha_train = tf.compat.v1.train.AdamOptimizer(self.LR_A).minimize(
                alpha_loss, var_list=log_alpha
            )
            self.lambda_train = tf.compat.v1.train.AdamOptimizer(self.LR_lag).minimize(
                labda_loss, var_list=log_labda
            )

            # Actor loss and optimizer graph
            a_loss = self.labda * self.l_delta + self.alpha * tf.reduce_mean(
                input_tensor=log_pis
            )
            self.a_loss = a_loss
            self.a_train = tf.compat.v1.train.AdamOptimizer(self.LR_A).minimize(
                a_loss, var_list=a_params
            )

            # Create Lyapunov Critic loss function and optimizer
            # NOTE: The control dependency makes sure the target networks are updated
            # first
            with tf.control_dependencies(target_update):

                # Lyapunov candidate constraint function graph
                l_target = self.R + GAMMA * (1 - self.terminal) * tf.stop_gradient(l_)

                self.l_error = tf.compat.v1.losses.mean_squared_error(
                    labels=l_target, predictions=self.l
                )
                self.l_train = tf.compat.v1.train.AdamOptimizer(self.LR_L).minimize(
                    self.l_error, var_list=l_params
                )

            # Initialize variables, create saver and diagnostics graph
            self.entropy = tf.reduce_mean(input_tensor=-self.log_pis)
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()
            self.diagnostics = [
                self.l_delta,
                self.labda,
                self.alpha,
                log_labda,
                log_alpha,
                labda_loss,
                alpha_loss,
                l_target,
                self.l_error,
                self.a_loss,
                self.entropy,
                self.l_,
                self.l,
            ]

            # Create optimizer array
            self.opt = [self.l_train, self.lambda_train, self.a_train, self.alpha_train]

    def _build_a(self, s, name="gaussian_actor", reuse=None, custom_getter=None):
        """Setup SquashedGaussianActor Graph.

        Args:
            s (tf.Tensor): [description]

            name (str, optional): Network name. Defaults to "actor".

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Create graph
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer sizes
            n1 = self.network_structure["actor"][0]
            n2 = self.network_structure["actor"][1]

            # Create actor hidden/ output layers
            net_0 = tf.compat.v1.layers.dense(
                s, n1, activation=tf.nn.relu, name="l1", trainable=trainable
            )  # 原始是30
            net_1 = tf.compat.v1.layers.dense(
                net_0, n2, activation=tf.nn.relu, name="l2", trainable=trainable
            )  # 原始是30
            mu = tf.compat.v1.layers.dense(
                net_1, self.a_dim, activation=None, name="mu", trainable=trainable
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

            # Create bijectors (Used in the reparameterization trick)
            squash_bijector = SquashBijector()
            affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(sigma))

            # Sample from the normal distribution and calculate the action
            batch_size = tf.shape(input=s)[0]
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
            )
            tfp_seed = tfp.util.SeedStream(RANDOM_SEED, salt="random_beta")
            epsilon = base_distribution.sample(batch_size, seed=tfp_seed())
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

    def _build_l(self, s, a, name="lyapunov_critic", reuse=None, custom_getter=None):
        """Setup lyapunov critic graph.

        Args:
            s (tf.Tensor): Tensor of observations.

            a (tf.Tensor): Tensor with actions.

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

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

            # Create actor hidden/ output layers
            layers = []
            w1_s = tf.compat.v1.get_variable(
                "w1_s", [self.s_dim, n1], trainable=trainable
            )
            w1_a = tf.compat.v1.get_variable(
                "w1_a", [self.a_dim, n1], trainable=trainable
            )
            b1 = tf.compat.v1.get_variable("b1", [1, n1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            layers.append(net_0)
            for i in range(1, len(self.network_structure["critic"])):
                n = self.network_structure["critic"][i]
                layers.append(
                    tf.compat.v1.layers.dense(
                        layers[i - 1],
                        n,
                        activation=tf.nn.relu,
                        name="l" + str(i + 1),
                        trainable=trainable,
                    )
                )

            # Return lyapunov critic object
            return tf.expand_dims(
                tf.reduce_sum(input_tensor=tf.square(layers[-1]), axis=1), axis=1
            )  # L(s,a)

    def learn(self, LR_A, LR_L, LR_lag, batch):
        """Runs the SGD to update all the optimize parameters.

        Args:
            LR_A (float): Current actor learning rate.
            LR_L (float): Lyapunov critic learning rate.
            LR_lag (float): Lyapunov constraint langrance multiplier learning rate.
            batch (numpy.ndarray): The batch of experiences.

        Returns:
            Tuple: Tuple with diagnostics variables of the SGD.
        """

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Fill optimizer variable feed cictic
        feed_dict = {
            self.a_input: ba,
            self.S: bs,
            self.S_: bs_,
            self.R: br,
            self.terminal: bterminal,
            self.LR_A: LR_A,
            self.LR_L: LR_L,
            self.LR_lag: LR_lag,
        }

        # Run optimization
        self.sess.run(self.opt, feed_dict)

        # Retrieve diagnostic variables from the optimization
        return self.sess.run(self.diagnostics, feed_dict)


####################################################
# Main function ####################################
####################################################
if __name__ == "__main__":

    # Create the Lyapunov Actor Critic agent
    policy = LAC()

    # Retrieve initial network weights
    ga_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/gaussian_actor",
    )
    ga_target_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Actor/gaussian_actor",
    )
    lc_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/lyapunov_critic",
    )
    lc_target_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Actor/lyapunov_critic",
    )
    ga_weights_biases = policy.sess.run(ga_vars)
    ga_target_weights_biases = policy.sess.run(ga_target_vars)
    lc_weights_biases = policy.sess.run(lc_vars)
    lc_target_weights_biases = policy.sess.run(lc_target_vars)
    ga_weights_biases = {
        "l1/weights": ga_weights_biases[0],
        "l1/bias": ga_weights_biases[1],
        "l2/weights": ga_weights_biases[2],
        "l2/bias": ga_weights_biases[3],
        "mu/weights": ga_weights_biases[4],
        "mu/bias": ga_weights_biases[5],
        "log_sigma/weights": ga_weights_biases[6],
        "log_sigma/bias": ga_weights_biases[7],
    }
    ga_target_weights_biases = {
        "l1/weights": ga_target_weights_biases[0],
        "l1/bias": ga_target_weights_biases[1],
        "l2/weights": ga_target_weights_biases[2],
        "l2/bias": ga_target_weights_biases[3],
        "mu/weights": ga_target_weights_biases[4],
        "mu/bias": ga_target_weights_biases[5],
        "log_sigma/weights": ga_target_weights_biases[6],
        "log_sigma/bias": ga_target_weights_biases[7],
    }
    lc_weights_biases = {
        "l1/w1_s": lc_weights_biases[0],
        "l1/w1_a": lc_weights_biases[1],
        "l1/b1": lc_weights_biases[2],
        "l2/weights": lc_weights_biases[3],
        "l2/bias": lc_weights_biases[4],
    }
    lc_target_weights_biases = {
        "l1/w1_s": lc_target_weights_biases[0],
        "l1/w1_a": lc_target_weights_biases[1],
        "l1/b1": lc_target_weights_biases[2],
        "l2/weights": lc_target_weights_biases[3],
        "l2/bias": lc_target_weights_biases[4],
    }

    # Create dummy input
    batch = {
        "s": tf.random.uniform((BATCH_SIZE, policy.s_dim)),
        "a": tf.random.uniform((BATCH_SIZE, policy.a_dim)),
        "r": tf.random.uniform((BATCH_SIZE, 1)),
        "terminal": tf.zeros((BATCH_SIZE, 1)),
        "s_": tf.random.uniform((BATCH_SIZE, policy.s_dim)),
    }
    batch = policy.sess.run(batch)

    # Perform training epoch
    (
        l_delta,
        labda,
        alpha,
        log_labda,
        log_alpha,
        labda_loss,
        alpha_loss,
        l_target,
        l_error,
        a_loss,
        entropy,
        l_,
        l,
    ) = policy.learn(LR_A, LR_L, LR_LAG, batch)

    # Pause here to debug
    print("DEBUG")
