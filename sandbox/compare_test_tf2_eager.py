"""Small example script used to investigate the difference in performance when enabling
eager execution. This is the eager enabled script."""

import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
# Used network functions ###########################
####################################################
class SquashedGaussianActor(tf.keras.Model):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        name,
        log_std_min=-20,
        log_std_max=2.0,
        seed=None,
        **kwargs
    ):
        """Squashed Gaussian actor network.

        Args:
            obs_dim (int): The dimension of the observation space.
            act_dim (int): The dimension of the action space.
            hidden_sizes (list): Array containing the sizes of the hidden layers.
            name (str): The keras module name.
            log_std_min (int, optional): The min log_std. Defaults to -20.
            log_std_max (float, optional): The max log_std. Defaults to 2.0.
            seed (int, optional): The random seed. Defaults to None.
        """
        super().__init__(name=name, **kwargs)

        # Get class parameters
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self.s_dim = obs_dim
        self.a_dim = act_dim
        self.tfp_seed = tfp.util.SeedStream(seed, salt="random_beta")

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
        epsilon = base_distribution.sample(batch_size, seed=self.tfp_seed())
        raw_action = affine_bijector.forward(epsilon)
        clipped_a = squash_bijector.forward(raw_action)

        # Transform distribution back to the original policy distribution
        reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
        distribution = tfp.distributions.TransformedDistribution(
            distribution=base_distribution, bijector=reparm_trick_bijector
        )
        clipped_mu = squash_bijector.forward(mu)
        return clipped_a, clipped_mu, distribution.log_prob(clipped_a)


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
        """Lyapunov Critic network.

        Args:
            obs_dim (int): The dimension of the observation space.
            act_dim (int): The dimension of the action space.
            hidden_sizes (list): Array containing the sizes of the hidden layers.
            name (str): The keras module name.
            log_std_min (int, optional): The min log_std. Defaults to -20.
            log_std_max (float, optional): The max log_std. Defaults to 2.0.
        """
        super().__init__(name=name, **kwargs)

        # Get class parameters
        self.s_dim = obs_dim
        self.a_dim = act_dim

        # Create input layer weights and biases
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
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=(hidden_sizes[0]), name=name + "/input",
                )
            ]
        )
        # FIXME: This layer is a different object that the one in guassian actor it has
        # no BIAS attribute
        for i in range(1, len(hidden_sizes)):
            n = hidden_sizes[i]
            self.net.add(
                tf.keras.layers.Dense(n, activation="relu", name="l" + str(i + 1))
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
        return tf.expand_dims(tf.reduce_sum(tf.math.square(net_out), axis=1), axis=1)


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

        # Create Learning rate placeholders
        self.LR_A = tf.Variable(LR_A, name="LR_A")
        self.LR_lag = tf.Variable(LR_LAG, name="LR_lag")
        self.LR_L = tf.Variable(LR_L, name="LR_L")

        # Create lagrance multiplier placeholders
        self.log_labda = tf.Variable(tf.math.log(LAMBDA), name="lambda")
        self.log_alpha = tf.Variable(tf.math.log(ALPHA), name="alpha")

        ###########################################
        # Create Networks #########################
        ###########################################

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
        self.ga = self._build_a()
        self.lc = self._build_l()

        # Create GA and LC target networks
        # Don't get optimized but get updated according to the EMA of the main
        # networks
        self.ga_ = self._build_a()
        self.lc_ = self._build_l()
        self.target_init()

        ###########################################
        # Create optimizers #######################
        ###########################################

        self.alpha_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.lambda_train = tf.keras.optimizers.Adam(learning_rate=self.LR_lag)
        self.a_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.l_train = tf.keras.optimizers.Adam(learning_rate=self.LR_L)

    def _build_a(self, name="gaussian_actor"):
        """Setup SquashedGaussianActor Graph.

        Args:
            name (str, optional): Network name. Defaults to "gaussian_actor".

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Return GA
        return SquashedGaussianActor(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["actor"],
            name=name,
            log_std_min=LOG_SIGMA_MIN_MAX[0],
            log_std_max=LOG_SIGMA_MIN_MAX[1],
            seed=RANDOM_SEED,
        )

    def _build_l(self, name="lyapunov_critic"):
        """Setup lyapunov critic graph.

        Args:
            name (str, optional): Network name. Defaults to "lyapunov_critic".

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Return LC
        return LyapunovCritic(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["critic"],
            name=name,
        )

    @tf.function
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

        # Calculate current value and target lyapunov multiplier value
        lya_a_, _, _ = self.ga(bs_)
        l_ = self.lc([bs_, lya_a_])

        # Calculate current lyapunov value
        l = self.lc([bs, ba])

        # # Calculate Lyapunov constraint function
        self.l_delta = tf.reduce_mean(l_ - l + ALPHA_3 * br)

        # Lagrance multiplier loss functions and optimizers graphs
        with tf.GradientTape() as tape:
            labda_loss = -tf.reduce_mean(self.log_labda * self.l_delta)

        # Apply gradients
        lambda_grads = tape.gradient(labda_loss, [self.log_labda])
        self.lambda_train.apply_gradients(zip(lambda_grads, [self.log_labda]))

        # Calculate log probability of a_input based on current policy
        _, _, log_pis = self.ga(bs)

        # Calculate alpha loss
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
            )

        # Apply gradients
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_train.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        # Calculate Lyapunov constraint function
        self.l_delta = tf.reduce_mean(l_ - l + ALPHA_3 * br)

        # Actor loss and optimizer graph
        with tf.GradientTape() as tape:

            # Calculate log probability of a_input based on current policy
            _, _, log_pis = self.ga(bs)

            # Calculate actor loss
            a_loss = tf.stop_gradient(self.labda) * self.l_delta + tf.stop_gradient(
                self.alpha
            ) * tf.reduce_mean(log_pis)

        # Apply gradients
        a_grads = tape.gradient(a_loss, self.ga.trainable_variables)
        self.a_train.apply_gradients(zip(a_grads, self.ga.trainable_variables))

        # Update target networks
        self.update_target()

        # Get Lypaunov target
        a_, _, _ = self.ga_(bs_)
        l_ = self.lc_([bs_, a_])
        l_target = br + GAMMA * (1 - bterminal) * tf.stop_gradient(l_)

        # Lyapunov candidate constraint function graph
        with tf.GradientTape() as tape:

            # Calculate current lyapunov value
            l = self.lc([bs, ba])

            # Calculate L_backup
            l_error = tf.compat.v1.losses.mean_squared_error(
                labels=l_target, predictions=l
            )

        # Apply gradients
        l_grads = tape.gradient(l_error, self.lc.trainable_variables)
        self.l_train.apply_gradients(zip(l_grads, self.lc.trainable_variables))

        # Return results
        return (
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
            tf.reduce_mean(tf.stop_gradient(-log_pis)),
            l_,
            l,
        )

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @property
    def labda(self):
        return tf.clip_by_value(tf.exp(self.log_labda), *SCALE_lambda_MIN_MAX)

    @tf.function
    def target_init(self):
        # Initializing targets to match main variables
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(pi_main)
        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(l_main)

    @tf.function
    def update_target(self):
        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(self.polyak * pi_targ + (1 - self.polyak) * pi_main)

        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(self.polyak * l_targ + (1 - self.polyak) * l_main)


####################################################
# Main function ####################################
####################################################
if __name__ == "__main__":

    # Create the Lyapunov Actor Critic agent
    policy = LAC()

    # Retrieve initial network weights
    ga_weights_biases = {
        "l1/weights": policy.ga.net_0.weights[0],
        "l1/bias": policy.ga.net_0.bias,
        "l2/weights": policy.ga.net_1.weights[0],
        "l2/bias": policy.ga.net_1.bias,
        "mu/weights": policy.ga.mu.weights[0],
        "mu/bias": policy.ga.mu.bias,
        "log_sigma/weights": policy.ga.log_sigma.weights[0],
        "log_sigma/bias": policy.ga.log_sigma.bias,
    }
    ga_target_weights_biases = {
        "l1/weights": policy.ga_.net_0.weights[0],
        "l1/bias": policy.ga_.net_0.bias,
        "l2/weights": policy.ga_.net_1.weights[0],
        "l2/bias": policy.ga_.net_1.bias,
        "mu/weights": policy.ga_.mu.weights[0],
        "mu/bias": policy.ga_.mu.bias,
        "log_sigma/weights": policy.ga_.log_sigma.weights[0],
        "log_sigma/bias": policy.ga_.log_sigma.bias,
    }
    lc_weights_biases = {
        "l1/w1_s": policy.lc.w1_s,
        "l1/w1_a": policy.lc.w1_a,
        "l1/b1": policy.lc.b1,
        "l2/weights": policy.lc.net.weights[0],
        "l2/bias": policy.lc.net.weights[1],
    }
    lc_target_weights_biases = {
        "l1/w1_s": policy.lc.w1_s,
        "l1/w1_a": policy.lc.w1_a,
        "l1/b1": policy.lc.b1,
        "l2/weights": policy.lc.net.weights[0],
        "l2/bias": policy.lc.net.weights[1],
    }

    # Create dummy input
    batch = {
        "s": tf.random.uniform((BATCH_SIZE, policy.s_dim)),
        "a": tf.random.uniform((BATCH_SIZE, policy.a_dim)),
        "r": tf.random.uniform((BATCH_SIZE, 1)),
        "terminal": tf.zeros((BATCH_SIZE, 1)),
        "s_": tf.random.uniform((BATCH_SIZE, policy.s_dim)),
    }

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
