"""Minimal working version of the LAC algorithm script.
"""

import time
from collections import deque
import os
import sys
import copy

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gaussian_actor import SquashedGaussianActor
from lyapunov_critic import LyapunovCritic
from squash_bijector import SquashBijector
from utils import evaluate_training_rollouts, get_env_from_name, training_evaluation
import logger
from pool import Pool

# Check if eager mode is enabled
print("\nEager enabled: " + str(tf.executing_eagerly()) + "\n")


###############################################
# Script settings #############################
###############################################
from variant import (
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
if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)


###############################################
# LAC algorithm class #########################
###############################################
class LAC(object):
    """The lyapunov actor critic.

    """

    def __init__(
        self, a_dim, s_dim,
    ):
        """Initiate object state.

        Args:
            a_dim (int): Action space dimension.
            s_dim (int): Observation space dimension.
        """

        # Save action and observation space as members
        self.a_dim = a_dim
        self.s_dim = s_dim

        # Set algorithm parameters as class objects
        self.polyak = (1 - ALG_PARAMS["tau"])
        self.network_structure = ALG_PARAMS["network_structure"]

        # Determine target entropy
        if ALG_PARAMS["target_entropy"]:
            self.target_entropy = ALG_PARAMS["target_entropy"]
        else:
            self.target_entropy = -self.a_dim  # lower bound of the policy entropy

        # Create Learning rate placeholders
        self.LR_A = tf.Variable(ALG_PARAMS["lr_a"], name="LR_A")
        self.LR_lag = tf.Variable(ALG_PARAMS["lr_a"], name="LR_lag")
        self.LR_L = tf.Variable(ALG_PARAMS["lr_l"], name="LR_L")

        # Create lagrance multiplier placeholders
        self.log_labda = tf.Variable(ALG_PARAMS["labda"], name="lambda")
        self.log_alpha = tf.Variable(ALG_PARAMS["alpha"], name="alpha")

        ###########################################
        # Create Networks #########################
        ###########################################

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
        self.ga = self._build_a()
        self.lc = self._build_l()

        # Create GA and LC target networks
        # Don't get optimized but get updated according to the EMA of the main
        # networks
        self.ga_ = self._build_a(name="TargetActor")
        self.lc_ = self._build_l(name="TargetCritic")
        self.target_init()  # Initiate target weights

        # Create Networks for the (fixed) lyapunov temperature boundary
        # NOTE: Used as a minimum lambda constraint boundary
        self.lya_ga_ = self._build_a()
        self.lya_lc_ = self._build_l()

        ###########################################
        # Create optimizers #######################
        ###########################################
        self.alpha_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.lambda_train = tf.keras.optimizers.Adam(learning_rate=self.LR_lag)
        self.a_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.l_train = tf.keras.optimizers.Adam(learning_rate=self.LR_L)

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
                _, deterministic_a, _ = self.ga(s[np.newaxis, :], training=False)
                return deterministic_a.numpy()[0]
            except ValueError:
                return
        else:
            a, _, _ = self.ga(s[np.newaxis, :], training=True)
            return a.numpy()[0]

    def learn(self, LR_A, LR_L, LR_lag, batch):
        """Runs the SGD to update all the optimizable parameters.

        Args:
            LR_A (float): Current actor learning rate.
            LR_L (float): Lyapunov critic learning rate.
            LR_lag (float): Lyapunov constraint langrance multiplier learning rate.
            batch (numpy.ndarray): The batch of experiences.
        """

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Calculate current value and target lyapunov multiplier value
        lya_a_, _, _ = self.lya_ga_(bs_)
        l_ = self.lya_lc_(tf.concat([bs_, lya_a_], axis=1))

        # Lagrance multiplier loss functions and optimizers graphs
        with tf.GradientTape() as tape:

            # Calculate current lyapunov value
            l = self.lc(tf.concat([bs, ba], axis=1))

            # Calculate Lyapunov constraint function
            self.l_delta = tf.reduce_mean(l_ - l + (ALG_PARAMS["alpha3"]) * br)

            # Calculate lambda loss
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

        # Actor loss and optimizer graph
        with tf.GradientTape() as tape:

            # Calculate log probability of a_input based on current policy
            # NOTE: Needed 2 times because of gradient!
            # TODO: Move to one big taper!
            _, _, log_pis = self.ga(bs)

            # Calculate current lyapunov value
            l = self.lc(tf.concat([bs, ba], axis=1))

            # Calculate Lyapunov constraint function
            self.l_delta = tf.reduce_mean(l_ - l + (ALG_PARAMS["alpha3"]) * br)

            # Calculate actor loss
            a_loss = self.labda * self.l_delta + self.alpha * tf.reduce_mean(log_pis)

        # Apply gradients
        a_grads = tape.gradient(a_loss, self.ga.trainable_variables)
        self.a_train.apply_gradients(zip(a_grads, self.ga.trainable_variables))

        # Update target networks
        self.update_target()

        # Get Lypaunov target
        a_, _, _ = self.ga_(bs_)
        l_ = self.lc_(tf.concat([bs_, a_], axis=1))  # FIXME: Confusing name
        l_target = br + ALG_PARAMS["gamma"] * (1 - bterminal) * tf.stop_gradient(l_)

        # Lyapunov candidate constraint function graph
        with tf.GradientTape() as tape:

            # Calculate current lyapunov value
            l = self.lc(tf.concat([bs, ba], axis=1))

            # Calculate L_backup
            l_error = tf.compat.v1.losses.mean_squared_error(
                labels=l_target, predictions=l
            )

        # Apply gradients
        l_grads = tape.gradient(l_error, self.lc.trainable_variables)
        self.l_train.apply_gradients(zip(l_grads, self.lc.trainable_variables))

        # Update target networks
        # self.update_target()

        # Return results
        return (
            self.labda.numpy(),
            self.alpha.numpy(),
            l_error.numpy(),
            tf.reduce_mean(-1.0 * tf.stop_gradient(log_pis)).numpy(),
            a_loss.numpy(),
        )

    def _build_a(self, name="actor", reuse=None, custom_getter=None):
        """Setup SquashedGaussianActor Graph.

        Args:
            name (str, optional): Network name. Defaults to "actor".

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Return GA
        return SquashedGaussianActor(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["actor"],
            name=name,
        )

    def _build_l(self, name="Critic", reuse=None, custom_getter=None):
        """Setup lyapunov critic graph.

        Args:
            s (tf.Tensor): Tensor of observations.

            a (tf.Tensor): Tensor with actions.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Return GA
        return LyapunovCritic(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["critic"],
            name=name,
        )

    def save_result(self, path):
        """Save current policy.

        Args:
            path (str): The path where you want to save the policy.
        """

        save_path = self.saver.save(self.sess, path + "/policy/model.ckpt")
        print("Save to path: ", save_path)

    def restore(self, path):
        """Restore policy.

        Args:
            path (str): The path where you want to save the policy.

        Returns:
            bool: Boolean specifying whether the policy was loaded succesfully.
        """
        model_file = tf.train.latest_checkpoint(path + "/")
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True
        return success_load

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @property
    def labda(self):
        labda_scaled = tf.clip_by_value(tf.exp(self.log_labda), *SCALE_lambda_MIN_MAX)
        return labda_scaled

    # @tf.function
    def target_init(self):
        # Initializing targets to match main variables
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(pi_main)
        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(l_main)

    # @tf.function
    def update_target(self):
        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(self.polyak * pi_targ + (1 - self.polyak) * pi_main)

        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(self.polyak * l_targ + (1 - self.polyak) * l_main)


def train(log_dir):
    """Performs the agent traning.

    Args:
        log_dir (str): The directory in which the final model (policy) and the
        log data is saved.
    """

    # Create environment
    env = get_env_from_name(ENV_NAME, ENV_SEED)

    # Set initial learning rates
    # TODO: REdundant
    lr_a, lr_l = (
        ALG_PARAMS["lr_a"],
        ALG_PARAMS["lr_l"],
    )
    lr_a_now = ALG_PARAMS["lr_a"]  # learning rate for actor, lambda and alpha
    lr_l_now = ALG_PARAMS["lr_l"]  # learning rate for lyapunov critic

    # Get observation and action space dimension and limits from the environment
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    # Create the Lyapunov Actor Critic agent
    policy = LAC(a_dim, s_dim)

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
        current_path = {
            "rewards": [],
            "a_loss": [],
            "alpha": [],
            "lambda": [],
            "lyapunov_error": [],
            "entropy": [],
        }

        # Stop training if max number of steps has been reached
        if global_step > ENV_PARAMS["max_global_steps"]:
            break

        # Reset environment
        s = env.reset()

        # Training Episode loop
        for j in range(ENV_PARAMS["max_ep_steps"]):

            # Render environment if requested
            if ENV_PARAMS["eval_render"]:
                env.render()

            # Retrieve (scaled) action based on the current policy
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2

            # Perform action in env
            s_, r, done, info = env.step(action)

            # Increment global setp count
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
                    labda, alpha, l_loss, entropy, a_loss = policy.learn(
                        lr_a_now, lr_l_now, lr_a, batch
                    )

            # Save path results
            if training_started:
                current_path["rewards"].append(r)
                current_path["lyapunov_error"].append(l_loss)
                current_path["alpha"].append(alpha)
                current_path["lambda"].append(labda)
                current_path["entropy"].append(entropy)
                current_path["a_loss"].append(a_loss)

            # Evalute the current performance and log results
            if (
                training_started
                and global_step % TRAIN_PARAMS["evaluation_frequency"] == 0
                and global_step > 0
            ):
                logger.logkv("total_timesteps", global_step)
                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    if TRAIN_PARAMS["num_of_evaluation_paths"] > 0:
                        eval_diagnotic = training_evaluation(env, policy)
                        [
                            logger.logkv(key, eval_diagnotic[key])
                            for key in eval_diagnotic.keys()
                        ]
                        training_diagnotic.pop("return")
                    [
                        logger.logkv(key, training_diagnotic[key])
                        for key in training_diagnotic.keys()
                    ]
                    logger.logkv("lr_a", lr_a_now)
                    logger.logkv("lr_l", lr_l_now)
                    string_to_print = ["time_step:", str(global_step), "|"]
                    if TRAIN_PARAMS["num_of_evaluation_paths"] > 0:
                        [
                            string_to_print.extend(
                                [key, ":", str(eval_diagnotic[key]), "|"]
                            )
                            for key in eval_diagnotic.keys()
                        ]
                    [
                        string_to_print.extend(
                            [key, ":", str(round(training_diagnotic[key], 2)), "|"]
                        )
                        for key in training_diagnotic.keys()
                    ]
                    print("".join(string_to_print))
                logger.dumpkvs()

            # Update state
            s = s_

            # Decay learning rate
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)
                frac = 1.0 - (global_step - 1.0) / ENV_PARAMS["max_global_steps"]
                lr_a_now = lr_a * frac  # learning rate for actor, lambda, alpha
                lr_l_now = lr_l * frac  # learning rate for lyapunov critic
                break

    # Save model and print Running time
    policy.save_result(log_dir)
    print("Running time: ", time.time() - t1)
    return
