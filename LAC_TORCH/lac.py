"""LAC algorithm class

This module contains a Pytorch implementation of the Lyapunov Actor Critic (LAC)
Reinforcement learning algorithm of
[Han et al. 2019](https://arxiv.org/pdf/2004.14288.pdf).

.. note::
    Code Conventions:
        In the code we use a `_` suffix to distinguish the target network from the main
        network.
"""

import time
from collections import deque
import os
import sys
import random
from copy import deepcopy
import os.path as osp
import itertools

import torch
from torch.optim import Adam
import numpy as np

from gaussian_actor import SquashedGaussianMLPActor
from lyapunov_critic import MLPLyapunovCritic
from q_critic import QCritic
from utils import (
    evaluate_training_rollouts,
    get_env_from_name,
    training_evaluation,
    colorize,
    save_config,
)
from pool import Pool
import logger

# Script settings
from variant import (
    USE_GPU,
    ENV_NAME,
    RANDOM_SEED,
    ENV_SEED,
    TRAIN_PARAMS,
    ALG_PARAMS,
    ENV_PARAMS,
    SCALE_lambda_MIN_MAX,
)

# # Change torch backend backend (GPU: Speed improvement)
# # Note (rickstaa): To speed up training, when you are using GPU, you can uncomment
# # one of the following lines to change the torch backend (see:
# # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.fastest = True

# Set random seed to get comparable results for each run
if RANDOM_SEED is not None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# TODO: Check if adding variables as tensors speeds up computation
# TODO: Put losses in their own function (See if possible)
# TODO: Validate weight seeding.
# TODO: Change pytorch env to lower


class LAC(object):
    """The lyapunov actor critic.

    Attributes:
        ga (torch.nn.Module): The Squashed Gaussian Actor network.

        ga_ (torch.nn.Module): The Squashed Gaussian Actor target network.

        lc (torch.nn.Module): The Lyapunov Critic network.

        lc_ (torch.nn.Module): The Lyapunov Critic target network.

        q_1 (torch.nn.Module): The first Q-Critic network.

        q_2 (torch.nn.Module): The second Q-Critic network.

        q_2_ (torch.nn.Module): The first Q-Critic target network.

        q_2_ (torch.nn.Module): The second Q-Crictic target network.

        log_alpha (torch.Tensor): The temperature lagrance multiplier.

        log_labda (torch.Tensor): The lyapunov lagrance multiplier.

        target_entropy (int): The target entropy.

        device (str): The device the networks are placed on (CPU or GPU).

        use_lyapunov (bool): Whether the Lyapunov Critic is used (use_lyapunov=True) or
            the regular Q-critic (use_lyapunov=false).
    """

    def __init__(self, a_dim, s_dim, act_limits=None):
        """Initiates object state.

        Args:
            a_dim (int): Action space dimension.

            s_dim (int): Observation space dimension.

            act_limits (dict, optional): The "high" and "low" action bounds of the
                environment. Used for rescaling the actions that comes out of network
                from (-1, 1) to (low, high). Defaults to (-1, 1).
        """

        # Check if GPU is requested and available and set the device
        self.device = (
            torch.device("cuda")
            if (torch.cuda.is_available() and USE_GPU)
            else torch.device("cpu")
        )
        if not torch.cuda.is_available() and USE_GPU:
            print(
                "GPU computing was enabled but the GPU can not be reached. "
                "Reverting back to using CPU."
            )
        device_str = "GPU" if str(self.device) == "cuda" else str(self.device)
        print(colorize(f"INFO: Torch is using the {device_str}.", "cyan", bold=True))

        # Display information about the algorithm being used (LAC or SAC)
        if ALG_PARAMS["use_lyapunov"]:
            print(
                colorize(
                    ":INFO: You are training the LAC algorithm.", "cyan", bold=True
                )
            )
        else:
            print(
                colorize(
                    "WARN: You are training the SAC algorithm.", "yellow", bold=True
                )
            )

        # Save action and observation space as members
        self._a_dim = a_dim
        self._s_dim = s_dim
        self._act_limits = act_limits

        # Save algorithm parameters as class objects
        self.use_lyapunov = ALG_PARAMS["use_lyapunov"]
        self._network_structure = ALG_PARAMS["network_structure"]
        self._polyak = 1 - ALG_PARAMS["tau"]
        self._gamma = ALG_PARAMS["gamma"]
        self._alpha_3 = ALG_PARAMS["alpha3"]

        # Determine target entropy
        # NOTE (rickstaa): If not defined we use the Lower bound of the policy entropy
        if ALG_PARAMS["target_entropy"] is None:
            self.target_entropy = -self._a_dim
        else:
            self.target_entropy = ALG_PARAMS["target_entropy"]

        # Create Learning rate placeholders
        self._lr_a = ALG_PARAMS["lr_a"]
        if self.use_lyapunov:
            self._lr_lag = ALG_PARAMS["lr_a"]
            self._lr_l = ALG_PARAMS["lr_l"]
        else:
            self._lr_c = ALG_PARAMS["lr_c"]

        # Create variables for the Lagrance multipliers
        self.log_alpha = torch.tensor(ALG_PARAMS["alpha"], dtype=torch.float32).log()
        self.log_alpha.requires_grad = True
        if self.use_lyapunov:
            self.log_labda = torch.tensor(
                ALG_PARAMS["labda"], dtype=torch.float32
            ).log()
            self.log_labda.requires_grad = True

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) or Q-Critic (QC) networks
        self.ga = SquashedGaussianMLPActor(
            obs_dim=self._s_dim,
            act_dim=self._a_dim,
            hidden_sizes=self._network_structure["actor"],
            act_limits=self.act_limits,
        ).to(self.device)
        if self.use_lyapunov:
            self.lc = MLPLyapunovCritic(
                obs_dim=self._s_dim,
                act_dim=self._a_dim,
                hidden_sizes=self._network_structure["critic"],
            ).to(self.device)
        else:
            # NOTE (rickstaa): We create two Q-critics so we can use the Clipped
            # double-Q trick.
            self.q_1 = QCritic(
                obs_dim=self._s_dim,
                act_dim=self._a_dim,
                hidden_sizes=self._network_structure["q_critic"],
            ).to(self.device)
            self.q_2 = QCritic(
                obs_dim=self._s_dim,
                act_dim=self._a_dim,
                hidden_sizes=self._network_structure["q_critic"],
            ).to(self.device)

        # Create GA, LC and QC target networks
        # Don't get optimized but get updated according to the EMA of the main
        # networks
        self.ga_ = deepcopy(self.ga).to(self.device)
        if self.use_lyapunov:
            self.lc_ = deepcopy(self.lc).to(self.device)
        else:
            self.q_1_ = deepcopy(self.q_1).to(self.device)
            self.q_2_ = deepcopy(self.q_2).to(self.device)

        # Freeze target networks
        for p in self.ga_.parameters():
            p.requires_grad = False
        if self.use_lyapunov:
            for p in self.lc_.parameters():
                p.requires_grad = False
        else:
            for p in self.q_1_.parameters():
                p.requires_grad = False
            for p in self.q_2_.parameters():
                p.requires_grad = False

        # Create optimizers
        # NOTE (rickstaa): We here optimize for log_alpha and log_labda instead of
        # alpha and labda because it is more numerically stable (see:
        # https://github.com/rail-berkeley/softlearning/issues/136)
        self._alpha_train = Adam([self.log_alpha], lr=self._lr_a)
        self._a_train = Adam(self.ga.parameters(), lr=self._lr_a)
        if self.use_lyapunov:
            self._lambda_train = Adam([self.log_labda], lr=self._lr_lag)
            self._l_train = Adam(self.lc.parameters(), lr=self._lr_l)
        else:
            q_params = itertools.chain(
                self.q_1.parameters(), self.q_2.parameters()
            )  # Chain parameters of the two Q-critics
            self._q_train = Adam(q_params, lr=self._lr_c)

    def choose_action(self, s, evaluation=False):
        """Returns the current action of the policy.

        Args:
            s (np.numpy): The current state.
            evaluation (bool, optional): Whether to return a deterministic action.
            Defaults to False.

        Returns:
            np.numpy: The current action.
        """

        # Convert s to tensor if not yet the case
        s = torch.as_tensor(s, dtype=torch.float32).to(self.device)

        # Get current best action
        if evaluation is True:
            try:
                with torch.no_grad():
                    _, deterministic_a, _ = self.ga(s.unsqueeze(0))
                    return (
                        deterministic_a[0].cpu().numpy()
                    )  # IMPROVE: Check if this is the fastest method
            except ValueError:
                return
        else:
            with torch.no_grad():
                a, _, _ = self.ga(s.unsqueeze(0))
                return (
                    a[0].cpu().numpy()
                )  # IMPROVE: Check if this is the fastest method

    def learn(self, lr_a, lr_l, lr_c, lr_lag, batch):
        """Runs the SGD to update all the optimize parameters.

        Args:
            lr_a (float): Current actor learning rate.

            lr_l (float): Lyapunov critic learning rate.

            lr_c (float): Q-Critic learning rate.

            lr_lag (float): Lyapunov constraint langrance multiplier learning rate.

            batch (numpy.ndarray): The batch of experiences.

        Returns:
            tuple: Tuple with some diagnostics about the training.
        """

        # Adjust optimizer learning rates (decay)
        self._set_learning_rates(
            lr_a=lr_a, lr_alpha=lr_a, lr_l=lr_l, lr_labda=lr_lag, lr_c=lr_c
        )

        # Unpack states from the replay buffer batch
        b_s, b_a, b_r, b_terminal, b_s_ = (
            batch["s"],  # State
            batch["a"],  # Action
            batch["r"],  # Reward
            batch["terminal"],  # Done
            batch["s_"],  # Next state
        )

        # Calculate variables from which we do not require the gradients
        with torch.no_grad():
            if self.use_lyapunov:
                a_, _, _ = self.ga_(b_s_)
                l_ = self.lc_(b_s_, a_)
                l_target = b_r + self._gamma * (1 - b_terminal) * l_.detach()
            else:
                # Target actions come from *current* policy
                a2, _, logp_a2 = self.ga(b_s_)

                # Target Q-values
                q1_pi_targ = self.q_1_(b_s_, a2)
                q2_pi_targ = self.q_2_(b_s_, a2)
                q_pi_targ = torch.max(
                    q1_pi_targ,
                    q2_pi_targ,  # IMPROVE: Test if max is now workign add argument to switch?
                )  # Use min clipping to prevent overestimation bias
                backup = b_r + self._gamma * (1 - b_terminal) * (
                    q_pi_targ - self.alpha * logp_a2
                )

        # Calculate current lyapunov Q values
        if self.use_lyapunov:
            l = self.lc(b_s, b_a)

            # Calculate current value and target lyapunov multiplier value
            lya_a_, _, _ = self.ga(b_s_)
            lya_l_ = self.lc(b_s_, lya_a_)
        else:
            # Retrieve the Q values from the two networks
            q1 = self.q_1(b_s, b_a)
            q2 = self.q_2(b_s, b_a)

        # Calculate log probability of a_input based on current policy
        pi, _, log_pis = self.ga(b_s)

        # Calculate Lyapunov constraint function
        if self.use_lyapunov:
            self.l_delta = torch.mean(lya_l_ - l.detach() + self._alpha_3 * b_r)

            # Zero gradients on labda
            self._lambda_train.zero_grad()

            # Lagrance multiplier loss functions and optimizers graphs
            # FIXME: Validate if using self.labda gives the same result.
            labda_loss = -torch.mean(self.labda * self.l_delta.detach())

            # Apply gradients to log_lambda
            labda_loss.backward()
            self._lambda_train.step()
        else:

            # Retrieve the current Q values for the action given by the current policy
            q1_pi = self.q_1(b_s, pi)
            q2_pi = self.q_2(b_s, pi)
            q_pi = torch.max(q1_pi, q2_pi)  # Add change parameter

        # Zero gradients on alpha
        self._alpha_train.zero_grad()

        # Calculate alpha loss
        alpha_loss = -torch.mean(self.alpha * log_pis.detach() + self.target_entropy)

        # Apply gradients
        alpha_loss.backward()
        self._alpha_train.step()

        # Zero gradients on the actor
        self._a_train.zero_grad()

        # Calculate actor loss
        if self.use_lyapunov:
            a_loss = (self.labda.detach() * self.l_delta) + (
                self.alpha.detach() * torch.mean(log_pis)
            )
        else:
            a_loss = (self.alpha * log_pis - q_pi).mean()

        # Apply gradients
        a_loss.backward()
        self._a_train.step()

        # Optimize critic
        if self.use_lyapunov:
            # Zero gradients on the critic
            self._l_train.zero_grad()

            # Calculate L_backup
            # NOTE (rickstaa): I used manual implementation as it is 2 times than F.MSE
            # Change to F.mse_loss when TorchScript is used.
            l_error = ((l_target - l) ** 2).mean()

            # Apply gradients
            l_error.backward()
            self._l_train.step()
        else:

            # Zero gradients on q critic
            self._q_train.zero_grad()

            # MSE loss against Bellman backup
            # NOTE (rickstaa): The 0.5 multiplication factor was added to make the
            # derivation cleaner and can be safely removed without influencing the
            # minimization. We kept it here for consistency.
            # NOTE (rickstaa): I used manual implementation as it is 2 times than F.MSE
            # Change to F.mse_loss when TorchScript is used.
            loss_q1 = 0.5 * ((q1 - backup) ** 2).mean()
            loss_q2 = 0.5 * ((q2 - backup) ** 2).mean()
            loss_q = loss_q1 + loss_q2

            # Apply gradients
            loss_q.backward()
            self._q_train.step()

        # Update target networks
        self._update_targets()

        # Return results
        # IMPROVE: Check if this it the right location to do this
        # NOTE (rickstaa): Not needed when porting to machine learning control as the
        # analysis is alreadyu GPU compatible
        if self.use_lyapunov:
            return (
                # self.labda.detach(),
                # self.alpha.detach(),
                # l_error.detach(),
                # torch.mean(-log_pis.detach()),
                # a_loss.detach(),
                self.labda.cpu().detach(),
                self.alpha.cpu().detach(),
                l_error.cpu().detach(),
                torch.mean(-log_pis.cpu().detach()),
                a_loss.cpu().detach(),
            )
        else:
            return (
                # self.labda.detach(),
                # self.alpha.detach(),
                # l_error.detach(),
                # torch.mean(-log_pis.detach()),
                # a_loss.detach(),
                self.alpha.cpu().detach(),
                loss_q.cpu().detach(),
                torch.mean(-log_pis.cpu().detach()),
                a_loss.cpu().detach(),
            )

    def save_result(self, path):
        """Saves current policy.

        Args:
            path (str): The path where you want to save the policy.
        """

        # Save all models/tensors in the _save_dict
        save_path = osp.abspath(path + "/policy/model.pth")

        # Create folder if not exist
        if osp.exists(osp.dirname(save_path)):
            print(
                colorize(
                    (
                        "WARN: Log dir %s already exists! Storing info there anyway."
                        % osp.dirname(save_path)
                    ),
                    "red",
                    bold=True,
                )
            )
        else:
            os.makedirs(osp.dirname(save_path))

        # Create models state dictionary
        if self.use_lyapunov:
            models_state_save_dict = {
                "use_lyapunov": self.use_lyapunov,
                "ga_state_dict": self.ga.state_dict(),
                "lc_state_dict": self.lc.state_dict(),
                "ga_targ_state_dict": self.ga_.state_dict(),
                "lc_targ_state_dict": self.lc_.state_dict(),
                "log_alpha": self.log_alpha,
                "log_labda": self.log_labda,
            }
        else:
            models_state_save_dict = {
                "use_lyapunov": self.use_lyapunov,
                "ga_state_dict": self.ga.state_dict(),
                "ga_targ_state_dict": self.ga_.state_dict(),
                "q1_state_dict": self.q_1.state_dict(),
                "q2_state_dict": self.q_2.state_dict(),
                "q1_targ_state_dict": self.q_1_.state_dict(),
                "q2_targ_state_dict": self.q_2_.state_dict(),
                "log_alpha": self.log_alpha,
            }

        # Save model state dictionary
        torch.save(models_state_save_dict, save_path)
        print(colorize(f"INFO: Save to path: {save_path}", "cyan", bold=True))

    def restore(self, path, restore_lagrance_multipliers=True):
        """Restores policy.

        Args:
            path (str): The path where you want to save the policy.

            restore_lagrance_multipliers (bool, optional): Whether you want to restore
                the lagrance multipliers.

        Returns:
            bool: Boolean specifying whether the policy was loaded successfully.
        """

        # Create load path
        load_path = osp.abspath(path + "/model.pth")

        # Load the model state
        try:
            models_state_dict = torch.load(load_path)
        except NotADirectoryError:
            success_load = False
            return success_load

        # Restore network parameters
        if models_state_dict["use_lyapunov"]:
            self.use_lyapunov = models_state_dict["use_lyapunov"]
            self.ga.load_state_dict(models_state_dict["ga_state_dict"])
            self.lc.load_state_dict(models_state_dict["lc_state_dict"])
            self.ga_.load_state_dict(models_state_dict["ga_targ_state_dict"])
            self.lc_.load_state_dict(models_state_dict["lc_targ_state_dict"])
            if restore_lagrance_multipliers:
                self.log_alpha = models_state_dict["log_alpha"]
                self.log_labda = models_state_dict["log_labda"]
        else:
            self.use_lyapunov = models_state_dict["use_lyapunov"]
            self.ga.load_state_dict(models_state_dict["ga_state_dict"])
            self.ga_.load_state_dict(models_state_dict["ga_targ_state_dict"])
            self.q_1.load_state_dict(models_state_dict["q1_state_dict"])
            self.q_2.load_state_dict(models_state_dict["q2_state_dict"])
            self.q_1_.load_state_dict(models_state_dict["q1_targ_state_dict"])
            self.q_2_.load_state_dict(models_state_dict["q2_targ_state_dict"])
            if restore_lagrance_multipliers:
                self.log_alpha = models_state_dict["log_alpha"]

        # Return result
        success_load = True
        return success_load

    def _update_targets(self):
        """Updates the target networks based on a Exponential moving average.
        """
        # Polyak averaging for target variables
        with torch.no_grad():
            for pi_main, pi_targ in zip(self.ga.parameters(), self.ga_.parameters()):
                pi_targ.data.mul_(self._polyak)
                pi_targ.data.add_((1 - self._polyak) * pi_main.data)
            if self.use_lyapunov:
                for pi_main, pi_targ in zip(
                    self.lc.parameters(), self.lc_.parameters()
                ):
                    pi_targ.data.mul_(self._polyak)
                    pi_targ.data.add_((1 - self._polyak) * pi_main.data)
            else:
                for pi_main, pi_targ in zip(
                    self.q_1.parameters(), self.q_1_.parameters()
                ):
                    pi_targ.data.mul_(self._polyak)
                    pi_targ.data.add_((1 - self._polyak) * pi_main.data)
                for pi_main, pi_targ in zip(
                    self.q_2.parameters(), self.q_2_.parameters()
                ):
                    pi_targ.data.mul_(self._polyak)
                    pi_targ.data.add_((1 - self._polyak) * pi_main.data)

    def _set_learning_rates(
        self, lr_a=None, lr_alpha=None, lr_l=None, lr_labda=None, lr_c=None
    ):
        """Adjusts the learning rates of the optimizers.

        Args:
            lr_a (float, optional): The learning rate of the actor optimizer. Defaults
                to None.

            lr_alpha (float, optional): The learning rate of the temperature optimizer.
                Defaults to None.

            lr_l (float, optional): The learning rate of the Lyapunov critic. Defaults
                to None.

            lr_labda (float, optional): The learning rate of the Lyapunov Lagrance
                multiplier optimizer. Defaults to None.

            lr_c (float, optional): The learning rate of the Q-Critic optimizer.
                Defaults to None.
        """
        if lr_a:
            for param_group in self._a_train.param_groups:
                param_group["lr"] = lr_a
        if lr_alpha:
            for param_group in self._alpha_train.param_groups:
                param_group["lr"] = lr_alpha
        if self.use_lyapunov:
            if lr_l:
                for param_group in self._l_train.param_groups:
                    param_group["lr"] = lr_l
                for param_group in self._lambda_train.param_groups:
                    param_group["lr"] = lr_labda
        else:
            if lr_c:
                for param_group in self._q_train.param_groups:
                    param_group["lr"] = lr_c

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def labda(self):
        return torch.clamp(self.log_labda.exp(), *SCALE_lambda_MIN_MAX)

    @property
    def act_limits(self):
        return self._act_limits

    @act_limits.setter
    def act_limits(self, act_limits):
        """Sets the action limits that are used for scaling the actions that are
        returned from the gaussian policy.
        """

        # Validate input
        missing_keys = [key for key in ["low", "high"] if key not in act_limits.keys()]
        if missing_keys:
            warn_string = "WARN: act_limits could not be set as {} not found.".format(
                f"keys {missing_keys} were"
                if len(missing_keys) > 1
                else f"key {missing_keys} was"
            )
            print(colorize(warn_string, "yellow"))
        invalid_length = [
            key for key, val in act_limits.items() if len(val) != self._a_dim
        ]
        if invalid_length:
            warn_string = (
                f"WARN: act_limits could not be set as the length of {invalid_length} "
                + "{}".format("were" if len(invalid_length) > 1 else "was")
                + f" unequal to the dimension of the action space (dim={self._a_dim})."
            )
            print(colorize(warn_string, "yellow"))

        # Set action limits
        self._act_limits = {"low": act_limits["low"], "high": act_limits["high"]}
        self.ga.act_limits = self._act_limits


def train(log_dir):
    """Performs the agent training.

    Args:
        log_dir (str): The directory in which the final model (policy) and the
            log data is saved.
    """

    # Create train and test environments
    print(
        colorize(
            f"INFO: You are training in the {ENV_NAME} environment.", "cyan", bold=True,
        )
    )
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

    # Create the Agent
    policy = LAC(a_dim, s_dim, act_limits={"low": a_lowerbound, "high": a_upperbound})

    # Load model if retraining is selected
    if TRAIN_PARAMS["continue_training"]:

        # Create retrain model path
        retrain_model_folder = TRAIN_PARAMS["continue_model_folder"]
        retrain_model_path = osp.abspath(
            osp.join(log_dir, "../../" + TRAIN_PARAMS["continue_model_folder"])
        )

        # Check if retrain model exists if not throw error
        if not osp.exists(retrain_model_path):
            print(
                colorize(
                    (
                        "ERROR: Shutting down training since the model you specified "
                        f"in the `continue_model_folder` `{retrain_model_folder}` "
                        f"argument was not found for the `{ENV_NAME}` environment."
                    ),
                    "red",
                    bold=True,
                )
            )
            sys.exit(0)

        # Load old model
        print(
            colorize(
                f"INFO: Restoring model `{retrain_model_path}`.", "cyan", bold=True
            )
        )
        result = policy.restore(
            osp.abspath(retrain_model_path + "/policy"),
            restore_lagrance_multipliers=(not ALG_PARAMS["reset_lagrance_multipliers"]),
        )
        if not result:
            print(
                colorize(
                    "ERROR: Shuting down training as something went wrong while "
                    "loading "
                    f"model `{retrain_model_folder}`.",
                    "red",
                    bold=True,
                )
            )
            sys.exit(0)

        # Create new storage folder
        log_dir_split = log_dir.split("/")
        log_dir_split[-2] = (
            "_".join(TRAIN_PARAMS["continue_model_folder"].split("/")) + "_finetune"
        )
        log_dir = "/".join(log_dir_split)
    else:
        print(colorize(f"INFO: Train new model `{log_dir}`", "cyan", bold=True))

    # Print logging folder path
    print(colorize(f"INFO: Logging results to `{log_dir}`.", "cyan", bold=True))

    # Create replay memory buffer
    pool = Pool(
        s_dim=s_dim,
        a_dim=a_dim,
        store_last_n_paths=TRAIN_PARAMS["num_of_training_paths"],
        memory_capacity=ALG_PARAMS["memory_capacity"],
        min_memory_size=ALG_PARAMS["min_memory_size"],
        device=policy.device,
    )

    # Setup logger and store/log hyperparameters
    logger.configure(dir=log_dir, format_strs=["csv"])
    logger.logkv("tau", ALG_PARAMS["tau"])
    logger.logkv("alpha3", ALG_PARAMS["alpha3"])
    logger.logkv("batch_size", ALG_PARAMS["batch_size"])
    logger.logkv("target_entropy", policy.target_entropy)
    save_config(locals(), log_dir)

    ####################################################
    # Training loop ####################################
    ####################################################

    # Setup training loop parameters
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=TRAIN_PARAMS["num_of_training_paths"])
    training_started = False

    # Train the agent in the environment until max_episodes has been reached
    print(colorize("INFO: Training...\n", "cyan", bold=True))
    for i in range(ENV_PARAMS["max_episodes"]):

        # Create variable to store information about the current path
        if policy.use_lyapunov:
            current_path = {
                "rewards": [],
                "lyapunov_error": [],
                "alpha": [],
                "lambda": [],
                "entropy": [],
                "a_loss": [],
            }
            # current_path = {
            #     "rewards": torch.tensor([], dtype=torch.float32),
            #     "a_loss": torch.tensor([], dtype=torch.float32),
            #     "alpha": torch.tensor([], dtype=torch.float32),
            #     "lambda": torch.tensor([], dtype=torch.float32),
            #     "lyapunov_error": torch.tensor([], dtype=torch.float32),
            #     "entropy": torch.tensor([], dtype=torch.float32),
            # }  # Improve: Check if this is the fastest way
        else:
            current_path = {
                "rewards": [],
                "critic_error": [],
                "alpha": [],
                "entropy": [],
                "a_loss": [],
            }

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
                    checkpoint_save_path = osp.abspath(
                        osp.join(log_dir, "checkpoints", "step_" + str(j))
                    )
                    os.makedirs(checkpoint_save_path, exist_ok=True)

                    # Save intermediate checkpoint
                    policy.save_result(checkpoint_save_path)

            # Break out of loop if global steps have been reached
            if global_step > ENV_PARAMS["max_global_steps"]:

                # Print step count, save model and stop the program
                print(f"\nINFO: Training stopped after {global_step} steps.")
                print("INFO: Running time: ", time.time() - t1)
                print("INFO: Saving Model")
                policy.save_result(log_dir)
                print("INFO: Running time: ", time.time() - t1)
                return

            # Render environment if requested
            if ENV_PARAMS["eval_render"]:
                env.render()

            # Retrieve (scaled) action based on the current policy
            # NOTE (rickstaa): The scaling operation is already performed inside the
            # policy based on the `act_limits` you supplied.
            a = policy.choose_action(s)

            # Perform action in env
            s_, r, done, _ = env.step(a)

            # Increment global step count
            if training_started:
                global_step += 1

            # Stop episode if max_steps has been reached
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            terminal = 1.0 if done else 0.0

            # Store experience in replay buffer
            pool.store(s, a, r, terminal, s_)

            # Optimize network weights and lagrance multipliers
            if (
                pool.memory_pointer > ALG_PARAMS["min_memory_size"]
                and global_step % ALG_PARAMS["steps_per_cycle"] == 0
            ):
                training_started = True

                # Perform STG a set number of times (train per cycle)
                for _ in range(ALG_PARAMS["train_per_cycle"]):
                    batch = pool.sample(ALG_PARAMS["batch_size"])
                    if policy.use_lyapunov:
                        labda, alpha, l_loss, entropy, a_loss = policy.learn(
                            lr_a_now, lr_l_now, lr_a, lr_c_now, batch
                        )
                    else:
                        alpha, loss_q, entropy, a_loss = policy.learn(
                            lr_a_now, lr_l_now, lr_a, lr_c_now, batch
                        )

            # Store current path results
            if training_started:
                if policy.use_lyapunov:
                    current_path["rewards"].append(r)
                    current_path["lyapunov_error"].append(l_loss.numpy())
                    current_path["alpha"].append(alpha.numpy())
                    current_path["lambda"].append(labda.numpy())
                    current_path["entropy"].append(entropy.numpy())
                    current_path["a_loss"].append(
                        a_loss.numpy()
                    )  # Improve: Check if this is the fastest way
                    # current_path["rewards"] = torch.cat(
                    #     (current_path["rewards"], torch.tensor([r]))
                    # )
                    # current_path["lyapunov_error"] = torch.cat(
                    #     (current_path["lyapunov_error"], torch.tensor([l_loss]))
                    # )
                    # current_path["alpha"] = torch.cat(
                    #     (current_path["alpha"], torch.tensor([alpha]))
                    # )
                    # current_path["lambda"] = torch.cat(
                    #     (current_path["lambda"], torch.tensor([labda]))
                    # )
                    # current_path["entropy"] = torch.cat(
                    #     (current_path["entropy"], torch.tensor([entropy]))
                    # )
                    # current_path["a_loss"] = torch.cat(
                    #     (current_path["a_loss"], torch.tensor([a_loss]))
                    # )
                else:
                    current_path["rewards"].append(r)
                    current_path["critic_error"].append(loss_q.numpy())
                    current_path["alpha"].append(alpha.numpy())
                    current_path["entropy"].append(entropy.numpy())
                    current_path["a_loss"].append(
                        a_loss.numpy()
                    )  # Improve: Check if this is the fastest way

            # Evalute the current policy performance and log the results
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
                    ]  # Improve: Check if this is the fastest way
                    # [
                    #     string_to_print.extend(
                    #         [
                    #             key,
                    #             ":",
                    #             str(
                    #                 (training_diagnostics["length"] * 10 ** 2)
                    #                 .round()
                    #                 .numpy()
                    #                 / (10 ** 2)
                    #             ),
                    #             "|",
                    #         ]
                    #     )
                    #     for key in training_diagnostics.keys()
                    # ]
                    prefix = (
                        colorize("LAC|", "green")
                        if ALG_PARAMS["use_lyapunov"]
                        else colorize("SAC|", "yellow")
                    )
                    print(
                        colorize(prefix, "yellow", bold=True) + "".join(string_to_print)
                    )
                logger.dumpkvs()

            # Update state
            s = s_

            # Check if episode is done (continue to next episode)
            if done:

                # Store paths
                if training_started:
                    last_training_paths.appendleft(current_path)

                # Decay learning rates
                frac = 1.0 - (global_step - 1.0) / ENV_PARAMS["max_global_steps"]
                lr_a_now = lr_a * frac  # learning rate for actor, lambda, alpha
                lr_l_now = lr_l * frac  # learning rate for lyapunov critic
                lr_c_now = lr_c * frac  # learning rate for q critic
                break  # Continue to next episode
