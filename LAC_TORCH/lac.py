"""Minimal working version of the LAC algorithm script."""

import time
from collections import deque
import os
import random
from copy import deepcopy
import os.path as osp
import itertools

import torch

# from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter
import numpy as np

from gaussian_actor import SquashedGaussianMLPActor
from lyapunov_critic import MLPLyapunovCritic
from q_critic import QCritic

from utils import evaluate_training_rollouts, get_env_from_name, training_evaluation
import logger
from pool import Pool

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
    # DEBUG_PARAMS,
)

# Change cudnn backend
# Note: Might speed up your training
# (see: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# Set random seed to get comparable results for each run
if RANDOM_SEED is not None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# TODO: Check if adding variables as tensors speeds up computation


# ###############################################
# # Helper classes and functions  ###############
# ###############################################
# class actor_critic_trace(nn.Module):
#     def __init__(self, ga, lc):
#         super().__init__()
#         self.ga = ga
#         self.lc = lc

#     def forward(self, obs):
#         a, _, _ = self.ga(obs)
#         l = self.lc(obs, a)
#         return l


###############################################
# LAC algorithm class #########################
###############################################
class LAC(object):
    """The lyapunov actor critic.
    """

    def __init__(self, a_dim, s_dim, log_dir="."):
        """Initiate object state.

        Args:
            a_dim (int): Action space dimension.
            s_dim (int): Observation space dimension.
        """

        # Check if GPU is requested and available
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
        print(f"INFO: Torch is using the {device_str}")

        # Save action and observation space as members
        self.a_dim = a_dim
        self.s_dim = s_dim

        # Set algorithm parameters as class objects
        self.network_structure = ALG_PARAMS["network_structure"]
        self.polyak = 1 - ALG_PARAMS["tau"]
        self.use_lyapunov = ALG_PARAMS["use_lyapunov"]

        # Determine target entropy
        if ALG_PARAMS["target_entropy"] is None:
            self.target_entropy = -self.a_dim  # lower bound of the policy entropy
        else:
            self.target_entropy = ALG_PARAMS["target_entropy"]

        # Create Learning rate placeholders
        self.LR_A = torch.tensor(ALG_PARAMS["lr_a"], dtype=torch.float32)
        if self.use_lyapunov:
            self.LR_lag = torch.tensor(ALG_PARAMS["lr_a"], dtype=torch.float32)
            self.LR_L = torch.tensor(ALG_PARAMS["lr_l"], dtype=torch.float32)
        else:
            self.LR_C = torch.tensor(ALG_PARAMS["lr_c"], dtype=torch.float32)

        # Create lagrance multiplier placeholders
        self.log_alpha = torch.tensor(ALG_PARAMS["alpha"], dtype=torch.float32).log()
        self.log_alpha.requires_grad = True  # Enable gradient computation
        if self.use_lyapunov:
            self.log_labda = torch.tensor(
                ALG_PARAMS["labda"], dtype=torch.float32
            ).log()
            self.log_labda.requires_grad = True  # Enable gradient computation

        ###########################################
        # Create Networks #########################
        ###########################################

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
        self.ga = self._build_a().to(self.device)
        if self.use_lyapunov:
            self.lc = self._build_l().to(self.device)
        else:
            self.q_1 = self._build_c().to(self.device)
            self.q_2 = self._build_c().to(self.device)

        # Create GA and LC target networks
        # Don't get optimized but get updated according to the EMA of the main
        # networks
        self.ga_ = deepcopy(self.ga).to(self.device)
        if self.use_lyapunov:
            self.lc_ = deepcopy(self.lc).to(self.device)
        else:
            self.q_1_ = self._build_c().to(self.device)
            self.q_2_ = self._build_c().to(self.device)

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

        # # Create summary writer
        # if DEBUG_PARAMS["use_tb"]:
        #     self.step = 0
        #     self.tb_writer = SummaryWriter(log_dir=log_dir)

        if not self.use_lyapunov:
            q_params = itertools.chain(
                self.q_1.parameters(), self.q_2.parameters()
            )  # Chain parameter iterators so we can pass them to the optimizer

        ###########################################
        # Create optimizers #######################
        ###########################################
        self.alpha_train = Adam([self.log_alpha], lr=self.LR_A)
        self.a_train = Adam(self.ga.parameters(), lr=self.LR_A)
        if self.use_lyapunov:
            self.lambda_train = Adam([self.log_labda], lr=self.LR_lag)
            self.l_train = Adam(self.lc.parameters(), lr=self.LR_L)
        else:
            self.q_train = Adam(q_params, lr=self.LR_C)

        # Create model save dict
        # FIXME: Where is this use?
        if self.use_lyapunov:
            self._save_dict = {"gaussian_actor": self.ga, "lyapunov_critic": self.lc}
        else:
            self._save_dict = {
                "gaussian_actor": self.ga,
                "q_critic_1": self.q_1,
                "q_critic_2": self.q_1,
            }

        # ###########################################
        # # Trace networks (DEBUGGING) ##############
        # ###########################################
        # if DEBUG_PARAMS["use_tb"]:
        #     if DEBUG_PARAMS["trace_net"]:

        #         # Create dummy input
        #         obs = torch.rand((ALG_PARAMS["batch_size"], self.s_dim))

        #         # Write trace to tensorboard
        #         with torch.no_grad():
        #             self.tb_writer.add_graph(actor_critic_trace(self.ga, self.lc), obs)

    def choose_action(self, s, evaluation=False):
        """Returns the current action of the policy.

        Args:
            s (np.numpy): The current state.
            evaluation (bool, optional): Whether to return a deterministic action.
            Defaults to False.

        Returns:
            np.numpy: The current action.
        """
        # TODO: Check return types

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

    def learn(self, LR_A, LR_L, LR_C, LR_lag, batch):
        """Runs the SGD to update all the optimize parameters.

        Args:
            LR_A (float): Current actor learning rate.
            LR_L (float): Lyapunov critic learning rate.
            LR_lag (float): Lyapunov constraint langrance multiplier learning rate.
            batch (numpy.ndarray): The batch of experiences.

        Returns:
            [type]: [description]
        """

        # Adjust optimizer learning rates (decay)
        for param_group in self.a_train.param_groups:
            param_group["lr"] = LR_A
        for param_group in self.alpha_train.param_groups:
            param_group["lr"] = LR_A
        if self.use_lyapunov:
            for param_group in self.l_train.param_groups:
                param_group["lr"] = LR_L
            for param_group in self.lambda_train.param_groups:
                param_group["lr"] = LR_lag
        else:
            for param_group in self.q_train.param_groups:
                param_group["lr"] = LR_C

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Update target networks
        self.update_target()

        # Calculate variables from which we do not require the gradients
        with torch.no_grad():
            if self.use_lyapunov:
                a_, _, _ = self.ga_(bs_)
                l_ = self.lc_(bs_, a_)
                l_target = br + ALG_PARAMS["gamma"] * (1 - bterminal) * l_.detach()
            else:
                # Target actions come from *current* policy
                a2, _, logp_a2 = self.ga(bs_)

                # Target Q-values
                q1_pi_targ = self.q_1_(bs_, a2)
                q2_pi_targ = self.q_2_(bs_, a2)
                q_pi_targ = torch.min(
                    q1_pi_targ, q2_pi_targ
                )  # Use min clipping to prevent overestimation bias (Replaced V by E(Q-H))
                # TODO: Replace log_alpha.exp() with alpha --> Make alpha property
                backup = br + ALG_PARAMS["gamma"] * (1 - bterminal) * (
                    q_pi_targ - self.log_alpha.exp() * logp_a2
                )

        # Calculate current lyapunov Q values
        if self.use_lyapunov:
            l = self.lc(bs, ba)

            # Calculate current value and target lyapunov multiplier value
            lya_a_, _, _ = self.ga(bs_)
            l_ = self.lc(bs_, lya_a_)
        else:
            # Retrieve the Q values from the two networks
            q1 = self.q_1(bs, ba)
            q2 = self.q_2(bs, ba)

        # Calculate log probability of a_input based on current policy
        pi, _, log_pis = self.ga(bs)

        # Calculate Lyapunov constraint function
        if self.use_lyapunov:
            self.l_delta = torch.mean(l_ - l.detach() + (ALG_PARAMS["alpha3"]) * br)

            # Zero gradients on labda
            self.lambda_train.zero_grad()

            # Lagrance multiplier loss functions and optimizers graphs
            labda_loss = -torch.mean(self.log_labda * self.l_delta.detach())

            # Apply gradients to log_lambda
            labda_loss.backward()
            self.lambda_train.step()
        else:
            # Retrieve the current Q values for the action given by the current policy
            q1_pi = self.q_1(bs, pi)
            q2_pi = self.q_2(bs, pi)
            q_pi = torch.min(q1_pi, q2_pi)

        # Zero gradients on alpha
        self.alpha_train.zero_grad()

        # Calculate alpha loss
        alpha_loss = -torch.mean(
            self.log_alpha * log_pis.detach() + self.target_entropy
        )

        # Apply gradients
        alpha_loss.backward()
        self.alpha_train.step()

        # Zero gradients on the actor
        self.a_train.zero_grad()

        # Calculate actor loss
        if self.use_lyapunov:
            # a_loss = self.labda * self.l_delta + self.alpha * torch.mean(log_pis)
            a_loss = self.labda.detach() * self.l_delta + self.alpha.detach() * torch.mean(
                log_pis
            )  # DEBUG
        else:
            a_loss = (self.log_alpha * log_pis - q_pi).mean()

        # Apply gradients
        a_loss.backward()
        self.a_train.step()

        # Optimize critic
        if self.use_lyapunov:
            # Zero gradients on the critic
            self.l_train.zero_grad()

            # Calculate L_backup
            l_error = F.mse_loss(l_target, l)

            # Apply gradients
            l_error.backward()
            self.l_train.step()
        else:

            # Zero gradients on q critic
            self.q_train.zero_grad()

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup) ** 2).mean()
            loss_q2 = ((q2 - backup) ** 2).mean()
            loss_q = loss_q1 + loss_q2

            # Apply graidents
            loss_q.backward()
            self.q_train.step()

        # Return results
        # TODO: return to CPU
        # IMPROVE: Check if this it the right location to do this
        # NOTE: Not needed in SPINNINGUP version as the analysis is alreadyu GPU compatible
        if self.use_lyapunov:
            return (
                # self.labda.detach(),
                # self.alpha.detach(),
                # l_error.detach(),
                # torch.mean(-log_pis.detach()),
                # a_loss.detach(),
                self.labda.cpu().detach(),  # IMPROVE: Not needed in spinning up since the logger can handle GPU data
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

    def _build_a(self, name="gaussian_actor"):
        """Setup SquashedGaussianActor Graph.

        Args:
            name (str, optional): Network name. Defaults to "gaussian_actor".

        Returns:
            tuple: Tuple with network output tensors.
        """
        # Return GA
        return SquashedGaussianMLPActor(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["actor"],
            log_std_min=LOG_SIGMA_MIN_MAX[0],
            log_std_max=LOG_SIGMA_MIN_MAX[1],
        )

    def _build_l(self, name="lyapunov_critic"):
        # TODO: Update docstring
        """Setup lyapunov critic graph.

        Args:
            name (str, optional): Network name. Defaults to "lyapunov_critic".

        Returns:
            tuple: Tuple with network output tensors.
        """
        # Return Lc
        return MLPLyapunovCritic(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["critic"],
        )

    def _build_c(self, name="lyapunov_critic"):
        """Setup q critic .

        Args:
            name (str, optional): Network name. Defaults to "lyapunov_critic".

        Returns:
            tuple: Tuple with network output tensors.
        """
        # Return Lc
        return QCritic(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["critic"],
        )

    def save_result(self, path):
        """Save current policy.

        Args:
            path (str): The path where you want to save the policy.
        """

        # Save all models/tensors in the _save_dict
        save_path = os.path.abspath(path + "/policy/model.pth")

        # Create folder if not exist
        if osp.exists(os.path.dirname(save_path)):
            print(
                "Warning: Log dir %s already exists! Storing info there anyway."
                % os.path.dirname(save_path)
            )
        else:
            os.makedirs(os.path.dirname(save_path))

        # Create models state dict and save
        if self.use_lyapunov:
            models_state_save_dict = {
                "ga_state_dict": self.ga.state_dict(),
                "lc_state_dict": self.lc.state_dict(),
                "ga_targ_state_dict": self.ga_.state_dict(),
                "lc_targ_state_dict": self.lc_.state_dict(),
                "log_alpha": self.log_alpha,
                "log_labda": self.log_labda,
            }
        else:
            models_state_save_dict = {
                "ga_state_dict": self.ga.state_dict(),
                "ga_targ_state_dict": self.ga_.state_dict(),
                "q1_state_dict": self.q_1.state_dict(),
                "q2_state_dict": self.q_2.state_dict(),
                "q1_targ_state_dict": self.q_1_.state_dict(),
                "q2_targ_state_dict": self.q_2_.state_dict(),
                "log_alpha": self.log_alpha,
            }
        torch.save(models_state_save_dict, save_path)
        print("Save to path: ", save_path)

    def restore(self, path):
        # TODO: ADD object which remembers if LAC or SAC was trained!
        """Restore policy.

        Args:
            path (str): The path where you want to save the policy.

        Returns:
            bool: Boolean specifying whether the policy was loaded successfully.
        """

        # Create load path
        load_path = os.path.abspath(path + "/model.pth")

        # Load the model state
        try:
            models_state_dict = torch.load(load_path)
        except NotADirectoryError:
            success_load = False
            return success_load

        # Restore network parameters
        # Question: Do I restore everything correctly?-
        if self.use_lyapunov:  # TODO: RePLACE byu object
            self.ga.load_state_dict(models_state_dict["ga_state_dict"])
            self.lc.load_state_dict(models_state_dict["lc_state_dict"])
            self.ga_.load_state_dict(models_state_dict["ga_targ_state_dict"])
            self.lc_.load_state_dict(models_state_dict["lc_targ_state_dict"])
            self.log_alpha = models_state_dict["log_alpha"]
            self.log_labda = models_state_dict["log_labda"]
        else:
            self.ga.load_state_dict(models_state_dict["ga_state_dict"])
            self.ga_.load_state_dict(models_state_dict["ga_targ_state_dict"])
            self.q_1.load_state_dict(models_state_dict["q1_state_dict"])
            self.q_2.load_state_dict(models_state_dict["q2_state_dict"])
            self.q_1_.load_state_dict(models_state_dict["q1_targ_state_dict"])
            self.q_2_.load_state_dict(models_state_dict["q2_targ_state_dict"])
            self.log_alpha = models_state_dict["log_alpha"]

        # Return result
        success_load = True
        return success_load

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def labda(self):
        return torch.clamp(self.log_labda.exp(), *SCALE_lambda_MIN_MAX)

    def update_target(self):
        # Polyak averaging for target variables
        # TODO: Change name
        # TODO: CHECK IF NEEDED ON THE GPU?
        with torch.no_grad():
            for pi_main, pi_targ in zip(self.ga.parameters(), self.ga_.parameters()):
                pi_targ.data.mul_(self.polyak)
                pi_targ.data.add_((1 - self.polyak) * pi_main.data)
            if self.use_lyapunov:
                for pi_main, pi_targ in zip(
                    self.lc.parameters(), self.lc_.parameters()
                ):
                    pi_targ.data.mul_(self.polyak)
                    pi_targ.data.add_((1 - self.polyak) * pi_main.data)
            else:
                for pi_main, pi_targ in zip(
                    self.q_1.parameters(), self.q_1_.parameters()
                ):
                    pi_targ.data.mul_(self.polyak)
                    pi_targ.data.add_((1 - self.polyak) * pi_main.data)
                for pi_main, pi_targ in zip(
                    self.q_2.parameters(), self.q_2_.parameters()
                ):
                    pi_targ.data.mul_(self.polyak)
                    pi_targ.data.add_((1 - self.polyak) * pi_main.data)


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
    policy = LAC(a_dim, s_dim, log_dir=log_dir)

    # TODO: Fix retrining
    # # Load model if retraining is selected
    # if TRAIN_PARAMS["continue_training"]:

    #     # Create retrain path
    #     retrain_model_folder = TRAIN_PARAMS["continue_model_folder"]
    #     retrain_model_path = os.path.abspath(
    #         os.path.join(log_dir, "../../" + TRAIN_PARAMS["continue_model_folder"])
    #     )

    #     # Check if retrain model exists if not throw error
    #     if not os.path.exists(retrain_model_path):
    #         print(
    #             "Shutting down training since the model you specified in the "
    #             f"`continue_model_folder` `{retrain_model_folder}` "
    #             f"argument was not found for the `{ENV_NAME}` environment."
    #         )
    #         sys.exit(0)

    #     # Load retrain model
    #     print(f"Restoring model `{retrain_model_path}`")
    #     result = policy.restore(os.path.abspath(retrain_model_path + "/policy"))
    #     if not result:
    #         print(
    #             "Shuting down training as something went wrong while loading "
    #             f"model `{retrain_model_folder}`."
    #         )
    #         sys.exit(0)

    #     # Create new storage folder
    #     log_dir_split = log_dir.split("/")
    #     log_dir_split[-2] = (
    #         "_".join(TRAIN_PARAMS["continue_model_folder"].split("/"))
    #         + "_finetune"
    #         # + "_retrained_"
    #         # + log_dir_split[-2]
    #     )
    #     log_dir = "/".join(log_dir_split)

    #     # Reset lagrance multipliers if requested
    #     if ALG_PARAMS["reset_lagrance_multipliers"]:
    #         policy.sess.run(policy.log_alpha.assign(tf.math.log(ALG_PARAMS["alpha"])))
    #         policy.sess.run(policy.log_labda.assign(tf.math.log(ALG_PARAMS["labda"])))
    # else:
    #     print(f"Train new model `{log_dir}`")

    # Print logging folder
    print(f"Logging results to `{log_dir}`.")

    # Create replay memory buffer
    pool = Pool(
        s_dim=s_dim,
        a_dim=a_dim,
        store_last_n_paths=TRAIN_PARAMS["num_of_training_paths"],
        memory_capacity=ALG_PARAMS["memory_capacity"],
        min_memory_size=ALG_PARAMS["min_memory_size"],
        device=policy.device,
    )

    # # Log initial values to tensorboard
    # if DEBUG_PARAMS["use_tb"]:
    #     policy.tb_writer.add_scalar("lr_a", lr_a_now, global_step=0)
    #     policy.tb_writer.add_scalar("lr_l", lr_l_now, global_step=0)
    #     policy.tb_writer.add_scalar("lr_lag", lr_a, global_step=0)
    #     policy.tb_writer.add_scalar("alpha", policy.alpha, global_step=0)
    #     policy.tb_writer.add_scalar("lambda", policy.labda, global_step=0)

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
                "alpha": [],
                "lambda": [],
                "entropy": [],
                "a_loss": [],
            }  # DEBUG
            # current_path = {
            #     "rewards": torch.tensor([], dtype=torch.float32),
            #     "a_loss": torch.tensor([], dtype=torch.float32),
            #     "alpha": torch.tensor([], dtype=torch.float32),
            #     "lambda": torch.tensor([], dtype=torch.float32),
            #     "lyapunov_error": torch.tensor([], dtype=torch.float32),
            #     "entropy": torch.tensor([], dtype=torch.float32),
            # }  # DEBUG: Check if this is the fastest way
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
                        labda, alpha, l_loss, entropy, a_loss = policy.learn(
                            lr_a_now, lr_l_now, lr_a, lr_c_now, batch
                        )
                    else:
                        alpha, loss_q, entropy, a_loss = policy.learn(
                            lr_a_now, lr_l_now, lr_a, lr_c_now, batch
                        )

            # Save path results
            if training_started:
                if policy.use_lyapunov:
                    current_path["rewards"].append(r)
                    current_path["lyapunov_error"].append(l_loss.numpy())
                    current_path["alpha"].append(alpha.numpy())
                    current_path["lambda"].append(labda.numpy())
                    current_path["entropy"].append(entropy.numpy())
                    current_path["a_loss"].append(
                        a_loss.numpy()
                    )  # DEBUG: Check if this is the fastest way
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
                    )  # DEBUG: Check if this is the fastest way

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
                    ]  # DEBUG: Check if this is the fastest way
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
                    print("".join(string_to_print))
                logger.dumpkvs()

            # Update state
            s = s_

            # Decay learning rate
            if done:

                # Store paths
                if training_started:
                    last_training_paths.appendleft(current_path)

                    # # Get current model performance for tb
                    # if DEBUG_PARAMS["use_tb"]:
                    #     training_diagnostics = evaluate_training_rollouts(
                    #         last_training_paths
                    #     )

                # # Log tb variables
                # if DEBUG_PARAMS["use_tb"]:
                #     if i % DEBUG_PARAMS["tb_freq"] == 0:

                #         # Log learning rate to tb
                #         policy.tb_writer.add_scalar(
                #             "lr_a", lr_a_now, global_step=policy.step
                #         )
                #         policy.tb_writer.add_scalar(
                #             "lr_l", lr_l_now, global_step=policy.step
                #         )
                #         policy.tb_writer.add_scalar(
                #             "lr_lag", lr_a, global_step=policy.step
                #         )
                #         policy.tb_writer.add_scalar(
                #             "alpha", policy.alpha, global_step=policy.step
                #         )
                #         policy.tb_writer.add_scalar(
                #             "lambda", policy.labda, global_step=policy.step
                #         )

                #         # Update and log other training vars to tensorboard
                #         if training_started:
                #             policy.tb_writer.add_scalar(
                #                 "ep_ret",
                #                 training_diagnostics["return"],
                #                 global_step=policy.step,
                #             )
                #             policy.tb_writer.add_scalar(
                #                 "ep_length",
                #                 training_diagnostics["length"],
                #                 global_step=policy.step,
                #             )
                #             policy.tb_writer.add_scalar(
                #                 "a_loss",
                #                 training_diagnostics["a_loss"],
                #                 global_step=policy.step,
                #             )
                #             policy.tb_writer.add_scalar(
                #                 "lyapunov_error",
                #                 training_diagnostics["lyapunov_error"],
                #                 global_step=policy.step,
                #             )
                #             policy.tb_writer.add_scalar(
                #                 "entropy",
                #                 training_diagnostics["entropy"],
                #                 global_step=policy.step,
                #             )

                #             # Log network weights
                #             if DEBUG_PARAMS["write_w_b"]:

                #                 # GaussianActor weights/biases
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/l1/weights",
                #                     policy.ga.net[0].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/l1/bias",
                #                     policy.ga.net[0].bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/l2/weights",
                #                     policy.ga.net[2].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/l2/bias",
                #                     policy.ga.net[2].bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/mu/weights",
                #                     policy.ga.mu.weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/mu/bias",
                #                     policy.ga.mu.bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/log_sigma/weights",
                #                     policy.ga.log_sigma.weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga/log_sigma/bias",
                #                     policy.ga.log_sigma.bias,
                #                     global_step=policy.step,
                #                 )

                #                 # Target GaussianActor weights/biases
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/l1/weights",
                #                     policy.ga_.net[0].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/l1/bias",
                #                     policy.ga_.net[0].bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/l2/weights",
                #                     policy.ga_.net[2].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/l2/bias",
                #                     policy.ga_.net[2].bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/mu/weights",
                #                     policy.ga_.mu.weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/mu/bias",
                #                     policy.ga_.mu.bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/log_sigma/weights",
                #                     policy.ga_.log_sigma.weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Ga_/log_sigma/bias",
                #                     policy.ga_.log_sigma.bias,
                #                     global_step=policy.step,
                #                 )

                #                 # Lyapunov critic weights/biases
                #                 policy.tb_writer.add_histogram(
                #                     "Lc/net/l1/weights",
                #                     policy.lc.l[0].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Lc/net/l1/bias",
                #                     policy.lc.l[0].bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Lc/net/l2/weights",
                #                     policy.lc.l[2].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Lc/net/l2/bias",
                #                     policy.lc.l[2].bias,
                #                     global_step=policy.step,
                #                 )

                #                 # Target Lyapunov critic weights/biases
                #                 policy.tb_writer.add_histogram(
                #                     "Lc_/net/l1/weights",
                #                     policy.lc_.l[0].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Lc_/net/l1/bias",
                #                     policy.lc_.l[0].bias,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Lc_/net/l2/weights",
                #                     policy.lc_.l[2].weight,
                #                     global_step=policy.step,
                #                 )
                #                 policy.tb_writer.add_histogram(
                #                     "Lc_/net/l2/bias",
                #                     policy.lc_.l[2].bias,
                #                     global_step=policy.step,
                #                 )

                # Decay learning rates
                frac = 1.0 - (global_step - 1.0) / ENV_PARAMS["max_global_steps"]
                lr_a_now = lr_a * frac  # learning rate for actor, lambda, alpha
                lr_l_now = lr_l * frac  # learning rate for lyapunov critic
                lr_c_now = lr_c * frac  # learning rate for q critic
                break  # FIXME: Redundant
