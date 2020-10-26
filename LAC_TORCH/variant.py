"""File containing the algorithm parameters.
"""

import sys
import os.path as osp
import time

from utils import colorize

########################################################
# Main parameters ######################################
########################################################

# General parameters
REL_PATH = False  # Use relative paths
USE_GPU = False  # Use GPU
ENV_SEED = 0  # The environment seed
RANDOM_SEED = 0  # The script random seed

# Environment parameters
ENV_NAME = "oscillator"  # The environment used for training

# Training parameters
episodes = int(1.1e4)  # Max episodes
num_of_policies = 1  # Number of randomly seeded trained agents
use_lyapunov = False  # Use LAC (If false SAC is used)
continue_training = (
    True  # Whether we want to continue training an already trained model
)
continue_model_folder = "SAC20201026_1010/0"  # Which model you want to use
reset_lagrance_multipliers = False  # Reset lagrance multipliers before retraining
save_checkpoints = False  # Store intermediate models
checkpoint_save_freq = 10000  # Intermediate model save frequency

# Evaluation parameters
eval_list = ["SAC20201026_1010"]
which_policy_for_inference = [
    0
]  # If this is empty, it means all the policies are evaluated;
num_of_paths_for_eval = 10  # How many paths you want to perform for each policy


########################################################
# Other parameters #####################################
########################################################

# Setup log path and time string
alg_prefix = "LAC" if use_lyapunov else "SAC"
if REL_PATH:
    LOG_PATH = "/".join(
        ["./log", ENV_NAME.lower(), alg_prefix + time.strftime("%Y%m%d_%H%M")]
    )
else:
    dirname = osp.dirname(__file__)
    LOG_PATH = osp.abspath(
        osp.join(
            dirname,
            "./log/" + ENV_NAME.lower(),
            alg_prefix + time.strftime("%Y%m%d_%H%M"),
        )
    )
timestr = time.strftime("%Y%m%d_%H%M")

# Training parameters
TRAIN_PARAMS = {
    "episodes": episodes,
    "num_of_policies": num_of_policies,
    "continue_training": continue_training,
    "continue_model_folder": continue_model_folder,
    "save_checkpoints": save_checkpoints,
    "checkpoint_save_freq": checkpoint_save_freq,
    "num_of_training_paths": 100,  # Number of episodes used in the performance analysis
    "evaluation_frequency": 4000,  # After how many steps the performance is evaluated
    "num_of_evaluation_paths": 20,  # Rollouts use for test performance analysis
    "start_of_trial": 0,  # The start number of the rollouts (used during model save)
}

# Inference parameters
EVAL_PARAMS = {
    "which_policy_for_inference": which_policy_for_inference,
    "eval_list": eval_list,
    "additional_description": timestr,
    "num_of_paths": num_of_paths_for_eval,
    "plot_average": True,
    "directly_show": True,
    "plot_soi": True,  # Plot the states of interest and the corresponding references.
    "sio_merged": True,  # Display all the states of interest in one figure.
    "soi": [],  # Which state of interest you want to plot (empty means all sio).
    "soi_title": "True and Estimated Quatonian",  # SOI figure title.
    "plot_obs": True,  # Plot the observations.
    "obs_merged": True,  # Display all the obserations in one figure.
    "obs": [],  # Which observations you want to plot (empty means all obs).
    "obs_title": "Observations",  # Obs figure title.
    "plot_cost": True,  # Plot the cost.
    "cost_title": "Mean cost",  # TCost figure title.
    "save_figs": True,  # Save the figures to pdf.
    "fig_file_type": "pdf",  # The file type you want to use for saving the figures.
}

# Learning algorithm parameters
ALG_PARAMS = {
    "use_lyapunov": use_lyapunov,
    "reset_lagrance_multipliers": reset_lagrance_multipliers,
    "memory_capacity": int(1e6),  # The max replay buffer size
    "min_memory_size": 1000,  # The minimum replay buffer size before STG starts
    "batch_size": 256,  # The SGD batch size
    "labda": 1.0,  # Initial value for the lyapunov constraint lagrance multiplier
    "alpha": 1.0,  # The initial value for the entropy lagrance multiplier
    "alpha3": 0.1,  # The value of the stability condition multiplier
    "tau": 5e-3,  # Decay rate used in the polyak averaging
    "lr_a": 1e-4,  # The actor learning rate
    "lr_l": 3e-4,  # The lyapunov critic
    "lr_c": 3e-4,  # The SAC critic
    "gamma": 0.999,  # Discount factor
    "steps_per_cycle": 100,  # The number of steps after which the model is trained
    "train_per_cycle": 80,  # How many times SGD is called during the training
    "adaptive_alpha": True,  # Enables automatic entropy temperature tuning
    "target_entropy": None,  # Set alpha target entropy, when None == -(action_dim)
    "network_structure": {
        "critic": [128, 64, 32],  # Lyapunov Critic
        "actor": [128, 64, 32],  # Gaussian actor
        "q_critic": [128, 64, 32],  # Q-Critic
    },  # The network structure of the agent.
}

# Environment parameters
ENVS_PARAMS = {
    "oscillator": {
        "max_ep_steps": 800,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex3_EKF_gyro": {
        "max_ep_steps": 800,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex3_EKF_gyro_dt": {
        "max_ep_steps": 120,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex3_EKF_gyro_dt_real": {
        "max_ep_steps": 1000,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
}

# Other paramters
SCALE_lambda_MIN_MAX = (0, 1)  # Range of lambda lagrance multiplier

# Check if specified environment is valid
if ENV_NAME in ENVS_PARAMS.keys():
    ENV_PARAMS = ENVS_PARAMS[ENV_NAME]
else:
    print(
        colorize(
            (
                f"ERROR: Environmen {ENV_NAME} does not exist yet. Please specify a "
                "valid environment and try again."
            ),
            "red",
            bold=True,
        )
    )
    sys.exit(0)
