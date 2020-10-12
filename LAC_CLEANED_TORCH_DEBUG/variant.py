"""File containing the algorithm parameters.
"""

import sys
import os
import time

REL_PATH = False  # DEBUG: Whether to use a relative path for storign and loading models
# REL_PATH = True  # Whether to use a relative path for storign and loading models
USE_GPU = False

# episodes = int(0.1e4)  # DEBUG
episodes = int(1.1e4)
num_of_policies = 1
num_of_paths_for_eval = 50
eval_list = ["LAC20201004_2339"]
use_lyapunov = True
# use_lyapunov = False
which_policy_for_inference = [
    0
]  # If this is empty, it means all the policies are evaluated;
continue_training = (
    False  # Whether we want to continue training an already trained model
)
continue_model_folder = "LAC20201004_2130/0"  # The path of the model for which you want to continue the training
reset_lagrance_multipliers = False  # Whether you want the lagrance multipliers to be reset when you continue training an old model
save_checkpoints = False  # Store intermediate models
checkpoint_save_freq = 10000  # Intermediate model save frequency

# Environment parameters
# ENV_NAME = "Ex3_EKF_gyro_dt_real"  # The gym environment you want to train in
# ENV_NAME = "Ex3_EKF_gyro"  # The gym environment you want to train in
# ENV_NAME = "Ex3_EKF_gyro_dt"  # The gym environment you want to train in
ENV_NAME = "oscillator"  # The gym environment you want to train in
ENV_SEED = 0  # The environment seed
RANDOM_SEED = 0  # The numpy random seed

# Setup log path and time string
alg_prefix = "LAC" if use_lyapunov else "SAC"
if REL_PATH:
    LOG_PATH = "/".join(["./log", ENV_NAME, alg_prefix + time.strftime("%Y%m%d_%H%M")])
else:
    dirname = os.path.dirname(__file__)
    LOG_PATH = os.path.abspath(
        os.path.join(
            dirname, "./log/" + ENV_NAME, alg_prefix + time.strftime("%Y%m%d_%H%M")
        )
    )
timestr = time.strftime("%Y%m%d_%H%M")

# Debug Parameters
DEBUG_PARAMS = {
    "trace_net": False,  # Whether we want to trace the network.
    "use_tb": False,  # Whether you want to log to tensorboard
    "tb_freq": 4,  # After how many episode we want to log to tensorboard
    "write_w_b": False,  # Whether you want to log the model weights and biases
}

# Main training loop parameters
TRAIN_PARAMS = {
    "episodes": episodes,  # The number of episodes you want to perform
    "num_of_training_paths": 10,  # Number of training rollouts stored for analysis
    "evaluation_frequency": 2048,  # After how many steps the performance is evaluated
    "num_of_evaluation_paths": 0,  # number of rollouts for evaluation  # DEBUG
    "num_of_trials": num_of_policies,  # number of randomly seeded trained agents # TODO: CHANGE NAME to NUM_OF_ROLLOUTS
    "start_of_trial": 0,  # The start number of the rollouts (used during model save)
}

# Main evaluation parameters
EVAL_PARAMS = {
    "which_policy_for_inference": which_policy_for_inference,  # Which policies you want to use for the inference
    "eval_list": eval_list,
    "additional_description": timestr,
    "num_of_paths": num_of_paths_for_eval,  # number of path for evaluation
    "plot_average": True,
    "directly_show": True,
    "plot_ref": True,  # Whether you also want to plot the states of reference
    "merged": True,  # Whether you want to display all the states of references in one fig
    "ref": [],  # Which state of reference you want to plot (empty means all obs).
    "plot_obs": True,  # Whether you also want to plot the observations
    "obs": [],  # Which observations you want to plot (empty means all obs).
    "plot_cost": True,  # Whether you also want to plot the cost
    "save_figs": True,  # Whether you want to save the figures to pdf.
    "fig_file_type": "pdf",  # The file type you want to use for saving the figures.
}

# Learning algorithm parameters
ALG_PARAMS = {
    "memory_capacity": int(1e6),  # The max replay buffer size
    "min_memory_size": 1000,  # The minimum replay buffer size before STG starts
    "batch_size": 256,  # The SGD batch size
    # "labda": 1.0,  # Initial value for the lyapunov constraint lagrance multiplier
    "labda": 0.99,  # Initial value for the lyapunov constraint lagrance multiplier
    # "alpha": 1.0,  # The initial value for the entropy lagrance multiplier
    "alpha": 0.99,  # The initial value for the entropy lagrance multiplier
    "alpha3": 0.2,  # The value of the stability condition multiplier
    "tau": 5e-3,  # Decay rate used in the polyak averaging
    "lr_a": 1e-4,  # The actor learning rate
    "lr_l": 3e-4,  # The lyapunov critic
    "gamma": 0.9,  # Discount factor
    "steps_per_cycle": 100,  # The number of steps after which the model is trained
    "train_per_cycle": 80,  # How many times SGD is called during the training
    "adaptive_alpha": True,  # Enables automatic entropy temperature tuning
    "target_entropy": None,  # Set alpha target entropy, when None == -(action_dim)
    "network_structure": {
        "critic": [128, 128],
        "actor": [64, 64],
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
    "Ex3_EKF": {
        "max_ep_steps": 500,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex4_EKF": {
        "max_ep_steps": 100,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
}

# Check if specified environment is valid
if ENV_NAME in ENVS_PARAMS.keys():
    ENV_PARAMS = ENVS_PARAMS[ENV_NAME]
else:
    print(
        f"Environmen {ENV_NAME} does not exist yet. Please specify a valid environment "
        "and try again."
    )
    sys.exit(0)

# Other paramters
LOG_SIGMA_MIN_MAX = (-20, 2)  # Range of log std coming out of the GA network
SCALE_lambda_MIN_MAX = (0, 1)  # Range of lambda lagrance multiplier
