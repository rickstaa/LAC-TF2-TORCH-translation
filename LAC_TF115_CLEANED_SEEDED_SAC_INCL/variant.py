"""File containing the algorithm parameters.
"""

import sys
import os
import time

REL_PATH = False  # DEBUG: Whether to use a relative path for storign and loading models
# REL_PATH = True  # Whether to use a relative path for storign and loading models
USE_GPU = False

episodes = int(1e5)  # DEBUG
# episodes = int(2e5)
num_of_paths_for_eval = 20
num_of_policies = 10
# eval_list = ["LAC20201002_1200"] # DEBUG
eval_list = ["SAC20201002_1903"]
# use_lyapunov = True
use_lyapunov = False
which_policy_for_inference = [0]

# Environment parameters
ENV_NAME = "Ex3_EKF_gyro"  # The gym environment you want to train in
# ENV_NAME = "oscillator"  # The gym environment you want to train in
ENV_SEED = None  # The environment seed
RANDOM_SEED = None  # The numpy random seed

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

# Main training loop parameters
TRAIN_PARAMS = {
    "episodes": episodes,  # The number of episodes you want to perform
    "num_of_training_paths": 100,  # Number of training rollouts stored for analysis
    "evaluation_frequency": 2048,  # After how many steps the performance is evaluated
    "num_of_evaluation_paths": 10,  # number of rollouts for evaluation  # DEBUG
    # "num_of_evaluation_paths": 0,  # number of rollouts for evaluation
    # "num_of_trials": 4,  # number of randomly seeded trained agents
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
}

# Learning algorithm parameters
ALG_PARAMS = {
    "use_lyapunov": use_lyapunov,  # If false the SAC algorithm will be used
    "memory_capacity": int(1e6),  # The max replay buffer size
    "min_memory_size": 1000,  # The minimum replay buffer size before STG starts
    "batch_size": 256,  # The SGD batch size
    "labda": 1.0,  # Initial value for the lyapunov constraint lagrance multiplier
    "alpha": 1.0,  # The initial value for the entropy lagrance multiplier
    "alpha3": 0.2,  # The value of the stability condition multiplier
    "tau": 5e-3,  # Decay rate used in the polyak averaging
    "lr_a": 1e-4,  # The actor learning rate
    "lr_l": 3e-4,  # The lyapunov critic
    "lr_c": 3e-4,  # The lyapunov critic
    "gamma": 0.999,  # Discount factor
    "steps_per_cycle": 100,  # The number of steps after which the model is trained
    "train_per_cycle": 80,  # How many times SGD is called during the training
    "adaptive_alpha": True,  # Enables automatic entropy temperature tuning
    "target_entropy": None,  # Set alpha target entropy, when None == -(action_dim)
    "network_structure": {
        "critic": [128, 64, 32],  # LAC
        "actor": [128, 64, 32],
        "q_critic": [128, 64, 32],  # SAC
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
