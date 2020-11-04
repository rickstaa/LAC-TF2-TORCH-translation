"""File containing the algorithm parameters.
"""

# IMPROVEMENT: Replace with yaml config file

########################################################
# Main parameters ######################################
########################################################

# Debug parameters
DEBUG_PARAMS = {"debug": False}  # Disabled tf.functions

# General parameters
REL_PATH = False  # Use relative paths
USE_GPU = True  # Use GPU
ENV_SEED = 0  # The environment seed
RANDOM_SEED = 0  # The script random seed

# Environment parameters
ENV_NAME = "oscillator"  # The environment used for training

# Training parameters
EPISODES = int(1e5)  # Max episodes
NUM_OF_POLICIES = 5  # Number of randomly seeded trained agents
USE_LYAPUNOV = True  # Use LAC (If false SAC is used)
CONTINUE_TRAINING = (
    False  # Whether we want to continue training an already trained model
)
CONTINUE_MODEL_FOLDER = "SAC20201101_1026/0"  # Which model you want to use
RESET_LAGRANCE_MULTIPLIERS = True  # Reset lagrance multipliers before retraining
SAVE_CHECKPOINTS = False  # Store intermediate models
CHECKPOINT_SAVE_FREQ = 10000  # Intermediate model save frequency

# Evaluation parameters
EVAL_LIST = ["SAC20201101_1026"]
WHICH_POLICY_FOR_INFERENCE = [
    0
]  # If this is empty, it means all the policies are evaluated;
NUM_OF_PATHS_FOR_EVAL = 100  # How many paths you want to perform during inference.

########################################################
# Other parameters #####################################
########################################################

# Training parameters
TRAIN_PARAMS = {
    "episodes": EPISODES,
    "num_of_policies": NUM_OF_POLICIES,
    "continue_training": CONTINUE_TRAINING,
    "continue_model_folder": CONTINUE_MODEL_FOLDER,
    "save_checkpoints": SAVE_CHECKPOINTS,
    "checkpoint_save_freq": CHECKPOINT_SAVE_FREQ,
    "num_of_training_paths": 10,  # Number of episodes used in the performance analysis
    "evaluation_frequency": 4000,  # After how many steps the performance is evaluated
    "num_of_evaluation_paths": 20,  # Rollouts use for test performance analysis
    "start_of_trial": 0,  # The start number of the rollouts (used during model save)
}

# Inference parameters
EVAL_PARAMS = {
    "which_policy_for_inference": WHICH_POLICY_FOR_INFERENCE,
    "eval_list": EVAL_LIST,
    "num_of_paths": NUM_OF_PATHS_FOR_EVAL,
    "plot_average": True,
    "directly_show": True,
    "plot_soi": True,  # Plot the states of interest and the corresponding references.
    "sio_merged": True,  # Display all the states of interest in one figure.
    "soi": [],  # Which state of interest you want to plot (empty means all sio).
    "soi_title": "",  # SOI figure title.
    "plot_obs": True,  # Plot the observations.
    "obs_merged": True,  # Display all the obserations in one figure.
    "obs": [],  # Which observations you want to plot (empty means all obs).
    "obs_title": "",  # Obs figure title.
    "plot_cost": True,  # Plot the costs.
    "costs_merged": True,  # Display all the costs in one figure.
    "costs": [],  # Which costs you want to plot (empty means all obs).
    "costs_title": "",  # TCost figure title.
    "save_figs": True,  # Save the figures to pdf.
    "fig_file_type": "pdf",  # The file type you want to use for saving the figures.
}

# Learning algorithm parameters
ALG_PARAMS = {
    "use_lyapunov": USE_LYAPUNOV,
    "reset_lagrance_multipliers": RESET_LAGRANCE_MULTIPLIERS,
    "memory_capacity": int(1e6),  # The max replay buffer size
    "min_memory_size": 1000,  # The minimum replay buffer size before STG starts
    "batch_size": 256,  # The SGD batch size
    "labda": 0.99,  # Initial value for the Lyapunov constraint lagrance multiplier
    "alpha": 0.99,  # The initial value for the entropy lagrance multiplier
    "alpha3": 0.2,  # The value of the stability condition multiplier
    "tau": 5e-3,  # Decay rate used in the polyak averaging
    "lr_a": 1e-4,  # The actor learning rate
    "lr_l": 3e-4,  # The Lyapunov critic
    "lr_c": 3e-4,  # The SAC critic
    "gamma": 0.9,  # Discount factor
    "steps_per_cycle": 100,  # The number of steps after which the model is trained
    "train_per_cycle": 80,  # How many times SGD is called during the training
    "adaptive_alpha": True,  # Enables automatic entropy temperature tuning
    "target_entropy": None,  # Set alpha target entropy, when None == -(action_dim)
    "network_structure": {
        "critic": [128, 128],  # Lyapunov Critic
        "actor": [64, 64],  # Gaussian actor
        "q_critic": [128, 128],  # Q-Critic
    },  # The network structure of the agent.
}

# Environment parameters
# IMPROVEMENT: Place in its own configuration file.
# IMPROVEMENT: Create python module and register as gym environments.
ENVS_PARAMS = {
    "oscillator": {
        "module_name": "envs.oscillator",
        "class_name": "oscillator",
        "max_ep_steps": 800,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "ex3_ekf_gyro": {
        "module_name": "envs.Ex3_EKF_gyro",
        "class_name": "Ex3_EKF_gyro",
        "max_ep_steps": 800,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "ex3_ekf_gyro_dt": {
        "module_name": "envs.ex3_ekf_gyro_dt",
        "class_name": "Ex3_EKF_gyro",
        "max_ep_steps": 120,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "ex3_ekf_gyro_dt_real": {
        "module_name": "envs.ex3_ekf_gyro_dt_real",
        "class_name": "Ex3_EKF_gyro",
        "max_ep_steps": 1000,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
}

# Other paramters
SCALE_lambda_MIN_MAX = (0, 1)  # Range of lambda lagrance multiplier

# Check if specified environment is valid
ENVS_PARAMS = {
    key.lower(): val for key, val in ENVS_PARAMS.items()
}  # Make keys lowercase to prevent typing errors
