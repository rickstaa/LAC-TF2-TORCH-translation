import gym
import datetime
import numpy as np
import ENV.env
import time
import os

REL_PATH = True  # Whether to use a relative path for storign and loading models
USE_GPU = False

# episodes = int(1.1e4)  # DEBUG
episodes = int(1e5)
num_of_paths_for_eval = 20
num_of_policies = 5
# num_of_policies = 1 # DEBUG

ENV_SEED = 0
RANDOM_SEED = 0

# alpha = 1.0
alpha = 0.99
alpha3 = 0.2
actor = [128, 64, 32]
critic = [128, 64, 32]
approx_value = True
use_lyapunov = True
timestr = time.strftime("%Y%m%d_%H%M")

VARIANT = {
    "eval_list": ["LAC20201002_0852"],
    # "env_name": "Ex3_EKF",
    "env_name": "Ex3_EKF_gyro",
    # "env_name": "oscillator",
    "algorithm_name": "LAC",
    "additional_description": timestr,
    # 'evaluate': False,
    "train": True,
    # 'train': False,
    "num_of_trials": num_of_policies,  # number of random seeds
    # "num_of_evaluation_paths": 10,  # number of rollouts for evaluation  # DEBUG
    "num_of_evaluation_paths": 0,  # number of rollouts for evaluation
    "num_of_training_paths": 10,  # number of training rollouts stored for analysis
    "start_of_trial": 0,
    "evaluation_form": "dynamic",
    "trials_for_eval": [str(i) for i in range(0, 3)],
    "evaluation_frequency": 2048,
}
if VARIANT["algorithm_name"] == "RARL":
    ITA = 0
if REL_PATH:
    VARIANT["log_path"] = "/".join(
        [
            "./log",
            VARIANT["env_name"],
            VARIANT["algorithm_name"] + VARIANT["additional_description"],
        ]
    )
else:
    dirname = os.path.dirname(__file__)
    VARIANT["log_path"] = os.path.abspath(
        os.path.join(
            dirname,
            "./log/" + VARIANT["env_name"],
            "LAC" + time.strftime("%Y%m%d_%H%M"),
        )
    )

ENV_PARAMS = {
    "oscillator": {
        "max_ep_steps": 800,
        "max_global_steps": episodes,
        "max_episodes": int(1e6),
        "disturbance dim": 2,
        "eval_render": False,
        "network_structure": {"critic": critic, "actor": actor,},
    },
    "Ex3_EKF": {
        "max_ep_steps": 500,
        "max_global_steps": episodes,
        "max_episodes": int(1e6),
        "disturbance dim": 2,
        "eval_render": False,
        "network_structure": {"critic": critic, "actor": actor,},
    },
    "Ex3_EKF_gyro": {
        "max_ep_steps": 800,
        "max_global_steps": episodes,
        "max_episodes": int(1e6),
        "disturbance dim": 2,
        "eval_render": False,
        "network_structure": {"critic": critic, "actor": actor,},
    },
    "Ex4_EKF": {
        "max_ep_steps": 100,
        "max_global_steps": episodes,
        "max_episodes": int(1e6),
        "disturbance dim": 2,
        "eval_render": False,
        "network_structure": {"critic": critic, "actor": actor,},
    },
}
ALG_PARAMS = {
    "MPC": {"horizon": 5,},
    "LQR": {"use_Kalman": False,},
    "LAC": {
        "iter_of_actor_train_per_epoch": 50,
        "iter_of_disturber_train_per_epoch": 50,
        "memory_capacity": int(1e6),
        "min_memory_size": 1000,
        "batch_size": 256,
        "labda": 1.0,
        "alpha": alpha,
        "alpha3": alpha3,
        "tau": 5e-3,
        "lr_a": 1e-4,
        "lr_c": 3e-4,
        "lr_l": 3e-4,
        "gamma": 0.999,
        "steps_per_cycle": 100,
        "train_per_cycle": 80,
        "use_lyapunov": use_lyapunov,
        "adaptive_alpha": True,
        # 'approx_value': False,
        "approx_value": approx_value,
        "value_horizon": 2,
        # 'finite_horizon': True,
        "finite_horizon": False,
        "soft_predict_horizon": False,
        "target_entropy": None,
        "history_horizon": 0,  # 0 is using current state only
    },
}

EVAL_PARAMS = {
    "param_variation": {
        "param_variables": {
            "mass_of_pole": np.arange(0.05, 0.55, 0.05),  # 0.1
            "length_of_pole": np.arange(0.1, 2.1, 0.1),  # 0.5
            "mass_of_cart": np.arange(0.1, 2.1, 0.1),  # 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1
        },
        "grid_eval": True,
        # 'grid_eval': False,
        "grid_eval_param": ["length_of_pole", "mass_of_cart"],
        "num_of_paths": 100,  # number of path for evaluation
    },
    "impulse": {
        "magnitude_range": np.arange(80, 155, 5),
        "num_of_paths": 100,  # number of path for evaluation
        "impulse_instant": 200,
    },
    "constant_impulse": {
        "magnitude_range": np.arange(0.1, 1.0, 0.1),
        "num_of_paths": 20,  # number of path for evaluation
        "impulse_instant": 20,
    },
    "various_disturbance": {
        "form": ["sin", "tri_wave"][0],
        "period_list": np.arange(2, 11, 1),
        # 'magnitude': np.array([1, 1, 1, 1, 1, 1]),
        "magnitude": np.array([80]),
        # 'grid_eval': False,
        "num_of_paths": 100,  # number of path for evaluation
    },
    "trained_disturber": {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        "path": "./log/HalfCheetahcost-v0/RLAC-horizon=inf-dis=.1/0/",
        "num_of_paths": 100,  # number of path for evaluation
    },
    "dynamic": {
        "additional_description": "original",
        "num_of_paths": num_of_paths_for_eval,  # number of path for evaluation
        "plot_average": True,
        # "plot_average": False,
        "directly_show": True,
        "plot_ref": True,  # Whether you also want to plot the states of reference
        "merged": True,  # Whether you want to display all the states of references in one fig
        "ref": [],  # Which state of reference you want to plot (empty means all obs).
        "plot_obs": True,  # Whether you also want to plot the observations
        "obs": [],  # Which observations you want to plot (empty means all obs).
        "plot_cost": True,  # Whether you also want to plot the cost
    },
}
VARIANT["env_params"] = ENV_PARAMS[VARIANT["env_name"]]
VARIANT["eval_params"] = EVAL_PARAMS[VARIANT["evaluation_form"]]
VARIANT["alg_params"] = ALG_PARAMS[VARIANT["algorithm_name"]]

RENDER = True


def get_env_from_name(name, ENV_SEED=None):
    if name == "oscillator":
        from envs.oscillator import oscillator as env

        env = env()
        env = env.unwrapped
    elif name == "Ex3_EKF":
        from envs.Ex3_EKF import Ex3_EKF as env

        env = env()
        env = env.unwrapped
    elif name == "Ex3_EKF_gyro":
        from envs.Ex3_EKF_gyro import Ex3_EKF as env

        env = env()
        env = env.unwrapped
    elif name == "Ex4_EKF":
        from envs.Ex4_EKF import Ex4_EKF as env

        env = env()
        env = env.unwrapped
    if ENV_SEED is not None:
        env.seed(ENV_SEED)
    return env


def get_train(name):
    if "LAC" in name:
        from LAC.LAC_V1 import train
    return train


def get_policy(name):

    if "LAC" in name:
        from LAC.LAC_V1 import LAC as build_func
    elif "LQR" in name:
        from LAC.lqr import LQR as build_func
    elif "MPC" in name:
        from LAC.MPC import MPC as build_func

    return build_func


def get_eval(name):
    if "LAC" in name or "SAC_cost" in name:
        from LAC.LAC_V1 import eval

    return eval
