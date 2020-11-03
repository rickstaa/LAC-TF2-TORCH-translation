"""A set of common utilities used within the algorithm code.
"""

import sys
import os.path as osp
import importlib
from collections import OrderedDict
import copy
import time

import numpy as np

from variant import ENVS_PARAMS, TRAIN_PARAMS, ENV_NAME, REL_PATH, USE_LYAPUNOV

# Script parameters
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def get_log_path(env_name=ENV_NAME, agent_name=None):
    """Retrieve model/results log path.

    Args:
        environment_name (str, optional): The name of the gym environment you are
            using. By default the value in the `variant.py` file is used.

        agent_name (str, optional): The name of the agent you are using. When no agent
            is supplied a agent name will be created.

    Returns:
        str: The model/results log path.
    """

    # Retrieve log_folder path
    log_folder = osp.join("./log", env_name.lower())

    # Create agent name if not supplied
    if not agent_name:
        alg_prefix = "LAC" if USE_LYAPUNOV else "SAC"
        agent_name = alg_prefix + time.strftime("%Y%m%d_%H%M")
        while 1:
            agent_folder = osp.join(log_folder, agent_name)

            # Check if created agent_name is valid
            if not osp.isdir(agent_folder):
                break
            else:  # Also add seconds if folder already exists
                agent_name = alg_prefix + time.strftime("%Y%m%d_%H%M%S")
    else:
        while 1:
            agent_folder = osp.join(log_folder, agent_name)

            # Check if supplied agent_name is valid
            if not osp.isdir(agent_folder):
                break
            else:  # Also add seconds if folder already exists
                agent_name = agent_name + "_" + time.strftime("%Y%m%d_%H%M%S")

    # Create log_path
    if REL_PATH:
        LOG_PATH = agent_folder
    else:
        dirname = osp.dirname(__file__)
        LOG_PATH = osp.abspath(osp.join(dirname, agent_folder))
        return LOG_PATH


def get_env_from_name(env_name, ENV_SEED):
    """Initializes the gym environment with the given name

    Args:
        env_name (str): The name of the gym environment you want to initialize.

    Returns:
        gym.Env: The gym environment.
    """

    # Retrieve Environment Parameters
    if env_name.lower() in ENVS_PARAMS.keys():
        env_params = ENVS_PARAMS[env_name.lower()]
        module_name = env_params["module_name"]
        class_name = env_params["class_name"]
    else:
        print(
            colorize(
                f"ERROR: Shutting down the training as the {env_name} environment "
                "was not specified in the `ENVS_PARAMS` dictionary. Please specify "
                "your environment in the `variant.py` file.",
                "red",
                bold=True,
            )
        )
        sys.exit(0)

    # Load the environment
    try:
        env = getattr(importlib.import_module(module_name), class_name)
        env = env()
        env = env.unwrapped  # Improve: It is better to register the environment
    except ModuleNotFoundError:
        print(
            colorize(
                (
                    f"ERROR: Shutting down the training as the {env_name} environment "
                    f"could not be found in module {module_name} and class "
                    f"{class_name}. Please check the `module_name` and `class_name` "
                    "variables in the `variant.py` file."
                ),
                "red",
                bold=True,
            )
        )
        sys.exit(0)

    # Set environment seed
    if ENV_SEED is not None:
        env.seed(ENV_SEED)
    return env


def evaluate_training_rollouts(paths):
    """Evaluates the performance of the policy in the training rollouts.

    Args:
       paths (collections.deque): The training paths.

    Returns:
        collections.OrderedDict: Dictionary with performance statistics.
    """
    data = copy.deepcopy(paths)
    if len(data) < 1:
        return None
    try:
        diagnostics = OrderedDict(
            (
                ("return", np.mean([np.sum(path["rewards"]) for path in data])),
                ("length", np.mean([len(p["rewards"]) for p in data])),
            )
        )
    except KeyError:
        return
    [path.pop("rewards") for path in data]
    for key in data[0].keys():
        result = [np.mean(path[key]) for path in data]
        diagnostics.update({key: np.mean(result)})
    return diagnostics


def training_evaluation(test_env, policy):
    """Evaluates the performance of the current policy in
    several test rollouts.

    Args:
        test_env (gym.Env): The test gym environment you want to use.

        policy (object): The current policy.

    Returns:
        collections.OrderedDict: Dictionary with performance statistics.
    """

    # Retrieve action space bounds from test_env and pass them to the policy
    a_lowerbound = test_env.action_space.low
    a_upperbound = test_env.action_space.high

    # Training setting
    total_cost = []
    episode_length = []
    die_count = 0
    seed_average_cost = []

    # Perform roolouts to evaluate performance
    for i in range(TRAIN_PARAMS["num_of_evaluation_paths"]):
        cost = 0
        if env.__class__.__name__.lower() == "ex3_ekf_gyro":
            s = env.reset(eval=True)
        else:
            s = env.reset()
        for j in range(ENV_PARAMS["max_ep_steps"]):
            if ENV_PARAMS["eval_render"]:
                env.render()
            a = policy.choose_action(s, True)
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
            s_, r, done, _ = env.step(action)
            cost += r
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            s = s_
            if done:
                seed_average_cost.append(cost)
                episode_length.append(j)
                if j < ENV_PARAMS["max_ep_steps"] - 1:
                    die_count += 1
                break

    # Save evaluation results
    total_cost.append(np.mean(seed_average_cost))
    total_cost_mean = np.average(total_cost)
    average_length = np.average(episode_length)

    # Return evaluation results
    diagnostic = {
        "average_test_return": total_cost_mean,
        "average_test_length": average_length,
    }
    return diagnostic


def validate_indices(indices, input_array):
    """Validates whether indices exist in the input_array.

    Args:
        indices (list): The indices you want to check.

        input_array (list): The input_array for which you want to check whether the
            indices exist.

    Returns:
        tuple: Tuple containing the valid and invalid indices (Valid indices, invalid
            indices).
    """
    if indices:
        invalid_indices = [
            idx for idx in indices if (idx > input_array.shape[0] or idx < 0)
        ]
        valid_indices = list(set(invalid_indices) ^ set(indices))
    else:
        invalid_indices = []
        valid_indices = list(range(0, (input_array.shape[0])))
    return valid_indices, invalid_indices
